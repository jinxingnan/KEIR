import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data import _number_signature, _text_key, load_refined
from .io import read_jsonl, stable_hash, write_json, write_jsonl
from .socratic import _target_type
from .target_review import check_target


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```")
        value = value.removesuffix("```").strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def filter_stage_a_response(
    original_question: str,
    response: Mapping[str, Any],
    variants: int = 3,
    near_duplicate_ratio: float = 0.82,
) -> List[Dict[str, Any]]:
    questions = response.get("rewritten_questions")
    if not isinstance(questions, list) or len(questions) != variants:
        raise ValueError("expected exactly %d rewritten_questions" % variants)
    original_numbers = Counter(_number_signature(original_question))
    original_type = _target_type(original_question)
    seen = [_text_key(original_question)]
    decisions = []
    for index, value in enumerate(questions, 1):
        question = str(value).strip()
        reason = "retained"
        maximum_similarity = 0.0
        target_check = check_target(original_question, question)
        if not question or not question.endswith("?"):
            reason = "not_natural_question"
        elif Counter(_number_signature(question)) != original_numbers:
            reason = "question_numbers_changed"
        elif target_check["conflicts"]:
            reason = "target_conflict"
        else:
            candidate_type = _target_type(question)
            if original_type and candidate_type and original_type != candidate_type:
                reason = "target_type_changed"
            else:
                key = _text_key(question)
                maximum_similarity = max(
                    SequenceMatcher(None, key, previous).ratio() for previous in seen
                )
                if maximum_similarity >= near_duplicate_ratio:
                    reason = "near_duplicate"
                else:
                    seen.append(key)
        decisions.append(
            {
                "variant_index": index,
                "question": question,
                "retained": reason == "retained",
                "reason": reason,
                "maximum_prior_similarity": maximum_similarity,
                "target_check": target_check,
            }
        )
    return decisions


def _usage_dict(usage: Any) -> Dict[str, Optional[int]]:
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
        "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
    }


def run_teacher_stage_a(
    requests_path: str,
    v0_path: str,
    responses_path: str,
    candidates_path: str,
    stats_path: str,
    cache_dir: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    workers: int = 8,
    timeout: float = 240.0,
    max_retries: int = 3,
    variants: int = 3,
    near_duplicate_ratio: float = 0.82,
) -> Dict[str, Any]:

    if workers < 1:
        raise ValueError("workers must be positive")
    api_key = os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("TEACHER_API_KEY is not set")
    model = model or os.environ.get("TEACHER_STAGE_A_MODEL")
    if not model:
        raise RuntimeError("set TEACHER_STAGE_A_MODEL or pass model explicitly")
    base_url = base_url or os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1")
    if not base_url:
        raise RuntimeError("set TEACHER_BASE_URL or pass base_url explicitly")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI-compatible SDK is required") from exc

    requests = list(read_jsonl(requests_path))
    originals = {item.id: item for item in load_refined(v0_path)}
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    def execute(request: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = stable_hash({"request": request, "model": model, "base_url": base_url,
                                  "temperature": 0.7, "max_tokens": 512, "seed": 42})
        target = cache / (request_id + ".json")
        if target.exists():
            with target.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("ok"):
                return cached
        last_error = ""
        for attempt in range(1, max_retries + 1):
            started = time.time()
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=0,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": request["system"]},
                        {"role": "user", "content": request["prompt"]},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                    seed=42,
                )
                raw = response.choices[0].message.content or ""
                parsed = _parse_json_object(raw)
                record = {
                    "ok": True,
                    "request_id": request_id,
                    "source_id": request["source_id"],
                    "v0_id": request["v0_id"],
                    "requested_model": model,
                    "response_model": str(getattr(response, "model", model)),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "created": getattr(response, "created", None),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": _usage_dict(response.usage),
                    "latency_seconds": time.time() - started,
                    "attempt": attempt,
                    "raw_response": raw,
                    "response": parsed,
                }
                write_json(str(target), record)
                return record
            except Exception as exc:
                last_error = type(exc).__name__
                if attempt < max_retries:
                    time.sleep(min(30.0, (2**attempt) + random.random()))
        return {
            "ok": False,
            "request_id": request_id,
            "source_id": request.get("source_id"),
            "v0_id": request.get("v0_id"),
            "requested_model": model,
            "error": last_error,
            "attempt": max_retries,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        responses = list(executor.map(execute, requests))
    write_jsonl(responses_path, responses)

    candidates: List[Dict[str, Any]] = []
    reasons = Counter()
    invalid_responses = Counter()
    for response in responses:
        if not response.get("ok"):
            invalid_responses[str(response.get("error", "unknown"))] += 1
            continue
        original = originals[str(response["v0_id"])]
        try:
            decisions = filter_stage_a_response(
                original.question,
                response["response"],
                variants=variants,
                near_duplicate_ratio=near_duplicate_ratio,
            )
        except Exception as exc:
            invalid_responses["%s: %s" % (type(exc).__name__, exc)] += 1
            continue
        for decision in decisions:
            reasons[decision["reason"]] += 1
            candidates.append(
                dict(
                    decision,
                    source_id=original.source_id,
                    v0_id=original.id,
                    original_question=original.question,
                    request_id=response["request_id"],
                )
            )
    write_jsonl(candidates_path, candidates)

    successful = [item for item in responses if item.get("ok")]
    usage_keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    usage = {
        key: sum(int(item.get("usage", {}).get(key) or 0) for item in successful)
        for key in usage_keys
    }
    stats: Dict[str, Any] = {
        "stage": "A_answer_equivalent_question_variants",
        "model_requested": model,
        "response_models": dict(Counter(item["response_model"] for item in successful)),
        "requests": len(requests),
        "successful_requests": len(successful),
        "failed_requests": len(requests) - len(successful),
        "candidate_questions": len(candidates),
        "retained_questions": reasons["retained"],
        "reasons": dict(sorted(reasons.items())),
        "invalid_responses": dict(invalid_responses),
        "near_duplicate_ratio": near_duplicate_ratio,
        "seed": 42,
        "workers": workers,
        "usage": usage,
        "mean_latency_seconds": (
            sum(float(item["latency_seconds"]) for item in successful) / len(successful)
            if successful
            else None
        ),
        "stage_b_calls": 0,
    }
    write_json(stats_path, stats)
    return stats
