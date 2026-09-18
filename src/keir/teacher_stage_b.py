import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .answers import answers_equal
from .data import load_refined
from .teacher_stage_a import _parse_json_object, _usage_dict
from .io import read_jsonl, stable_hash, write_json, write_jsonl
from .prompts import SOCRATIC_TEACHER_SYSTEM, variant_socratic_teacher_prompt
from .schemas import RefinedExample, SchemaError, SubtaskSolution
from .target_review import check_target
from .socratic import (
    SocraticValidationError,
    _annotation_values,
    _target_type,
    _validate_unannotated_calculations,
    step_answer_matches_gold,
)


def validate_stage_b_response(
    original: RefinedExample,
    variant_question: str,
    response: Mapping[str, Any],
) -> Tuple[List[SubtaskSolution], str]:
    target_check = check_target(original.question, variant_question)
    if target_check["conflicts"]:
        raise SocraticValidationError("variant_target_conflict", ", ".join(target_check["conflicts"]))
    steps = response.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SocraticValidationError("empty_chain")
    if len(steps) > 10:
        raise SocraticValidationError("too_many_steps", str(len(steps)))
    try:
        chain = [SubtaskSolution.from_dict(step) for step in steps]
    except SchemaError as exc:
        raise SocraticValidationError("invalid_chain_step", str(exc)) from exc
    for index, step in enumerate(chain, 1):
        if not step.task.endswith("?"):
            raise SocraticValidationError(
                "non_interrogative_subquestion", "step %d: %s" % (index, step.task)
            )
        _annotation_values(step.solution)
        _validate_unannotated_calculations(step.solution)
    normalized_subquestions = [
        " ".join(re.findall(r"[a-z0-9]+", step.task.lower())) for step in chain
    ]
    if len(set(normalized_subquestions)) != len(normalized_subquestions):
        raise SocraticValidationError("duplicate_subquestion")
    final_answer = str(response.get("final_answer", "")).strip()
    if not answers_equal(final_answer, original.answer, dataset="gsm8k"):
        raise SocraticValidationError(
            "final_answer_mismatch", "%s != %s" % (final_answer, original.answer)
        )
    context = "%s %s" % (original.statement, variant_question)
    if not step_answer_matches_gold(chain[-1].solution, original.answer, context):
        raise SocraticValidationError(
            "terminal_answer_mismatch", chain[-1].solution
        )
    original_type = _target_type(variant_question)
    terminal_type = _target_type(chain[-1].task)
    if original_type and terminal_type and original_type != terminal_type:
        raise SocraticValidationError(
            "terminal_target_type_mismatch",
            "%s != %s" % (original_type, terminal_type),
        )
    for question in (original.question, variant_question):
        terminal_check = check_target(question, chain[-1].task)
        if terminal_check["conflicts"]:
            raise SocraticValidationError("terminal_target_conflict", ", ".join(terminal_check["conflicts"]))
    return chain, final_answer


def run_teacher_stage_b(
    candidates_path: str,
    v0_path: str,
    responses_path: str,
    accepted_path: str,
    rejected_path: str,
    stats_path: str,
    cache_dir: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    workers: int = 16,
    timeout: float = 300.0,
    max_retries: int = 3,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    api_key = os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("TEACHER_API_KEY is not set")
    model = model or os.environ.get("TEACHER_STAGE_B_MODEL")
    if not model:
        raise RuntimeError("set TEACHER_STAGE_B_MODEL or pass model explicitly")
    base_url = base_url or os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1")
    if not base_url:
        raise RuntimeError("set TEACHER_BASE_URL or pass base_url explicitly")
    from openai import OpenAI

    candidates = [row for row in read_jsonl(candidates_path) if row.get("retained")]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = candidates[:limit]
    originals = {item.id: item for item in load_refined(v0_path)}
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        original = originals[str(candidate["v0_id"])]
        variant_index = int(candidate["variant_index"])
        prompt = variant_socratic_teacher_prompt(
            original, str(candidate["question"])
        )
        request_id = stable_hash(
            {
                "stage": "B_socratic_chain_adaptation_release_v1",
                "model": model,
                "base_url": base_url,
                "v0_id": original.id,
                "variant_index": candidate["variant_index"],
                "question": candidate["question"],
                "system": SOCRATIC_TEACHER_SYSTEM,
                "prompt": prompt,
                "seed": 42,
                "temperature": 0.2,
            }
        )
        jobs.append((candidate, original, prompt, request_id))

    def execute(job: Tuple[Mapping[str, Any], RefinedExample, str, str]) -> Dict[str, Any]:
        candidate, original, prompt, request_id = job
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
                        {"role": "system", "content": SOCRATIC_TEACHER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1600,
                    response_format={"type": "json_object"},
                    seed=42,
                )
                raw = response.choices[0].message.content or ""
                parsed = _parse_json_object(raw)
                record = {
                    "ok": True,
                    "request_id": request_id,
                    "source_id": original.source_id,
                    "v0_id": original.id,
                    "variant_index": candidate["variant_index"],
                    "variant_question": candidate["question"],

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
            "source_id": original.source_id,
            "v0_id": original.id,
            "variant_index": candidate["variant_index"],
            "variant_question": candidate["question"],

            "requested_model": model,
            "error": last_error,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        responses = list(executor.map(execute, jobs))
    write_jsonl(responses_path, responses)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reasons = Counter()
    for response in responses:
        if not response.get("ok"):
            reasons["api_or_json_failure"] += 1
            rejected.append(response)
            continue
        original = originals[str(response["v0_id"])]
        try:
            chain, _ = validate_stage_b_response(
                original,
                str(response["variant_question"]),
                response["response"],
            )
            item = RefinedExample(
                id="%s-v%d" % (original.source_id, int(response["variant_index"])),
                source_id=original.source_id,
                variant_index=int(response["variant_index"]),
                statement=original.statement,
                question=str(response["variant_question"]),
                answer=original.answer,
                chain=chain,
                target_quantity=chain[-1].task,
                is_original=False,
                metadata={
                    "dataset": "gsm8k",
                    "base_v0_id": original.id,
                    "original_question": original.question,
                    "target_review": {"status": "pending", "semantic_equivalence_verified": False},
                    "target_check": check_target(original.question, str(response["variant_question"])),
                    "teacher_protocol": "external_teacher_official_chain_release_v1",
                    "requested_model": response["requested_model"],
                    "response_model": response["response_model"],
                    "request_id": response["request_id"],
                    "stage_a_variant_index": response["variant_index"],

                    "validation": "strict_variant_v1",
                },
            )
            accepted.append(item.to_dict())
            reasons["retained"] += 1
        except SocraticValidationError as exc:
            reasons[exc.code] += 1
            rejected.append(
                {
                    "request_id": response["request_id"],
                    "v0_id": response["v0_id"],
                    "variant_index": response["variant_index"],
                    "variant_question": response["variant_question"],
                    "reason": exc.code,
                    "detail": exc.detail,
                    "response": response["response"],
                }
            )
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)

    successful = [item for item in responses if item.get("ok")]
    usage_keys = (
        "prompt_tokens", "completion_tokens", "total_tokens",
        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    )
    usage = {
        key: sum(int(item.get("usage", {}).get(key) or 0) for item in successful)
        for key in usage_keys
    }
    stats: Dict[str, Any] = {
        "stage": "B_socratic_chain_adaptation_release_v1",
        "model_requested": model,
        "response_models": dict(Counter(item["response_model"] for item in successful)),
        "requests": len(jobs),
        "successful_requests": len(successful),
        "failed_requests": len(jobs) - len(successful),
        "retained_chains": len(accepted),
        "rejected_chains": len(rejected),
        "reasons": dict(sorted(reasons.items())),
        "seed": 42,
        "workers": workers,
        "usage": usage,
        "mean_latency_seconds": (
            sum(float(item["latency_seconds"]) for item in successful) / len(successful)
            if successful else None
        ),
        "original_chain_api_calls": 0,
    }
    write_json(stats_path, stats)
    return stats
