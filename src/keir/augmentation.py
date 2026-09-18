import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .data import load_refined
from .io import stable_hash, write_json, write_jsonl
from .prompts import (
    SOCRATIC_TEACHER_SYSTEM,
    VARIANT_TEACHER_SYSTEM,
    variant_socratic_teacher_prompt,
    variant_teacher_prompt,
)
from .schemas import OriginalExample


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_gpt4o_augmentation(
    v0_path: str,
    requests_path: str,
    manifest_path: str,
    limit: Optional[int] = None,
    variants: int = 3,
    model: str = "gpt-4o-2024-08-06",
    stage_b_model: Optional[str] = None,
    near_duplicate_ratio: float = 0.82,
) -> Dict[str, Any]:

    originals = list(load_refined(v0_path))
    if limit is not None:
        originals = originals[:limit]
    if variants < 1:
        raise ValueError("variants must be positive")
    for item in originals:
        if not item.is_original or item.variant_index != 0:
            raise ValueError("augmentation input must contain only verified v0 originals")

    stage_a_parameters = {
        "model": model,
        "temperature": 0.7,
        "seed": 42,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    stage_b_parameters = {
        "model": stage_b_model or model,
        "temperature": 0.2,
        "seed": 42,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }
    requests = []
    for item in originals:
        source = OriginalExample(
            id=item.source_id,
            statement=item.statement,
            question=item.question,
            answer=item.answer,
            source="gsm8k",
            metadata=item.metadata,
        )
        prompt = variant_teacher_prompt(source, variants=variants)
        request_payload = {
            "stage": "A_answer_equivalent_question_variants",
            "source_id": item.source_id,
            "v0_id": item.id,
            "system": VARIANT_TEACHER_SYSTEM,
            "prompt": prompt,
            "parameters": stage_a_parameters,
        }
        request_payload["request_id"] = stable_hash(request_payload)
        requests.append(request_payload)
    write_jsonl(requests_path, requests)

    stage_b_preview = ""
    if originals:
        stage_b_preview = variant_socratic_teacher_prompt(
            originals[0], "<retained answer-equivalent variant question>"
        )
    manifest: Dict[str, Any] = {
        "status": "dry_run_no_api_calls",
        "input": v0_path,
        "input_sha256": _sha256(v0_path),
        "v0_originals_planned": len(originals),
        "stage_a_requests_written": len(requests),
        "stage_b_requests_written": 0,
        "original_chain_api_calls": 0,
        "variants_requested_per_original": variants,
        "maximum_variant_chain_calls_after_filtering": len(originals) * variants,
        "model_snapshot_requested": model,
        "stage_a_parameters": stage_a_parameters,
        "stage_b_parameters": stage_b_parameters,
        "filter_order": [
            "JSON schema and exact variant count",
            "nonempty natural question ending in ?",
            "exact question-only numeric multiset preservation",
            "target-type compatibility with the original question",
            "near-duplicate removal against original then retained variants in generation order",
            "stage-B chain generation only for retained variants",
            "Socratic schema and every subquestion ending in ?",
            "calculator-annotation and terminal-answer verification",
            "terminal target preservation",
        ],
        "near_duplicate": {
            "method": "normalized-character SequenceMatcher ratio",
            "primary_threshold": near_duplicate_ratio,
            "processing_order": "original first, then GPT variants in returned order",
        },
        "stage_b_strategy": (
            "adapt each retained variant from its verified official v0 Socratic chain; "
            "never regenerate or replace the original v0 chain"
        ),
        "stage_b_prompt_preview": stage_b_preview,
        "response_provenance_required": [
            "request_id",
            "requested_model",
            "response_model",
            "created timestamp",
            "finish_reason",
            "token usage",
            "raw response",
            "validation decision and rejection reason",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest
