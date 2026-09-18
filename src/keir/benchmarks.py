import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .answers import extract_answer
from .data import parse_gsm_answer, split_statement_question
from .io import read_jsonl, write_json, write_jsonl
from .schemas import OriginalExample


HF_SPECS = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "test"},
    "math": {"path": "EleutherAI/hendrycks_math", "name": None, "split": "test"},
    "svamp": {"path": "arkilpatel/SVAMP", "name": None, "split": "full_challenge_set"},
    "asdiv": {"path": "EleutherAI/asdiv", "name": "asdiv", "split": "validation"},
}

MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus")

EXPECTED_TEST_SIZES = {
    "asdiv": 2305,
    "gsm8k": 1319,
    "math": 5000,
    "mawps": 2373,
    "svamp": 1000,
}


def _first(record: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    lowered = {str(key).lower(): key for key in record}
    for key in keys:
        actual = lowered.get(key.lower())
        if actual is not None and record[actual] is not None:
            return record[actual]
    return default


def normalize_benchmark_record(
    benchmark: str, record: Mapping[str, Any], index: int
) -> OriginalExample:
    name = benchmark.lower()
    if name == "gsm8k":
        statement, question = split_statement_question(str(record["question"]))
        rationale, answer = parse_gsm_answer(str(record["answer"]))
        metadata = {}
    elif name == "math":
        problem = str(_first(record, ("problem", "question")))
        statement, question = split_statement_question(problem)
        rationale = str(_first(record, ("solution", "rationale", "answer")))
        answer = extract_answer(rationale, "math")
        metadata = {"subject": str(_first(record, ("type", "subject"), ""))}
    elif name == "svamp":
        statement = str(_first(record, ("Body", "body", "context"))).strip()
        question = str(_first(record, ("Question", "question"))).strip()
        answer = str(_first(record, ("Answer", "answer", "label"))).strip()
        rationale = str(_first(record, ("Equation", "equation", "rationale"))).strip()
        metadata = {}
    elif name in ("asdiv", "mawps"):
        statement = str(
            _first(record, ("Body", "body", "context", "narrative", "problem"))
        ).strip()
        question = str(_first(record, ("Question", "question", "query"))).strip()
        if not question and statement:
            statement, question = split_statement_question(statement)
        answer = str(_first(record, ("Answer", "answer", "label", "lSolutions"))).strip()
        if isinstance(_first(record, ("lSolutions",), ""), list):
            solutions = _first(record, ("lSolutions",), [])
            answer = str(solutions[0]) if solutions else ""
        rationale = str(_first(record, ("Formula", "formula", "equation", "lEquations"))).strip()
        metadata = {}
    else:
        raise ValueError("Unsupported benchmark: %s" % benchmark)
    if not statement:
        combined = str(_first(record, ("problem", "question", "Body"), "")).strip()
        statement, question = split_statement_question(combined)
    if not question:
        question = "What is the final answer?"
    if not answer:
        raise ValueError("%s record %d has no answer" % (benchmark, index))
    metadata["dataset"] = name
    source_id = _first(record, ("ID", "id", "iIndex", "_id"), "")
    if source_id != "":
        metadata["source_record_id"] = str(source_id)
    return OriginalExample(
        id="%s-test-%05d" % (name, index),
        statement=statement,
        question=question,
        answer=answer,
        rationale=rationale,
        source=name,
        metadata=metadata,
    )


def download_benchmark(benchmark: str, output_path: str, revision: str = "main",
                       trust_remote_code: bool = False) -> int:
    name = benchmark.lower()
    if name == "mawps":
        raise ValueError(
            "The garrethlee/MAWPS test split has 355 records; this configuration expects 2373. "
            "Import your MAWPS evaluation file with normalize-benchmark, then run validate-benchmark."
        )
    if name == "svamp":
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
            raise ValueError("Invalid SVAMP revision")
        if re.fullmatch(r"[0-9a-f]{40}", revision):
            resolved = revision
        else:
            with urlopen("https://api.github.com/repos/arkilpatel/SVAMP/commits/" + revision, timeout=60) as response:
                resolved = json.load(response)["sha"]
        url = "https://raw.githubusercontent.com/arkilpatel/SVAMP/" + resolved + "/SVAMP.json"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        records = json.loads(raw)
        rows = [normalize_benchmark_record(name, row, i).to_dict() for i, row in enumerate(records)]
        manifest = {"source": url, "resolved_revision": resolved, "split": "full_challenge_set",
                    "raw_sha256": hashlib.sha256(raw).hexdigest()}
        return _write_verified(name, output_path, rows, manifest)
    try:
        from datasets import load_dataset, concatenate_datasets
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("datasets is required for benchmark download") from exc
    if name not in HF_SPECS:
        raise ValueError("Unknown benchmark: %s" % benchmark)
    spec = HF_SPECS[name]
    resolved = HfApi().dataset_info(spec["path"], revision=revision).sha
    configs = MATH_CONFIGS if name == "math" else [spec["name"]]
    parts = [load_dataset(spec["path"], config, split=spec["split"], revision=resolved,
                          trust_remote_code=trust_remote_code) for config in configs]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    rows = [
        normalize_benchmark_record(name, record, index).to_dict()
        for index, record in enumerate(dataset)
    ]
    manifest = {"source": spec["path"], "configs": list(configs), "split": spec["split"],
                "resolved_revision": resolved, "dataset_fingerprint": dataset._fingerprint}
    return _write_verified(name, output_path, rows, manifest)


def _write_verified(name, output_path, rows, manifest):
    if len(rows) != EXPECTED_TEST_SIZES[name]:
        raise ValueError("Unexpected %s size: %d, expected %d" % (name, len(rows), EXPECTED_TEST_SIZES[name]))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate normalized IDs")
    count = write_jsonl(output_path, rows)
    manifest.update(benchmark=name, count=count,
                    sha256=hashlib.sha256(Path(output_path).read_bytes()).hexdigest())
    write_json(output_path + ".manifest.json", manifest)
    return count


def normalize_local_benchmark(benchmark: str, input_path: str, output_path: str,
                              source_url: Optional[str] = None, source_revision: Optional[str] = None,
                              source_split: Optional[str] = None) -> int:
    source = Path(input_path)
    if source.suffix.lower() == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("A local JSON benchmark must contain an array of records")
    else:
        records = list(read_jsonl(input_path))
    rows = [normalize_benchmark_record(benchmark, record, index).to_dict()
            for index, record in enumerate(records)]
    manifest = {"source": source_url or "user_supplied_file", "source_filename": Path(input_path).name,
                "raw_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "resolved_revision": None, "declared_revision": source_revision,
                "split": source_split or "user_supplied_evaluation_set",
                "source_metadata_basis": "user_declaration_not_remote_verification"}
    return _write_verified(benchmark.lower(), output_path, rows, manifest)


def validate_benchmark_file(
    benchmark: str, path: str, strict_size: bool = True
) -> Dict[str, Any]:
    examples = [OriginalExample.from_dict(row) for row in read_jsonl(path)]
    expected = EXPECTED_TEST_SIZES[benchmark.lower()]
    result = {
        "benchmark": benchmark.lower(),
        "path": str(Path(path)),
        "count": len(examples),
        "expected_count": expected,
        "ids_unique": len({item.id for item in examples}) == len(examples),
        "empty_statements": sum(not item.statement for item in examples),
        "empty_questions": sum(not item.question for item in examples),
        "empty_answers": sum(not item.answer for item in examples),
    }
    result["valid"] = (
        result["ids_unique"]
        and result["empty_statements"] == 0
        and result["empty_questions"] == 0
        and result["empty_answers"] == 0
        and (not strict_size or len(examples) == expected)
    )
    return result
