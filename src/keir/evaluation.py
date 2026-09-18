import copy
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .answers import (
    SCORING_PROTOCOL,
    answers_equal,
    extract_benchmark_prediction,
    gsm8k_official_exact_match,
    normalize_answer,
    majority_answer,
)
from .io import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from .schemas import OriginalExample, Prediction, RefinedExample
from .costs import summarize_costs


def scorer_digest():

    from hashlib import sha256
    return sha256(Path(__file__).with_name("answers.py").read_bytes() + b"\n"
                  + Path(__file__).read_bytes()).hexdigest()


def _row_dataset(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    explicit = str(metadata.get("dataset", metadata.get("benchmark", ""))).lower()
    if explicit:
        return explicit
    example_id = str(row.get("id", "")).lower()
    return example_id.split("-", 1)[0]


def prediction_row_answer(row: Mapping[str, Any]) -> str:

    dataset = _row_dataset(row)
    solutions = row.get("solutions", [])


    if solutions and int(row.get("metadata", {}).get("samples", 0)) > 0:
        if len(solutions) != int(row["metadata"]["samples"]):
            raise ValueError("Self-consistency rescoring requires every sampled trace")
        return majority_answer(solutions, dataset)
    if solutions:
        return extract_benchmark_prediction(str(solutions[-1]), dataset)


    return str(row.get("prediction", ""))


def prediction_row_correct(row: Mapping[str, Any]) -> bool:
    answer = prediction_row_answer(row)
    if not answer:
        return False
    return answers_equal(answer, str(row["gold"]), dataset=_row_dataset(row))


def prediction_row_official_correct(row: Mapping[str, Any]) -> bool:
    dataset = _row_dataset(row)
    solutions = row.get("solutions", [])
    if (
        dataset == "gsm8k"
        and solutions
        and int(row.get("metadata", {}).get("samples", 0)) <= 1
    ):
        return gsm8k_official_exact_match(str(solutions[-1]), str(row["gold"]))
    return prediction_row_correct(row)


def evaluate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No predictions to evaluate")
    correct = []
    official_correct = []
    prompt_tokens = []
    unique_prompt_tokens = []
    generated_tokens = []
    latencies = []
    peak_memory = []
    flops = []
    predicted_subtasks = []
    retained_subtasks = []
    explicit_final_answers = []
    empty_predictions = []
    executor_calls = []
    verification_calls = []
    by_subject = defaultdict(list)
    for row in rows:
        if row.get("gold") is None:
            raise ValueError("Prediction %s has no gold answer" % row.get("id", "unknown"))
        hit = prediction_row_correct(row)
        correct.append(hit)
        official_correct.append(prediction_row_official_correct(row))
        stats = row.get("stats", {})
        prompt_tokens.append(int(stats.get("prompt_tokens", 0)))
        unique_prompt_tokens.append(stats.get("unique_prompt_tokens"))
        generated_tokens.append(int(stats.get("generated_tokens", 0)))
        latencies.append(float(stats.get("latency_seconds", 0.0)))
        peak_memory.append(int(stats.get("peak_memory_bytes", 0)))
        flops.append(float(stats.get("estimated_flops", 0.0)))
        predicted_subtasks.append(len(row.get("raw_subtasks", [])))
        retained_subtasks.append(len(row.get("retained_subtasks", [])))
        solutions = row.get("solutions", [])
        terminal_text = str(solutions[-1]) if solutions else ""
        explicit_final_answers.append(
            bool(re.search(r"(?:final\s+answer|answer)\s*(?:is|:|=)", terminal_text, re.I))
        )
        empty_predictions.append(not str(row.get("prediction", "")).strip())
        executor_calls.append(len(solutions))
        verification_calls.append(int(row.get("metadata", {}).get("verification_calls", 0)))
        subject = row.get("metadata", {}).get("subject", "")
        if subject:
            by_subject[subject].append(hit)
    total_latency = sum(latencies)
    reduction = 0.0
    if sum(predicted_subtasks):
        reduction = 1.0 - sum(retained_subtasks) / float(sum(predicted_subtasks))
    result = {
        "scoring_protocol": SCORING_PROTOCOL,
        "scorer_sha256": scorer_digest(),
        "instances": len(rows),
        "correct": sum(correct),
        "accuracy": 100.0 * sum(correct) / len(correct),
        "marker_string_em_correct": sum(official_correct),
        "marker_string_em_accuracy": (
            100.0 * sum(official_correct) / len(official_correct)
        ),
        "mean_prompt_tokens": statistics.mean(prompt_tokens),
        "mean_processed_prompt_tokens": statistics.mean(prompt_tokens),
        "mean_unique_prompt_tokens": (statistics.mean(unique_prompt_tokens)
                                      if all(value is not None for value in unique_prompt_tokens) else None),
        "prompt_token_accounting": "processed=sum_all_calls; unique=exact_full_prompt_once_per_problem",
        "flops_accounting": "legacy=loaded_parameters_times_processed_tokens; nominal_proxies=explicit_fields",
        "mean_generated_tokens": statistics.mean(generated_tokens),
        "mean_latency_seconds": statistics.mean(latencies),
        "throughput_problems_per_second": len(rows) / total_latency if total_latency > 0 else None,
        "peak_memory_bytes": max(peak_memory),
        "mean_estimated_flops": statistics.mean(flops),
        "mean_loaded_processed_lm_gflops": statistics.mean(flops) / 1e9,
        "mean_predicted_subtasks": statistics.mean(predicted_subtasks),
        "mean_retained_subtasks": statistics.mean(retained_subtasks),
        "executor_call_reduction": reduction,
        "explicit_final_answer_rate": sum(explicit_final_answers) / len(rows),
        "empty_prediction_rate": sum(empty_predictions) / len(rows),
        "invalid_answer_extraction_rate": (
            sum(not prediction_row_answer(row) for row in rows) / len(rows)
        ),
        "mean_executor_calls": statistics.mean(executor_calls),
        "mean_verification_calls": statistics.mean(verification_calls),
        "accuracy_per_100_generated_tokens": (
            (100.0 * sum(correct) / len(correct)) * 100.0 / statistics.mean(generated_tokens)
            if statistics.mean(generated_tokens) > 0
            else None
        ),
    }
    result.update(summarize_costs(row.get("stats", {}) for row in rows))
    if by_subject:
        result["subject_accuracy"] = {
            subject: 100.0 * sum(values) / len(values)
            for subject, values in sorted(by_subject.items())
        }
    return result


def evaluate_prediction_file(path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    from hashlib import sha256
    if output_path and Path(output_path).resolve() == Path(path).resolve():
        raise ValueError("Scoring output must not overwrite predictions")
    rows = list(read_jsonl(path))
    result = evaluate_prediction_rows(rows)
    identities = [row.get("metadata", {}).get("run_identity") for row in rows]
    if all(identity is not None for identity in identities):
        if any(identity != identities[0] for identity in identities):
            raise ValueError("Prediction rows contain different run identities")
        result.update({key: value for key, value in identities[0].items()
                       if key not in {"scoring_protocol", "scorer_sha256"}})
    result["predictions_sha256"] = sha256(Path(path).read_bytes()).hexdigest()
    result["answer_source"] = "raw_traces_or_stored_answer_when_traces_absent"
    if output_path:
        write_json(output_path, result)
    return result


def run_benchmark(
    reasoner: Any,
    input_path: str,
    output_path: str,
    mode: str = "keir",
    samples: int = 1,
    consolidate: bool = True,
    limit: Optional[int] = None,
    seed: int = 42,
    resume: bool = False,
    training_seed: Optional[int] = None,
    configuration_id: Optional[str] = None,
) -> Dict[str, Any]:
    from .inference import solve_cot

    examples = [OriginalExample.from_dict(row) for row in read_jsonl(input_path)]
    if limit is not None:
        examples = examples[:limit]
    sources = {example.source for example in examples}
    if len(sources) != 1 or not all(sources):
        raise ValueError("One nonempty benchmark is required per evaluation run")
    runtime = getattr(reasoner, "runtime", reasoner)
    recorded_seed = getattr(runtime, "training_seed", None)
    if training_seed is not None and recorded_seed is not None and training_seed != recorded_seed:
        raise ValueError("Declared training seed differs from adapter training metadata")
    training_seed = training_seed if training_seed is not None else recorded_seed
    run_identity = {"benchmark": next(iter(sources)), "training_seed": training_seed,
                    "scoring_protocol": SCORING_PROTOCOL,
                    "scorer_sha256": scorer_digest(),
                    "inference_seed": seed, "configuration_id": configuration_id,
                    "input_sha256": stable_hash([example.to_dict() for example in examples]),
                    "mode": mode, "samples": samples, "consolidation": bool(consolidate and getattr(reasoner, "consolidator", None) is not None)}
    partial_path = str(Path(output_path).with_suffix(".partial.jsonl"))
    rows: List[Dict[str, Any]] = []
    if resume and Path(partial_path).exists():
        rows = list(read_jsonl(partial_path))
        if len(rows) > len(examples):
            raise ValueError(
                "Partial prediction count %d exceeds input count %d"
                % (len(rows), len(examples))
            )
        for index, row in enumerate(rows):
            example = examples[index]
            if str(row.get("id", "")) != example.id:
                raise ValueError(
                    "Partial prediction %d has id %s; expected %s"
                    % (index, row.get("id", ""), example.id)
                )
            if row.get("metadata", {}).get("run_identity") != run_identity:
                raise ValueError("Partial predictions belong to a different or unidentified run")
            if not answers_equal(
                str(row.get("gold", "")), example.answer, dataset=example.source
            ):
                raise ValueError("Partial prediction gold mismatch for %s" % example.id)
    for index in range(len(rows), len(examples)):
        example = examples[index]
        if mode == "keir":
            prediction = reasoner.solve(
                example_id=example.id,
                statement=example.statement,
                question=example.question,
                gold=example.answer,
                consolidate=consolidate,
                seed=seed + index * 100,
                dataset=example.source,
            )
        elif mode == "cot":
            runtime = getattr(reasoner, "runtime", reasoner)
            prediction = solve_cot(
                runtime=runtime,
                example_id=example.id,
                statement=example.statement,
                question=example.question,
                samples=samples,
                max_new_tokens=512 if samples > 1 or example.source == "math" else 256,
                seed=seed + index * 100,
                gold=example.answer,
                dataset=example.source,
            )
        else:
            raise ValueError("mode must be keir or cot")
        row = prediction.to_dict()
        row["metadata"].update(example.metadata)
        row["metadata"]["dataset"] = example.source
        row["metadata"]["run_identity"] = run_identity
        rows.append(row)


        write_jsonl(partial_path, rows)
    write_jsonl(output_path, rows)
    metrics = evaluate_prediction_rows(rows)
    metrics.update(run_identity)
    metrics["latency_scope"] = "end_to_end_single_problem"
    metrics["timing_protocol"] = "one_pass_no_discarded_warmup"
    write_json(str(Path(output_path).with_suffix(".metrics.json")), metrics)
    Path(partial_path).unlink(missing_ok=True)
    return metrics


def mcnemar(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
    continuity_corrected: bool = True,
) -> Dict[str, Any]:

    first = {str(row["id"]): row for row in first_rows}
    second = {str(row["id"]): row for row in second_rows}
    if len(first) != len(first_rows) or len(second) != len(second_rows):
        raise ValueError("Duplicate prediction IDs are not valid paired observations")
    if set(first) != set(second):
        missing_first = sorted(set(second) - set(first))
        missing_second = sorted(set(first) - set(second))
        raise ValueError(
            "Prediction IDs differ; missing first=%d, missing second=%d"
            % (len(missing_first), len(missing_second))
        )
    n01 = 0
    n10 = 0
    for example_id in first:
        gold_first = str(first[example_id]["gold"])
        gold_second = str(second[example_id]["gold"])
        dataset = _row_dataset(first[example_id]) or _row_dataset(second[example_id])
        if not answers_equal(gold_first, gold_second, dataset=dataset):
            raise ValueError("Gold answer mismatch for %s" % example_id)
        first_correct = prediction_row_correct(first[example_id])
        second_correct = prediction_row_correct(second[example_id])
        if second_correct and not first_correct:
            n01 += 1
        elif first_correct and not second_correct:
            n10 += 1
    discordant = n01 + n10
    if discordant == 0:
        statistic, p_value = 0.0, 1.0
    else:
        correction = 1 if continuity_corrected else 0
        statistic = (max(0, abs(n01 - n10) - correction) ** 2) / float(discordant)

        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "instances": len(first),
        "n01": n01,
        "n10": n10,
        "discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "continuity_corrected": continuity_corrected,
    }


def mcnemar_files(first_path: str, second_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    result = mcnemar(list(read_jsonl(first_path)), list(read_jsonl(second_path)))
    if output_path:
        write_json(output_path, result)
    return result


def aggregate_seeds(
    metric_files: Sequence[str], output_path: Optional[str] = None
) -> Dict[str, Any]:
    metrics = []
    for path in metric_files:
        from .io import read_json

        metrics.append(read_json(path))
    if not metrics:
        raise ValueError("No metric files supplied")
    _validate_metric_group(metrics)
    if len({item["benchmark"] for item in metrics}) != 1:
        raise ValueError("aggregate-seeds requires one benchmark; use aggregate-paper for the full grid")
    numeric_keys = sorted(
        key
        for key in set.intersection(*(set(item) for item in metrics))
        if key not in {"training_seed", "inference_seed"}
        and all(isinstance(item[key], (int, float)) and not isinstance(item[key], bool) for item in metrics)
    )
    summary = {"seeds": len(metrics), "metrics": {}}
    for key in numeric_keys:
        values = [float(item[key]) for item in metrics]
        summary["metrics"][key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    if output_path:
        write_json(output_path, summary)
    return summary


PAPER_BENCHMARKS = ("asdiv", "gsm8k", "math", "mawps", "svamp")
PAPER_SEEDS = (42, 123, 456)


def _validate_metric_group(metrics):
    seen = set()
    identities = set()
    sources = {}
    for item in metrics:
        if item.get("scoring_protocol") != SCORING_PROTOCOL or item.get("scorer_sha256") != scorer_digest():
            raise ValueError("Rescore predictions with the current scoring_protocol before aggregation")
        seed, benchmark = item.get("training_seed"), item.get("benchmark")
        if not isinstance(seed, int) or isinstance(seed, bool) or not benchmark:
            raise ValueError("Metrics need explicit benchmark and training_seed; inference seed is not a training seed")
        if not item.get("configuration_id") or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("input_sha256", ""))):
            raise ValueError("Metrics need configuration_id and input_sha256 from evaluation")
        if (seed, benchmark) in seen:
            raise ValueError("Duplicate training_seed/benchmark pair")
        seen.add((seed, benchmark))
        identities.add((item["configuration_id"], item.get("mode"), item.get("samples"), item.get("consolidation")))
        value = item.get("accuracy")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("Accuracy must be a finite percentage in [0, 100]")
        count = item.get("instances")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Metrics need a positive instance count")
        source = (item["input_sha256"], count)
        if benchmark in sources and sources[benchmark] != source:
            raise ValueError("Benchmark contents/counts differ across training seeds: " + benchmark)
        sources[benchmark] = source
    if len(identities) != 1:
        raise ValueError("Do not aggregate different model or evaluation configurations")


def aggregate_paper(metric_files, output_path=None):

    metrics = [read_json(path) for path in metric_files]
    if not metrics:
        raise ValueError("No metric files supplied")
    _validate_metric_group(metrics)
    indexed = {(item["training_seed"], item["benchmark"]): item for item in metrics}
    expected = {(seed, benchmark) for seed in PAPER_SEEDS for benchmark in PAPER_BENCHMARKS}
    if set(indexed) != expected:
        raise ValueError("Expected exactly the 3 seeds (42,123,456) x 5 paper benchmarks; missing=%s extra=%s"
                         % (sorted(expected - set(indexed)), sorted(set(indexed) - expected)))
    per_seed = {str(seed): {"benchmark_accuracy": {b: indexed[seed, b]["accuracy"] for b in PAPER_BENCHMARKS},
                           "macro_accuracy": statistics.mean(indexed[seed, b]["accuracy"] for b in PAPER_BENCHMARKS)}
                for seed in PAPER_SEEDS}
    macro = [per_seed[str(seed)]["macro_accuracy"] for seed in PAPER_SEEDS]
    result = {"configuration_id": metrics[0]["configuration_id"], "training_seeds": list(PAPER_SEEDS),
              "benchmarks": list(PAPER_BENCHMARKS), "per_seed": per_seed,
              "instances_per_benchmark": {b: indexed[PAPER_SEEDS[0], b]["instances"] for b in PAPER_BENCHMARKS},
              "macro_accuracy": {"mean": statistics.mean(macro), "std": statistics.stdev(macro), "values": macro},
              "benchmark_accuracy": {b: {"mean": statistics.mean(indexed[s, b]["accuracy"] for s in PAPER_SEEDS),
                                          "std": statistics.stdev(indexed[s, b]["accuracy"] for s in PAPER_SEEDS)}
                                     for b in PAPER_BENCHMARKS},
              "aggregation_order": "unweighted benchmark mean within each seed, then mean and sample SD across seeds"}
    if output_path:
        write_json(output_path, result)
    return result
