import statistics
from .schemas import OriginalExample
from .io import read_jsonl, write_json
from .inference import solve_cot
from .costs import request_costs, summarize_costs


def measure(reasoner, input_path, output_path, warmup=10, repetitions=3,
            limit=None, seed=42, mode="keir", samples=1):
    examples = [OriginalExample.from_dict(x) for x in read_jsonl(input_path)]
    if limit is not None:
        examples = examples[:limit]
    if not examples or warmup < 0 or repetitions < 1:
        raise ValueError("Need nonempty examples, nonnegative warmup and positive repetitions")

    def solve(example, index):
        common = dict(example_id=example.id, statement=example.statement, question=example.question,
                      gold=example.answer, seed=seed + index * 100, dataset=example.source)
        if mode == "keir":
            return reasoner.solve(**common)
        return solve_cot(getattr(reasoner, "runtime", reasoner), samples=samples,
                         max_new_tokens=512 if samples > 1 or example.source == "math" else 256, **common)

    for i in range(warmup):
        solve(examples[i % len(examples)], i % len(examples))
    runs, all_stats = [], []
    for run in range(repetitions):
        timings, peaks, ids, costs = [], [], [], []
        for i, example in enumerate(examples):
            prediction = solve(example, i)
            timings.append(prediction.stats.latency_seconds)
            peaks.append(prediction.stats.peak_memory_bytes)
            ids.append(example.id)
            all_stats.append(prediction.stats.to_dict())
            costs.append({"processed_prompt_tokens": prediction.stats.prompt_tokens,
                          "unique_prompt_tokens": prediction.stats.unique_prompt_tokens,
                          "generated_tokens": prediction.stats.generated_tokens,
                          "estimated_lm_flops": prediction.stats.estimated_flops,
                          "loaded_parameters": prediction.stats.loaded_parameters,
                          "nominal_parameters": prediction.stats.nominal_parameters,
                          **request_costs(prediction.stats.to_dict())})
        runs.append({"run": run + 1, "ids": ids, "latency_seconds": timings,
                     "peak_memory_bytes": peaks, "costs": costs, "mean_latency_seconds": statistics.mean(timings)})
    means = [r["mean_latency_seconds"] for r in runs]
    result = {"scope": "end_to_end_single_problem", "batch_size": 1,
              "discarded_warmup_requests": warmup, "repetitions": repetitions,
              "instances_per_run": len(examples), "seed": seed,
              "mean_latency_seconds": statistics.mean(means),
              "std_latency_seconds": statistics.stdev(means) if len(means) > 1 else 0.0,
              "peak_memory_bytes": max(max(r["peak_memory_bytes"]) for r in runs), "runs": runs}
    costs = [cost for run in runs for cost in run["costs"]]
    for key in ("processed_prompt_tokens", "unique_prompt_tokens", "generated_tokens", "estimated_lm_flops"):
        values = [cost[key] for cost in costs]
        result["mean_" + key] = statistics.mean(values) if all(v is not None for v in values) else None
    result["prompt_token_accounting"] = "processed=sum_all_calls; unique=exact_full_prompt_once_per_problem"
    result["flops_accounting"] = "legacy=loaded_parameters_times_processed_tokens; nominal_proxies=explicit_fields"
    result.update(summarize_costs(all_stats))
    write_json(output_path, result)
    return result
