import math
import statistics

COST_PROTOCOL = "keir-lm-token-proxy-v2"


def estimate_lm_cost(parameters, prefill_tokens, generated_tokens, prefill_accounting,
                     parameter_basis="nominal_backbone"):

    if prefill_accounting not in {"processed", "exact_full_prompt_once", "reported"}:
        raise ValueError("Unknown prefill accounting")
    if parameter_basis not in {"nominal_backbone", "loaded_model"}:
        raise ValueError("Unknown parameter basis")
    for name, value in (("parameters", parameters), ("prefill_tokens", prefill_tokens),
                        ("generated_tokens", generated_tokens)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + " must be finite and nonnegative")
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    flops = 2.0 * parameters * (prefill_tokens + generated_tokens)
    if not math.isfinite(flops):
        raise ValueError("Cost estimate overflow")
    return {"cost_protocol": COST_PROTOCOL, "formula": "2 * parameters * (prefill_tokens + generated_tokens)",
            "parameter_basis": parameter_basis, "parameters": parameters,
            "prefill_accounting": prefill_accounting, "prefill_tokens": prefill_tokens,
            "generated_tokens": generated_tokens, "estimated_lm_flops": flops,
            "estimated_lm_gflops": flops / 1e9,
            "scope": "analytical_lm_token_proxy_not_measured_hardware_operations"}


def request_costs(stats):

    parameters = stats.get("nominal_parameters")
    if parameters is None:
        return {"nominal_processed_lm_flops": None, "nominal_unique_prompt_lm_flops": None}
    generated = stats["generated_tokens"]
    processed = estimate_lm_cost(parameters, stats["prompt_tokens"], generated, "processed")
    unique = stats.get("unique_prompt_tokens")
    return {"nominal_processed_lm_flops": processed["estimated_lm_flops"],
            "nominal_unique_prompt_lm_flops": (
                estimate_lm_cost(parameters, unique, generated, "exact_full_prompt_once")["estimated_lm_flops"]
                if unique is not None else None)}


def summarize_costs(stats_rows):
    rows = list(stats_rows)
    values = [request_costs(row) for row in rows]
    parameters = {row.get("nominal_parameters") for row in rows}
    if len(parameters) > 1:
        raise ValueError("Cannot combine different nominal parameter bases")
    result = {"cost_protocol": COST_PROTOCOL, "nominal_parameters": next(iter(parameters)) if parameters else None}
    for key in ("nominal_processed_lm_flops", "nominal_unique_prompt_lm_flops"):
        series = [value[key] for value in values]
        mean = statistics.mean(series) if series and all(value is not None for value in series) else None
        result["mean_" + key] = mean
        result["mean_" + key.replace("_flops", "_gflops")] = mean / 1e9 if mean is not None else None
    return result
