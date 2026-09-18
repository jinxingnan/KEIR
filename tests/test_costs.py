import pytest

from keir.cli import main
from keir.costs import estimate_lm_cost, request_costs, summarize_costs
from keir.io import read_json


@pytest.mark.parametrize("prefill,generated,gflops", [
    (120.0, 184.1, 4257.4), (1166.0, 390.0, 21784.0), (923.0, 286.4, 16931.6),
])
def test_paper_table_arithmetic(prefill, generated, gflops):
    result = estimate_lm_cost(7_000_000_000, prefill, generated, "reported")
    assert result["estimated_lm_gflops"] == pytest.approx(gflops)
    assert result["estimated_lm_flops"] / 1e9 == pytest.approx(gflops)
    assert result["parameter_basis"] == "nominal_backbone"


def test_repeated_prompts_remain_distinct_cost_views():
    stats = {"nominal_parameters": 7_000_000_000, "prompt_tokens": 500,
             "unique_prompt_tokens": 100, "generated_tokens": 50}
    costs = request_costs(stats)
    assert costs["nominal_processed_lm_flops"] == 14e9 * 550
    assert costs["nominal_unique_prompt_lm_flops"] == 14e9 * 150
    summary = summarize_costs([stats, stats])
    assert summary["mean_nominal_processed_lm_gflops"] == 14 * 550
    assert "paper_lm_flops" not in summary


def test_missing_ledger_is_unknown_not_zero():
    assert request_costs({})["nominal_processed_lm_flops"] is None
    result = request_costs({"nominal_parameters": 7e9, "prompt_tokens": 50, "generated_tokens": 10})
    assert result["nominal_unique_prompt_lm_flops"] is None


@pytest.mark.parametrize("parameters,prefill,generated", [(0, 1, 1), (-1, 1, 1), (7e9, -1, 1),
                                                           (7e9, 1, float("nan")), (True, 1, 1)])
def test_invalid_cost_inputs_fail(parameters, prefill, generated):
    with pytest.raises(ValueError):
        estimate_lm_cost(parameters, prefill, generated, "reported")


def test_cost_cli_uses_explicit_inputs(tmp_path, capsys):
    path = tmp_path / "cost.json"
    main(["estimate-cost", "--nominal-parameters", "7000000000", "--prefill-tokens", "923",
          "--generated-tokens", "286.4", "--prefill-accounting", "reported", "--output", str(path)])
    assert read_json(path)["estimated_lm_gflops"] == pytest.approx(16931.6)


def test_mixed_parameter_bases_fail():
    with pytest.raises(ValueError, match="different nominal"):
        summarize_costs([{"nominal_parameters": 7e9, "prompt_tokens": 10, "generated_tokens": 1},
                         {"nominal_parameters": 8e9, "prompt_tokens": 10, "generated_tokens": 1}])
