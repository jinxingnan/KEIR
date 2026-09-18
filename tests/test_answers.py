import pytest

from keir.answers import (
    answers_equal,
    extract_answer,
    extract_benchmark_prediction,
    gsm8k_official_exact_match,
    normalize_answer,
    normalize_benchmark_prediction,
    normalize_math_answer,
)


def test_extract_gsm_and_boxed_answers():
    assert extract_answer("work\n#### 1,234") == "1,234"
    assert extract_answer(r"Thus \boxed{\frac{3}{4}}.") == r"\frac{3}{4}"


def test_normalize_numeric_forms():
    assert normalize_answer("$1,234.00") == "1234"
    assert normalize_answer(r"\frac{3}{4}") == "3/4"
    assert answers_equal("Final answer: 0.75", r"\boxed{\frac{3}{4}}")


def test_percent_is_not_silently_converted():
    assert normalize_answer(r"10\%") == "10%"
    assert not answers_equal("10", "10%")
    assert answers_equal("10%", "10", dataset="gsm8k")
    assert not answers_equal("10%", "0.1", dataset="gsm8k")


def test_strict_gsm8k_extraction_never_uses_rationale_fallback():
    assert extract_benchmark_prediction("work 17\nFinal answer: 18", "gsm8k") == "18"
    assert extract_benchmark_prediction("work ends with 18", "gsm8k") == ""
    assert gsm8k_official_exact_match("Final answer: 1,234", "1234")
    assert not gsm8k_official_exact_match("Final answer: 18.0", "18")


def test_official_math_extraction_and_equivalence():
    assert extract_answer(r"\frac{3}{4}", "math") == r"\frac{3}{4}"
    assert extract_benchmark_prediction(r"first \boxed{1}; then \boxed{\frac{3}{4}}", "math") == r"\frac{3}{4}"
    assert normalize_math_answer(r"x=\sqrt3") == r"\sqrt{3}"
    assert answers_equal("0.5", r"\frac{1}{2}", dataset="math")
    assert answers_equal(r"60^{\circ}", "60", dataset="math")
    assert answers_equal(r"5\text{ cm}", "5", dataset="math")


@pytest.mark.parametrize("text, expected", [
    (r"Final answer: \frac{2}{4}", r"\frac{2}{4}"),
    (r"Final answer: A+B", "A+B"),
    (r"Final answer: x=\sqrt3", r"\sqrt{3}"),
    (r"Final answer: 60^{\circ}", "60"),
    ("Final answer: 0.5", r"\frac{1}{2}"),
    ("The intermediate result is 17", ""),
])
def test_math_prediction_storage_uses_scoring_normalizer(text, expected):
    result = normalize_benchmark_prediction(text, "math")
    assert result == expected
    if result:
        assert answers_equal(result, extract_benchmark_prediction(text, "math"), dataset="math")


def test_arithmetic_prediction_storage_preserves_existing_protocol():
    assert normalize_benchmark_prediction("Final answer: 1,234.00", "gsm8k") == "1234"
    assert normalize_benchmark_prediction("Intermediate result: 17", "gsm8k") == ""


@pytest.mark.parametrize("dataset", ["gsm8k", "asdiv", "mawps", "svamp"])
@pytest.mark.parametrize("value, gold", [
    ("1/2", "0.5"), ("-1 / 2", "-0.5"), ("1e3", "1000"),
    ("+0.5", "1/2"), (".5", "1/2"), (r"\frac{1}{2}", "0.5"),
    ("1.5/3", "0.5"), ("1,000.00", "1000"), ("1/2 hours", "0.5"),
])
def test_complete_numeric_answer_roundtrip(dataset, value, gold):
    text = "Final answer: " + value
    result = normalize_benchmark_prediction(text, dataset)
    assert result
    assert answers_equal(result, gold, dataset)


@pytest.mark.parametrize("value", ["1/0", "1/", "1e", "1e+", "1e3x", "1+2",
                                  "1 or 2", "1 and 2", "1/2 = 0.5", "1,23", "1!",
                                  "NaN", "inf", "", "1.2.3", "1/2/3", "1e999999", "1.."])
def test_malformed_values_never_become_numeric_prefixes(value):
    assert extract_benchmark_prediction("Final answer: " + value, "gsm8k") == ""
    assert not answers_equal(value, "1", "gsm8k")


def test_fraction_never_matches_its_numerator_or_denominator():
    from keir.evaluation import evaluate_prediction_rows
    for dataset in ("gsm8k", "asdiv", "mawps", "svamp"):
        for gold, expected in (("1", 0), ("2", 0), ("0.5", 1)):
            row = {"id": dataset + "-1", "prediction": "1", "gold": gold,
                   "solutions": ["Final answer: 1/2"], "metadata": {"dataset": dataset}}
            assert evaluate_prediction_rows([row])["correct"] == expected
        assert answers_equal("1/2", "0.5", dataset)
        assert not answers_equal("1/2", "2", dataset)


def test_last_explicit_marker_including_invalid_marker_wins():
    assert extract_benchmark_prediction("#### 1\nFinal answer: 1/2", "gsm8k") == "1/2"
    assert extract_benchmark_prediction("Final answer: 1\n#### 1/2", "gsm8k") == "1/2"
    assert extract_benchmark_prediction("Final answer: 1\nFinal answer:", "gsm8k") == ""


def test_numeric_equality_is_exact_not_float_tolerant():
    assert not answers_equal("0.0000000001", "0", "gsm8k")
    assert not answers_equal("", "", "gsm8k")
    assert not answers_equal("", "", "math")
    assert not answers_equal("0.1234567890123456789012345678901", "0.1234567890123456789012345678902", "gsm8k")


def test_self_consistency_is_revoted_from_raw_traces():
    from keir.evaluation import prediction_row_answer
    row = {"prediction": "1", "solutions": ["Final answer: 1/2", "Final answer: 0.5", "Final answer: 1"],
           "metadata": {"samples": 3, "dataset": "gsm8k"}}
    assert answers_equal(prediction_row_answer(row), "0.5", "gsm8k")
    row["metadata"]["samples"] = 4
    with pytest.raises(ValueError, match="every sampled trace"):
        prediction_row_answer(row)


def test_self_consistency_groups_equivalent_numeric_forms():
    from keir.answers import majority_answer
    traces = ["Final answer: 1", "Final answer: 1", "Final answer: 1/2",
              "Final answer: 0.5", "Final answer: 5e-1"]
    assert answers_equal(majority_answer(traces, "gsm8k"), "0.5", "gsm8k")


def test_rescore_preserves_identity_and_does_not_overwrite_predictions(tmp_path):
    from keir.evaluation import evaluate_prediction_file
    from keir.io import write_jsonl
    path = tmp_path / "predictions.jsonl"
    identity = {"benchmark": "gsm8k", "training_seed": 42, "scoring_protocol": "old"}
    write_jsonl(str(path), [{"id": "gsm8k-1", "prediction": "1", "gold": "0.5",
                           "solutions": ["Final answer: 1/2"],
                           "metadata": {"dataset": "gsm8k", "run_identity": identity}}])
    before = path.read_bytes()
    result = evaluate_prediction_file(str(path), str(tmp_path / "rescored.json"))
    assert result["correct"] == 1 and result["training_seed"] == 42
    assert result["scoring_protocol"] == "keir-normalized-em-v2"
    assert result["predictions_sha256"]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_prediction_file(str(path), str(path))
