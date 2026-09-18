import copy

import pytest

from keir.data import export_training_data, split_statement_question
from keir.evaluation import (
    PAPER_BENCHMARKS, PAPER_SEEDS, aggregate_paper, aggregate_seeds, scorer_digest,
)
from keir.io import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from keir.schemas import RefinedExample, SubtaskSolution
from keir.socratic import SocraticValidationError
from keir.target_review import (
    apply_target_reviews, check_target, has_current_approval,
    record_digest, write_target_review_queue,
)
from keir.teacher_stage_a import filter_stage_a_response
from keir.teacher_stage_b import validate_stage_b_response


@pytest.mark.parametrize("statement,question", [
    ("She buys 3 notebooks.", "How much does she pay if each notebook costs $2.50?"),
    ("She buys 3 notebooks.", "How much does she pay if each costs 2.5 dollars?"),
    ("Dr. Lee buys 3 notebooks.", "How much does each cost?"),
    ("The price is in U.S. dollars.", "How much is paid?"),
    ("The shop opens at 3 p.m.", "How many hours until 5 p.m.?"),
    ("A notebook costs 2.5 dollars.", "Calculate the cost of 3 notebooks."),
])
def test_question_split_preserves_decimals_and_abbreviations(statement, question):
    assert split_statement_question(statement + " " + question, strict=True) == (statement, question)


def test_single_question_keeps_all_givens():
    question = "What is the cost of 3 notebooks at $2.50 each?"
    assert split_statement_question(question, strict=True) == (question, question)


@pytest.mark.parametrize("text", ["There are 2 apples.", "How many apples? How many oranges?",
                                   "How many apples? Calculate the cost."])
def test_ambiguous_targets_require_review(text):
    with pytest.raises(ValueError, match="ambiguous_statement_question_split"):
        split_statement_question(text, strict=True)
    assert split_statement_question(text) == (text, "What is the final answer?")


def test_construction_preserves_ambiguous_input_in_rejection_file(tmp_path):
    from keir.socratic import build_official_socratic_v0
    problem = "There are 2 apples. How many apples? How many oranges?"
    main, socratic = str(tmp_path / "main.jsonl"), str(tmp_path / "socratic.jsonl")
    rejected, output = str(tmp_path / "rejected.jsonl"), str(tmp_path / "out.jsonl")
    write_jsonl(main, [{"question": problem, "answer": "#### 2"}])
    write_jsonl(socratic, [{"question": problem, "answer": "How many apples? ** There are 2.\n#### 2"}])
    result = build_official_socratic_v0(socratic, main, output, rejected, str(tmp_path / "stats.json"))
    assert result["rejection_reasons"] == {"ambiguous_statement_question_split": 1}
    assert result["accepted_instances"] == 0
    assert list(read_jsonl(rejected))[0]["source"]["question"] == problem


@pytest.mark.parametrize("original,variant,conflict", [
    ("How many red balls are there?", "What is the number of blue balls?", "color_changed"),
    ("How many apples are there?", "What is the number of oranges?", "entity_changed"),
    ("How many meters remain?", "What is the number of feet remaining?", "unit_changed"),
])
def test_explicit_target_conflicts_are_rejected(original, variant, conflict):
    assert conflict in check_target(original, variant)["conflicts"]
    decision = filter_stage_a_response(original, {"rewritten_questions": [variant]}, variants=1)[0]
    assert decision["reason"] == "target_conflict"
    assert not decision["retained"]


def test_passing_target_guard_is_not_a_semantic_certificate():
    for candidate in ("How many red spheres remain?", "How large is the resulting collection?"):
        result = check_target("How many red balls remain?", candidate)
        assert result["status"] == "needs_review"
        assert not result["conflicts"] and not result["semantic_equivalence_verified"]


def test_stage_a_preserves_number_surface_form():
    decision = filter_stage_a_response(
        "How much remains from 1,000 dollars?",
        {"rewritten_questions": ["What amount remains from 1000 dollars?"]}, variants=1)[0]
    assert decision["reason"] == "question_numbers_changed"


def ball_original():
    return RefinedExample(id="x-v0", source_id="x", variant_index=0,
                          statement="There are 5 red balls and 2 blue balls.",
                          question="How many red balls are there?", answer="5",
                          chain=[SubtaskSolution("How many red balls are there?", "5")],
                          is_original=True)


@pytest.mark.parametrize("variant,terminal,code", [
    ("What is the number of blue balls?", "How many blue balls are there?", "variant_target_conflict"),
    ("What is the number of red balls?", "How many blue balls are there?", "terminal_target_conflict"),
    ("What is the requested count?", "How many blue balls are there?", "terminal_target_conflict"),
])
def test_wrong_target_cannot_pass_stage_b_just_by_echoing_gold(variant, terminal, code):
    with pytest.raises(SocraticValidationError) as error:
        validate_stage_b_response(ball_original(), variant,
                                  {"steps": [{"subquestion": terminal, "answer": "5"}], "final_answer": "5"})
    assert error.value.code == code


def corpus_with_variant(tmp_path):
    rows = [RefinedExample(
        id="source-%d-v0" % i, source_id="source-%d" % i, variant_index=0,
        statement="A box contains %d beads." % count, question="How many beads are there?",
        answer=str(count), chain=[SubtaskSolution("How many beads are there?", str(count))],
        is_original=True,
    ).to_dict() for i, count in enumerate((7, 12))]
    variant = copy.deepcopy(rows[0])
    variant.update(id="source-0-v1", variant_index=1, is_original=False,
                   question="What is the number of beads in the box?")
    variant["metadata"].update(original_question=rows[0]["question"], target_review={"status": "pending"})
    source = tmp_path / "corpus.jsonl"
    write_jsonl(str(source), rows + [variant])
    return str(source), variant


def test_review_queue_content_bound_approval_and_strict_export(tmp_path):
    source, variant = corpus_with_variant(tmp_path)
    counts = export_training_data(source, str(tmp_path / "normal"))
    assert counts["variants_without_human_target_approval"] == 1
    with pytest.raises(ValueError, match="Unreviewed"):
        export_training_data(source, str(tmp_path / "blocked"), require_target_review=True)
    queue = str(tmp_path / "queue.jsonl")
    assert write_target_review_queue(source, queue) == 1
    reviews = list(read_jsonl(queue))
    assert reviews[0]["decision"] == "pending" and reviews[0]["original_question"]
    reviews[0].update(decision="approved", reviewer="unit-test-reviewer")
    write_jsonl(queue, reviews)
    output = str(tmp_path / "reviewed.jsonl")
    assert apply_target_reviews(source, queue, output) == {"retained": 3, "rejected": 0, "reviewed": 1}
    approved = list(read_jsonl(output))[-1]
    assert has_current_approval(approved)
    assert record_digest(approved) == record_digest(variant)
    counts = export_training_data(output, str(tmp_path / "strict"), require_target_review=True)
    assert counts["variants_without_human_target_approval"] == 0
    approved["question"] = "How many blue beads remain?"
    assert not has_current_approval(approved)

    assert list(read_jsonl(source))[-1] == variant


@pytest.mark.parametrize("fault", ["stale", "no_reviewer", "unknown", "pending", "duplicate"])
def test_review_rejects_unverifiable_decisions(tmp_path, fault):
    source, _ = corpus_with_variant(tmp_path)
    queue = str(tmp_path / "queue.jsonl")
    write_target_review_queue(source, queue)
    reviews = list(read_jsonl(queue))
    reviews[0].update(decision="approved", reviewer="unit-test-reviewer")
    if fault == "stale":
        reviews[0]["record_sha256"] = stable_hash("different content")
    elif fault == "no_reviewer":
        reviews[0]["reviewer"] = ""
    elif fault == "unknown":
        reviews[0]["id"] = "not-in-corpus"
    elif fault == "pending":
        reviews[0]["decision"] = "pending"
    else:
        reviews.append(dict(reviews[0]))
    write_jsonl(queue, reviews)
    with pytest.raises(ValueError):
        apply_target_reviews(source, queue, str(tmp_path / "out.jsonl"))
    assert not (tmp_path / "out.jsonl").exists()


def test_rejected_variants_have_a_separate_audit_record(tmp_path):
    source, _ = corpus_with_variant(tmp_path)
    queue, output = str(tmp_path / "queue.jsonl"), str(tmp_path / "reviewed.jsonl")
    write_target_review_queue(source, queue)
    reviews = list(read_jsonl(queue))
    reviews[0].update(decision="rejected", reviewer="unit-test-reviewer", notes="Changed target")
    write_jsonl(queue, reviews)
    assert apply_target_reviews(source, queue, output)["rejected"] == 1
    assert len(list(read_jsonl(output))) == 2
    assert list(read_jsonl(output + ".rejected.jsonl"))[0]["metadata"]["target_review"]["status"] == "rejected"
    with pytest.raises(ValueError):
        write_target_review_queue(source, source)
    with pytest.raises(ValueError):
        apply_target_reviews(source, queue, source)


def metric_grid(tmp_path):
    files = []
    for index, seed in enumerate(PAPER_SEEDS):
        for benchmark in PAPER_BENCHMARKS:
            offset = (index - 1) * 10
            accuracy = 50 + (offset if benchmark == "asdiv" else -offset if benchmark == "gsm8k" else 0)
            path = str(tmp_path / ("%s_%s.json" % (seed, benchmark)))
            write_json(path, {"training_seed": seed, "inference_seed": 9, "benchmark": benchmark,
                              "configuration_id": stable_hash("unit-test configuration"),
                              "input_sha256": stable_hash(benchmark), "mode": "keir", "samples": 1,
                              "consolidation": True, "accuracy": accuracy, "instances": 100,
                              "scoring_protocol": "keir-normalized-em-v2", "scorer_sha256": scorer_digest()})
            files.append(path)
    return files


def test_macro_averages_within_seed_before_sample_standard_deviation(tmp_path):
    files = metric_grid(tmp_path)
    result = aggregate_paper(files, str(tmp_path / "summary.json"))
    assert result["macro_accuracy"] == {"mean": 50, "std": 0, "values": [50, 50, 50]}
    assert result["benchmark_accuracy"]["asdiv"]["std"] == 10
    one_benchmark = aggregate_seeds([p for p in files if p.endswith("_asdiv.json")])
    assert one_benchmark["metrics"]["accuracy"]["mean"] == 50
    assert one_benchmark["metrics"]["accuracy"]["std"] == 10
    assert "training_seed" not in one_benchmark["metrics"]
    with pytest.raises(ValueError, match="one benchmark"):
        aggregate_seeds(files)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "configuration", "input", "count", "seed", "nan", "range", "scorer"])
def test_macro_aggregation_rejects_incomplete_or_mixed_runs(tmp_path, fault):
    files = metric_grid(tmp_path)
    row = read_json(files[0])
    if fault == "missing":
        files.pop()
    elif fault == "duplicate":
        files.append(files[0])
    elif fault == "configuration":
        row["configuration_id"] = stable_hash("another model")
    elif fault == "input":
        row["input_sha256"] = stable_hash("another input")
    elif fault == "count":
        row["instances"] = 99
    elif fault == "seed":
        del row["training_seed"]
    elif fault == "nan":
        row["accuracy"] = float("nan")
    elif fault == "scorer":
        row["scoring_protocol"] = "keir-normalized-em-v1"
    else:
        row["accuracy"] = 101
    write_json(files[0], row)
    with pytest.raises(ValueError):
        aggregate_paper(files)


def test_new_cli_commands_are_wired(tmp_path, capsys):
    from keir.cli import main
    source, _ = corpus_with_variant(tmp_path)
    queue = str(tmp_path / "review.jsonl")
    main(["review-targets", "--input", source, "--output", queue])
    assert len(list(read_jsonl(queue))) == 1
    files = metric_grid(tmp_path)
    summary = str(tmp_path / "macro.json")
    main(["aggregate-paper", *files, "--output", summary])
    assert read_json(summary)["macro_accuracy"]["mean"] == 50
