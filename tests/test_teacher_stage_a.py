import pytest

from keir.teacher_stage_a import filter_stage_a_response


def test_stage_a_filter_applies_number_and_duplicate_checks_in_order():
    decisions = filter_stage_a_response(
        "How many items are there after 3 days?",
        {
            "rewritten_questions": [
                "After 3 days, what is the number of items?",
                "How many items are there after 3 days?",
                "How many items are there?",
            ]
        },
        near_duplicate_ratio=0.95,
    )
    assert decisions[0]["reason"] == "retained"
    assert decisions[1]["reason"] == "near_duplicate"
    assert decisions[2]["reason"] == "question_numbers_changed"


def test_stage_a_filter_requires_exact_response_count():
    with pytest.raises(ValueError):
        filter_stage_a_response("How many?", {"rewritten_questions": ["What number?"]})


def test_stage_a_filter_detects_number_words_copied_from_statement():
    decisions = filter_stage_a_response(
        "If he spends three-quarters of this time reading, how many pages did he read?",
        {
            "rewritten_questions": [
                "If he spends three-quarters of the planned three hours reading, how many pages did he read?"
            ]
        },
        variants=1,
    )
    assert decisions[0]["reason"] == "question_numbers_changed"


def test_stage_a_filter_detects_lexicalized_counts():
    decisions = filter_stage_a_response(
        "How many marbles did the friends collect together?",
        {"rewritten_questions": ["How many marbles did the trio collect together?"]},
        variants=1,
    )
    assert decisions[0]["reason"] == "question_numbers_changed"
