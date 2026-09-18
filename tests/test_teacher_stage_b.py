import pytest

from keir.teacher_stage_b import validate_stage_b_response
from keir.schemas import RefinedExample, SubtaskSolution
from keir.socratic import SocraticValidationError


def original():
    return RefinedExample(
        id="x-v0",
        source_id="x",
        variant_index=0,
        statement="A has 2 items and gets 3 more.",
        question="How many items are there?",
        answer="5",
        chain=[SubtaskSolution("How many items are there?", "2+3=5")],
        is_original=True,
    )


def test_stage_b_accepts_valid_socratic_chain():
    chain, answer = validate_stage_b_response(
        original(),
        "What is the total number of items?",
        {
            "steps": [
                {
                    "subquestion": "What is the total number of items?",
                    "answer": "There are <<2+3=5>>5 items.",
                }
            ],
            "final_answer": "5",
        },
    )
    assert answer == "5"
    assert len(chain) == 1


def test_stage_b_does_not_impose_an_unreported_lexical_overlap_threshold():
    chain, answer = validate_stage_b_response(
        original(), "What is the total number of items?",
        {"steps": [{"subquestion": "How large is the resulting collection?", "answer": "2+3=5"}],
         "final_answer": "5"},
    )
    assert answer == "5" and len(chain) == 1


def test_stage_b_rejects_non_socratic_step():
    with pytest.raises(SocraticValidationError) as error:
        validate_stage_b_response(
            original(),
            "What is the total number of items?",
            {
                "steps": [{"subquestion": "Compute the total", "answer": "5"}],
                "final_answer": "5",
            },
        )
    assert error.value.code == "non_interrogative_subquestion"


def test_stage_b_rejects_duplicate_subquestions():
    with pytest.raises(SocraticValidationError) as error:
        validate_stage_b_response(
            original(),
            "What is the total number of items?",
            {
                "steps": [
                    {"subquestion": "What is the total number of items?", "answer": "5"},
                    {"subquestion": "What is the total number of items?", "answer": "5"},
                ],
                "final_answer": "5",
            },
        )
    assert error.value.code == "duplicate_subquestion"
