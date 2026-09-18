import json

import pytest

from keir.data import split_statement_question
from keir.socratic import (
    SocraticValidationError,
    build_official_socratic_v0,
    parse_official_socratic_answer,
    step_answer_matches_gold,
    _validate_unannotated_calculations,
)


def test_parse_official_socratic_chain_and_calculations():
    raw = (
        "How many clips were sold in May? ** Half of 48 is <<48/2=24>>24.\n"
        "How many clips were sold altogether? ** The total is <<48+24=72>>72.\n"
        "#### 72"
    )
    chain, gold = parse_official_socratic_answer(raw)
    assert gold == "72"
    assert [step.task for step in chain] == [
        "How many clips were sold in May?",
        "How many clips were sold altogether?",
    ]
    assert step_answer_matches_gold(chain[-1].solution, gold)


def test_parser_enforces_paper_maximum_chain_length():
    raw = "\n".join("How many items? ** There are 5." for _ in range(11)) + "\n#### 5"
    with pytest.raises(SocraticValidationError) as error:
        parse_official_socratic_answer(raw)
    assert error.value.code == "too_many_steps"


@pytest.mark.parametrize(
    "raw,code",
    [
        ("Define x ** x is 2.\n#### 2", "non_interrogative_subquestion"),
        ("How many? **\n#### 2", "invalid_step_delimiter"),
        ("How many? ** <<1+1=3>>3.\n#### 3", "incorrect_calculation"),
        ("How many? ** <<1+1=2>>2.\n#### 3", "terminal_answer_mismatch"),
        ("How many? ** 4+4+6=18.\n#### 18", "incorrect_unannotated_calculation"),
    ],
)
def test_strict_parser_rejects_invalid_chains(raw, code):
    with pytest.raises(SocraticValidationError) as error:
        parse_official_socratic_answer(raw)
    assert error.value.code == code


def test_prepare_v0_aligns_main_and_writes_rejections(tmp_path):
    main = tmp_path / "main.jsonl"
    socratic = tmp_path / "socratic.jsonl"
    output = tmp_path / "v0.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    stats = tmp_path / "stats.json"
    questions = [
        "A has 2 items and gets 3 more. How many items are there?",
        "A has 4 items. How many items are there?",
    ]
    main.write_text(
        "\n".join(
            json.dumps({"question": question, "answer": "work\n#### %s" % gold})
            for question, gold in zip(questions, ("5", "4"))
        )
        + "\n",
        encoding="utf-8",
    )
    socratic.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": questions[0],
                        "answer": "How many items are there? ** 2+3=<<2+3=5>>5.\n#### 5",
                    }
                ),
                json.dumps(
                    {
                        "question": questions[1],
                        "answer": "Count the items ** There are 4.\n#### 4",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_official_socratic_v0(
        str(socratic), str(main), str(output), str(rejected), str(stats)
    )
    assert result["accepted_instances"] == 1
    assert result["rejection_reasons"] == {"non_interrogative_subquestion": 1}
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["id"] == "gsm8k-train-00000-v0"
    assert row["chain"][0]["subquestion"] == "How many items are there?"


def test_split_terminal_imperative_sentence():
    statement, question = split_statement_question(
        "There are 2 red hats. Calculate the total number of hats."
    )
    assert statement == "There are 2 red hats."
    assert question == "Calculate the total number of hats."


def test_terminal_answer_uses_conclusion_not_last_context_number():
    raw = (
        "How far? ** 23 total - 13 known = 10 meters after the 3rd turn.\n"
        "#### 10"
    )
    chain, gold = parse_official_socratic_answer(raw)
    assert step_answer_matches_gold(chain[-1].solution, gold)


def test_terminal_answer_allows_explicit_unit_scaling():
    raw = "How many cents? ** The cost is $<<1/4=0.25>>0.25.\n#### 25"
    parse_official_socratic_answer(raw, problem_context="How many cents?")


def test_terminal_answer_does_not_use_intermediate_operand():
    raw = "How many cavities? ** 16 / 4 = <<16/4=4>>4 cavities.\n#### 16"
    with pytest.raises(SocraticValidationError) as error:
        parse_official_socratic_answer(raw)
    assert error.value.code == "terminal_answer_mismatch"


def test_unannotated_fraction_result_is_not_truncated():
    _validate_unannotated_calculations("Half of half is 1/2 * 1/2 = 1/4.")


def test_prepare_v0_rejects_changed_how_many_quantity(tmp_path):
    question = "An amoeba doubles every two days. How many days until there are 16?"
    main = tmp_path / "main.jsonl"
    socratic = tmp_path / "socratic.jsonl"
    main.write_text(
        json.dumps({"question": question, "answer": "work\n#### 8"}) + "\n",
        encoding="utf-8",
    )
    socratic.write_text(
        json.dumps(
            {
                "question": question,
                "answer": "How many amoebae are there after 8 days? ** There are 16 after 8 days.\n#### 8",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_official_socratic_v0(
        str(socratic),
        str(main),
        str(tmp_path / "out.jsonl"),
        str(tmp_path / "rejected.jsonl"),
        str(tmp_path / "stats.json"),
    )
    assert result["rejection_reasons"] == {"terminal_target_type_mismatch": 1}
