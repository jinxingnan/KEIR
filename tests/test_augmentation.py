import json

from keir.augmentation import plan_gpt4o_augmentation


def test_augmentation_plan_is_dry_run_and_preserves_v0(tmp_path):
    source = tmp_path / "v0.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "x-v0",
                "source_id": "x",
                "variant_index": 0,
                "statement": "A has 2 items and gets 3 more.",
                "question": "How many items are there?",
                "answer": "5",
                "chain": [
                    {
                        "subquestion": "How many items are there?",
                        "answer": "2+3=5.",
                    }
                ],
                "target_quantity": "How many items are there?",
                "is_original": True,
                "metadata": {"validation": "strict_v1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    requests = tmp_path / "requests.jsonl"
    manifest = tmp_path / "manifest.json"
    result = plan_gpt4o_augmentation(
        str(source), str(requests), str(manifest), variants=3
    )
    assert result["status"] == "dry_run_no_api_calls"
    assert result["original_chain_api_calls"] == 0
    assert result["stage_b_requests_written"] == 0
    request = json.loads(requests.read_text(encoding="utf-8"))
    assert request["stage"] == "A_answer_equivalent_question_variants"
    assert request["parameters"]["model"] == "gpt-4o-2024-08-06"
