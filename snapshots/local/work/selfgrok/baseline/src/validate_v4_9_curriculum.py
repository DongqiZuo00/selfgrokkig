from __future__ import annotations

import json

from self_grok_decisive_v4_3_prepare import apply_prepend_curriculum


def row(prefix: str):
    return {
        "name": f"Prepend {prefix}",
        "messages": [{"role": "user", "content": "DSL rules\n\n# Task\nold task"}],
        "ground_truth": [],
        "difficulty": "old",
        "candidate_id": "old",
        "generator_config": json.dumps({"name": "old"}),
    }


def main() -> None:
    first = apply_prepend_curriculum(
        [row("R"), row("B")],
        {"name": "opposite", "prepend_curriculum_inputs": ["$opposite"]},
    )
    assert first[0]["ground_truth"] == [
        {
            "input": "B",
            "expected_output": "RB",
            "expected_accepted": True,
            "check_output": True,
            "description": "Allowed input 'B' -> output 'RB'",
        }
    ]
    assert first[1]["ground_truth"][0]["input"] == "R"
    assert first[1]["ground_truth"][0]["expected_output"] == "BR"
    assert "allowed inputs are exactly: B" in first[0]["messages"][0]["content"]
    assert "# Task" in first[0]["messages"][0]["content"]
    two = apply_prepend_curriculum(
        [row("R"), row("B")],
        {
            "name": "mixed_two",
            "prepend_curriculum_inputs": ["$opposite_then_prefix"],
        },
    )
    assert two[0]["ground_truth"][0]["input"] == "BR"
    assert two[0]["ground_truth"][0]["expected_output"] == "RBR"
    assert two[1]["ground_truth"][0]["input"] == "RB"
    assert two[1]["ground_truth"][0]["expected_output"] == "BRB"
    print("v4.9 explicit finite PREPEND curriculum: PASS")


if __name__ == "__main__":
    main()
