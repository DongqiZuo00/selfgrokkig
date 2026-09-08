from __future__ import annotations

import json

from common import verify_full
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


def task(operation: str):
    return apply_prepend_curriculum(
        [row("R")],
        {
            "name": operation,
            "prepend_curriculum_inputs": ["$opposite"],
            "prepend_curriculum_operation": operation,
        },
    )[0]


def main() -> None:
    repeated = task("replace_repeat")
    prepend = task("prepend")
    assert repeated["ground_truth"][0]["input"] == "B"
    assert repeated["ground_truth"][0]["expected_output"] == "RR"
    assert prepend["ground_truth"][0]["expected_output"] == "RB"
    repeated_program = """START start:
    NEXT pull
PULLER_RB pull:
    [B] paint_r_1
    [R] NONE
    [EMPTY] NONE
PAINTER_RED paint_r_1:
    NEXT paint_r_2
PAINTER_RED paint_r_2:
    NEXT end
END end"""
    prepend_program = repeated_program.replace(
        "PAINTER_RED paint_r_2", "PAINTER_BLUE paint_r_2"
    )
    assert verify_full(repeated_program, repeated["ground_truth"]).reward == 1
    assert verify_full(prepend_program, prepend["ground_truth"]).reward == 1
    assert "exactly 'RR'" in repeated["messages"][0]["content"]
    print("v5.1 second-painter ladder: PASS")


if __name__ == "__main__":
    main()
