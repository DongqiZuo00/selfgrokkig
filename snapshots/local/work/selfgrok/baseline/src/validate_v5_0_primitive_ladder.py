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


def transform(prefix: str, operation: str):
    return apply_prepend_curriculum(
        [row(prefix)],
        {
            "name": operation,
            "prepend_curriculum_inputs": ["$opposite"],
            "prepend_curriculum_operation": operation,
        },
    )[0]


def main() -> None:
    consume = transform("R", "consume")
    replace = transform("R", "replace")
    prepend = transform("R", "prepend")
    assert consume["ground_truth"][0]["input"] == "B"
    assert consume["ground_truth"][0]["expected_output"] == ""
    assert replace["ground_truth"][0]["expected_output"] == "R"
    assert prepend["ground_truth"][0]["expected_output"] == "RB"
    assert "expected output is the empty tape" in consume["messages"][0]["content"]
    assert "Replace the complete input tape" in replace["messages"][0]["content"]
    consume_program = """START start:
    NEXT pull
PULLER_RB pull:
    [B] end
    [R] NONE
    [EMPTY] NONE
END end"""
    replace_program = """START start:
    NEXT pull
PULLER_RB pull:
    [B] paint_r
    [R] NONE
    [EMPTY] NONE
PAINTER_RED paint_r:
    NEXT end
END end"""
    prepend_program = """START start:
    NEXT pull
PULLER_RB pull:
    [B] paint_r
    [R] NONE
    [EMPTY] NONE
PAINTER_RED paint_r:
    NEXT replay_b
PAINTER_BLUE replay_b:
    NEXT end
END end"""
    assert verify_full(consume_program, consume["ground_truth"]).reward == 1
    assert verify_full(replace_program, replace["ground_truth"]).reward == 1
    assert verify_full(prepend_program, prepend["ground_truth"]).reward == 1
    print("v5.0 strict primitive ladder: PASS")


if __name__ == "__main__":
    main()
