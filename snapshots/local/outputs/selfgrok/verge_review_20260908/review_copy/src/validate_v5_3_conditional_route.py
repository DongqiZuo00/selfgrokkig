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


def task(prefix: str, operation: str):
    return apply_prepend_curriculum(
        [row(prefix)],
        {
            "name": operation,
            "prepend_curriculum_inputs": ["$opposite", "$prefix"],
            "prepend_curriculum_operation": operation,
            "finite_curriculum_prompt_mode": "io_table",
            "prepend_curriculum_semantic_label": False,
        },
    )[0]


PROGRAMS = {
    ("R", "branch_consume_same"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] prefix_for_b
    [R] end
    [EMPTY] NONE
PAINTER_RED prefix_for_b:
    NEXT replay_b
PAINTER_BLUE replay_b:
    NEXT end
END end""",
    ("B", "branch_consume_same"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] end
    [R] prefix_for_r
    [EMPTY] NONE
PAINTER_BLUE prefix_for_r:
    NEXT replay_r
PAINTER_RED replay_r:
    NEXT end
END end""",
    ("R", "branch_replace_same"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] prefix_for_b
    [R] same_r
    [EMPTY] NONE
PAINTER_RED prefix_for_b:
    NEXT replay_b
PAINTER_BLUE replay_b:
    NEXT end
PAINTER_RED same_r:
    NEXT end
END end""",
    ("B", "branch_replace_same"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] same_b
    [R] prefix_for_r
    [EMPTY] NONE
PAINTER_BLUE prefix_for_r:
    NEXT replay_r
PAINTER_RED replay_r:
    NEXT end
PAINTER_BLUE same_b:
    NEXT end
END end""",
    ("R", "prepend"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] prefix_for_b
    [R] prefix_for_r
    [EMPTY] NONE
PAINTER_RED prefix_for_b:
    NEXT replay_b
PAINTER_BLUE replay_b:
    NEXT end
PAINTER_RED prefix_for_r:
    NEXT replay_r
PAINTER_RED replay_r:
    NEXT end
END end""",
    ("B", "prepend"): """START start:
    NEXT pull
PULLER_RB pull:
    [B] prefix_for_b
    [R] prefix_for_r
    [EMPTY] NONE
PAINTER_BLUE prefix_for_r:
    NEXT replay_r
PAINTER_RED replay_r:
    NEXT end
PAINTER_BLUE prefix_for_b:
    NEXT replay_b
PAINTER_BLUE replay_b:
    NEXT end
END end""",
}


def main() -> None:
    for prefix in ("R", "B"):
        opposite = "B" if prefix == "R" else "R"
        for operation in ("branch_consume_same", "branch_replace_same", "prepend"):
            generated = task(prefix, operation)
            prompt = generated["messages"][0]["content"].split("# Task", 1)[1]
            assert "prepend" not in prompt.lower()
            assert "replace" not in prompt.lower()
            expected = {
                "branch_consume_same": {opposite: prefix + opposite, prefix: ""},
                "branch_replace_same": {
                    opposite: prefix + opposite,
                    prefix: prefix,
                },
                "prepend": {opposite: prefix + opposite, prefix: prefix + prefix},
            }[operation]
            actual = {
                case["input"]: case["expected_output"]
                for case in generated["ground_truth"]
            }
            assert actual == expected
            result = verify_full(PROGRAMS[(prefix, operation)], generated["ground_truth"])
            assert result.reward == 1, (prefix, operation, result)
    print("v5.3 conditional-route I/O ladder: PASS")


if __name__ == "__main__":
    main()
