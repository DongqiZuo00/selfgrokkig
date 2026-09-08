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


def task(*, semantic_label: bool):
    return apply_prepend_curriculum(
        [row("R")],
        {
            "name": "table",
            "prepend_curriculum_inputs": ["$opposite"],
            "prepend_curriculum_operation": "prepend",
            "finite_curriculum_prompt_mode": "io_table",
            "prepend_curriculum_semantic_label": semantic_label,
        },
    )[0]


def main() -> None:
    plain = task(semantic_label=False)
    labelled = task(semantic_label=True)
    prompt = plain["messages"][0]["content"].split("# Task", 1)[1]
    assert "Input B -> Output RB" in prompt
    assert "replace" not in prompt.lower()
    assert "prepend" not in prompt.lower()
    assert "PREPEND R" in labelled["messages"][0]["content"]
    assert plain["ground_truth"] == labelled["ground_truth"]
    program = """START start:
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
    assert verify_full(program, plain["ground_truth"]).reward == 1
    print("v5.2 invariant I/O-table lexical bridge: PASS")


if __name__ == "__main__":
    main()
