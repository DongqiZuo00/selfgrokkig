from __future__ import annotations

import json
from collections import Counter

from common import EXP_ROOT, read_json, read_jsonl


ROOT = (
    EXP_ROOT
    / "raw_results"
    / "recipe_v1"
    / "branches"
    / "mistral__root__basic__seed42"
    / "train"
)


def main() -> None:
    summaries = []
    rows_by_update = []
    for update in range(1, 101):
        summaries.append(read_json(ROOT / f"update_{update:04d}_summary.json"))
        rows_by_update.append(read_jsonl(ROOT / f"update_{update:04d}.jsonl"))

    blocks = []
    for start in range(0, 100, 10):
        block_summaries = summaries[start : start + 10]
        block_rows = [row for rows in rows_by_update[start : start + 10] for row in rows]
        finish = Counter(str(row.get("finish_reason")) for row in block_rows)
        blocks.append(
            {
                "updates": [start + 1, start + 10],
                "successes": sum(int(item["successes"]) for item in block_summaries),
                "updates_with_signal": sum(int(item["active_advantages"]) > 0 for item in block_summaries),
                "parse_valid": sum(bool(row.get("parse_valid")) for row in block_rows),
                "mean_completion_tokens": round(
                    sum(int(row["completion_tokens"]) for row in block_rows) / len(block_rows), 1
                ),
                "length_finish_fraction": round(finish["length"] / len(block_rows), 4),
                "max_grad_norm": round(max(float(item["grad_norm"]) for item in block_summaries), 4),
            }
        )

    signal_updates = [int(item["update"]) for item in summaries if int(item["active_advantages"]) > 0]
    print(
        json.dumps(
            {
                "total_rollouts": sum(len(rows) for rows in rows_by_update),
                "total_successes": sum(int(item["successes"]) for item in summaries),
                "updates_with_signal": len(signal_updates),
                "last_signal_update": max(signal_updates),
                "final_cumulative_zero_advantage_updates": summaries[-1]["zero_advantage_updates"],
                "final_cumulative_generated_tokens": summaries[-1]["training_generated_tokens"],
                "blocks": blocks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
