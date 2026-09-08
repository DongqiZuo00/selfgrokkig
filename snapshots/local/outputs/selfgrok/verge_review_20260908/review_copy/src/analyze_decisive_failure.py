from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

from common import EXP_ROOT, read_json, read_jsonl


DEC_ROOT = EXP_ROOT / "raw_results" / "self_grok_decisive_mistral_v4_1"
RECIPE_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def summarize(rows):
    lengths = sorted(int(row["completion_tokens"]) for row in rows)
    n = len(lengths)
    verifier_counts = Counter(str(row["verifier_message"])[:160] for row in rows)
    return {
        "rollouts": n,
        "rewards": sum(int(row["reward"]) for row in rows),
        "parse_valid": sum(bool(row["parse_valid"]) for row in rows),
        "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in rows)),
        "stop_reasons": dict(Counter(str(row.get("stop_reason")) for row in rows)),
        "length_min_p50_p90_max": [
            lengths[0],
            lengths[n // 2],
            lengths[min(n - 1, int(n * 0.9))],
            lengths[-1],
        ],
        "at_least_2048_tokens": sum(value >= 2048 for value in lengths),
        "contains_START_start": sum("START start:" in str(row["completion"]) for row in rows),
        "contains_END_end": sum("END end" in str(row["completion"]) for row in rows),
        "top_verifier_message_prefixes": verifier_counts.most_common(5),
    }


def main():
    runtime = read_json(
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_runtime_branches.json"
    )
    target_ids = {
        str(row["id"])
        for row in load_from_disk(str(EXP_ROOT / "data" / "target_splits" / "target_train"))
    }
    output = {
        "starting_target_endpoint": summarize(
            read_jsonl(DEC_ROOT / "starting_solver_target_endpoint.jsonl")
        ),
        "starting_scope_endpoint": summarize(
            read_jsonl(DEC_ROOT / "starting_solver_scope_endpoint.jsonl")
        ),
        "stage_probes": {},
        "branches": {},
    }
    for path in sorted((DEC_ROOT / "stage_probes").glob("*.jsonl")):
        if path.name.endswith(".telemetry.jsonl"):
            continue
        output["stage_probes"][path.stem] = summarize(read_jsonl(path))

    for branch in runtime["branches"]:
        branch_id = branch["branch_id"]
        endpoint = read_jsonl(
            RECIPE_ROOT / branch_id / "grounded_reward" / "update_0040.jsonl"
        )
        phase_rows = defaultdict(list)
        target_rows = []
        bridge_rows = []
        successes = []
        per_update = []
        for update in range(1, 41):
            summary = read_json(
                RECIPE_ROOT / branch_id / "train" / f"update_{update:04d}_summary.json"
            )
            rows = read_jsonl(
                RECIPE_ROOT / branch_id / "train" / f"update_{update:04d}.jsonl"
            )
            phase_rows[str(summary["phase"])].extend(rows)
            for row in rows:
                (target_rows if str(row["instance_id"]) in target_ids else bridge_rows).append(row)
                if int(row["reward"]) == 1:
                    successes.append(
                        {
                            "update": update,
                            "phase": summary["phase"],
                            "instance_id": row["instance_id"],
                            "is_target": str(row["instance_id"]) in target_ids,
                            "completion_tokens": row["completion_tokens"],
                            "program_excerpt": str(row.get("program", ""))[:600],
                        }
                    )
            per_update.append(
                {
                    "update": update,
                    "phase": summary["phase"],
                    "successes": summary["successes"],
                    "active_advantages": summary["active_advantages"],
                    "mean_tokens": sum(int(row["completion_tokens"]) for row in rows) / len(rows),
                }
            )
        output["branches"][branch_id] = {
            "endpoint": summarize(endpoint),
            "target_training": summarize(target_rows),
            "bridge_training": summarize(bridge_rows) if bridge_rows else None,
            "by_phase": {phase: summarize(rows) for phase, rows in phase_rows.items()},
            "nonzero_updates": [item for item in per_update if item["active_advantages"] > 0],
            "successful_training_samples": successes[:12],
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
