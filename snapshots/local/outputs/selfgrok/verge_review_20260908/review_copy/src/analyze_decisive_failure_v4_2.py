from __future__ import annotations

import json
from collections import Counter, defaultdict

from datasets import load_from_disk

from common import EXP_ROOT, read_json, read_jsonl


VERSION = "self_grok_decisive_mistral_v4_2"
DEC_ROOT = EXP_ROOT / "raw_results" / VERSION
RECIPE_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def summarize(rows):
    lengths = sorted(int(row["completion_tokens"]) for row in rows)
    n = len(lengths)
    verifier_counts = Counter(str(row.get("verifier_message", ""))[:200] for row in rows)
    finish = Counter(str(row.get("finish_reason")) for row in rows)
    return {
        "rollouts": n,
        "rewards": sum(int(row["reward"]) for row in rows),
        "parse_valid": sum(bool(row.get("parse_valid")) for row in rows),
        "mean_passed_case_fraction": (
            sum(float(row["passed_cases"]) / max(1, int(row["total_cases"])) for row in rows) / n
        ),
        "finish_reasons": dict(finish),
        "length_min_p50_p90_max": [
            lengths[0],
            lengths[n // 2],
            lengths[min(n - 1, int(n * 0.9))],
            lengths[-1],
        ],
        "at_least_2048_tokens": sum(value >= 2048 for value in lengths),
        "length_finish_fraction": finish["length"] / n,
        "contains_start": sum("START start:" in str(row.get("completion", "")) for row in rows),
        "contains_end": sum("END end" in str(row.get("completion", "")) for row in rows),
        "top_verifier_message_prefixes": verifier_counts.most_common(6),
        "mean_tokens_reward_1": (
            sum(int(row["completion_tokens"]) for row in rows if int(row["reward"]) == 1)
            / max(1, sum(int(row["reward"]) == 1 for row in rows))
        ),
        "mean_tokens_reward_0": (
            sum(int(row["completion_tokens"]) for row in rows if int(row["reward"]) == 0)
            / max(1, sum(int(row["reward"]) == 0 for row in rows))
        ),
    }


def main() -> None:
    runtime = read_json(EXP_ROOT / "manifests" / f"{VERSION}_runtime_branches.json")
    target_path = runtime["branches"][0]["target_training_path"]
    target_ids = {
        str(row["id"])
        for row in load_from_disk(str(EXP_ROOT / target_path))
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
        if not path.name.endswith(".telemetry.jsonl"):
            output["stage_probes"][path.stem] = summarize(read_jsonl(path))

    for branch in runtime["branches"]:
        branch_id = branch["branch_id"]
        endpoint = read_jsonl(
            RECIPE_ROOT / branch_id / "grounded_reward" / "update_0040.jsonl"
        )
        phase_rows = defaultdict(list)
        phase_target_rows = defaultdict(list)
        phase_bridge_rows = defaultdict(list)
        target_rows = []
        bridge_rows = []
        successful_samples = []
        update_rows = []
        for update in range(1, 41):
            summary = read_json(
                RECIPE_ROOT / branch_id / "train" / f"update_{update:04d}_summary.json"
            )
            rows = read_jsonl(
                RECIPE_ROOT / branch_id / "train" / f"update_{update:04d}.jsonl"
            )
            phase_rows[str(summary["phase"])].extend(rows)
            for row in rows:
                is_target = str(row["instance_id"]) in target_ids
                (target_rows if is_target else bridge_rows).append(row)
                (phase_target_rows if is_target else phase_bridge_rows)[str(summary["phase"])].append(row)
                if int(row["reward"]) == 1 and len(successful_samples) < 20:
                    successful_samples.append(
                        {
                            "update": update,
                            "phase": summary["phase"],
                            "is_target": is_target,
                            "completion_tokens": row["completion_tokens"],
                            "program_excerpt": str(row.get("program", ""))[:500],
                        }
                    )
            update_rows.append(
                {
                    "update": update,
                    "phase": summary["phase"],
                    "successes": int(summary["successes"]),
                    "active_advantages": int(summary["active_advantages"]),
                    "sigma_hat_zero_groups": int(summary["sigma_hat_zero_groups"]),
                    "grad_norm": float(summary["grad_norm"]),
                    "mean_tokens": sum(int(row["completion_tokens"]) for row in rows) / len(rows),
                    "parse_valid": sum(bool(row.get("parse_valid")) for row in rows),
                    "length_finished": sum(str(row.get("finish_reason")) == "length" for row in rows),
                }
            )
        blocks = []
        for first in range(0, 40, 5):
            block = update_rows[first:first + 5]
            blocks.append(
                {
                    "updates": [first + 1, first + 5],
                    "phase": sorted(set(row["phase"] for row in block)),
                    "successes": sum(row["successes"] for row in block),
                    "updates_with_signal": sum(row["active_advantages"] > 0 for row in block),
                    "parse_valid": sum(row["parse_valid"] for row in block),
                    "length_finished": sum(row["length_finished"] for row in block),
                    "mean_tokens": sum(row["mean_tokens"] for row in block) / len(block),
                    "max_grad_norm": max(row["grad_norm"] for row in block),
                }
            )
        output["branches"][branch_id] = {
            "proposal": branch.get("proposal"),
            "stages": branch.get("stages"),
            "endpoint": summarize(endpoint),
            "target_training": summarize(target_rows),
            "bridge_training": summarize(bridge_rows) if bridge_rows else None,
            "by_phase": {phase: summarize(rows) for phase, rows in phase_rows.items()},
            "target_by_phase": {
                phase: summarize(rows) for phase, rows in phase_target_rows.items()
            },
            "bridge_by_phase": {
                phase: summarize(rows) for phase, rows in phase_bridge_rows.items()
            },
            "five_update_blocks": blocks,
            "nonzero_updates": [row for row in update_rows if row["active_advantages"] > 0],
            "successful_training_samples": successful_samples,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
