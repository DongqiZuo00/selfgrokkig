from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from common import EXP_ROOT, atomic_json, read_json, read_jsonl
from prepend_conditions import condition_profile


VERSION = os.environ.get(
    "SELF_GROK_DECISIVE_VERSION", "self_grok_decisive_mistral_v4_1"
)
TAG = VERSION.removeprefix("self_grok_decisive_mistral_")
PROTOCOL_PATH = EXP_ROOT / "manifests" / f"{VERSION}.json"
BRANCHES_PATH = EXP_ROOT / "manifests" / f"{VERSION}_runtime_branches.json"
DEC_ROOT = EXP_ROOT / "raw_results" / VERSION
RECIPE_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def _summary(branch_id: str, split: str, goal: int) -> dict[str, Any]:
    return read_json(RECIPE_ROOT / branch_id / split / f"update_{goal:04d}_summary.json")


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    low = min(successes, trials - successes)
    tail = sum(math.comb(trials, k) for k in range(low + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def _paired_condition_test(
    candidate: dict[str, Any], direct: dict[str, Any], condition_index: int
) -> dict[str, Any]:
    positive = negative = ties = 0
    for key, candidate_vector in candidate["keyed_vectors"].items():
        direct_vector = direct["keyed_vectors"][key]
        delta = int(candidate_vector[condition_index]) - int(direct_vector[condition_index])
        positive += int(delta > 0)
        negative += int(delta < 0)
        ties += int(delta == 0)
    discordant = positive + negative
    return {
        "candidate_wins": positive,
        "direct_wins": negative,
        "ties": ties,
        "discordant_pairs": discordant,
        "two_sided_exact_p": _binomial_two_sided(positive, discordant),
    }


def _target_ignition(branch_id: str, target_ids: set[str], goal: int) -> dict[str, Any]:
    target_rollouts = successes = 0
    first_update = None
    for update in range(1, goal + 1):
        path = RECIPE_ROOT / branch_id / "train" / f"update_{update:04d}.jsonl"
        for item in read_jsonl(path):
            if str(item["instance_id"]) not in target_ids:
                continue
            target_rollouts += 1
            if int(item["reward"]) == 1:
                successes += 1
                first_update = update if first_update is None else first_update
    return {
        "target_training_rollouts": target_rollouts,
        "target_training_successes": successes,
        "first_target_success_update": first_update,
    }


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "keyed_vectors"}


def main() -> None:
    cfg = read_json(PROTOCOL_PATH)
    runtime = read_json(BRANCHES_PATH)
    branches = runtime["branches"]
    goal = int(cfg["solver_branches"]["matched_updates"])
    selection = [
        dict(row)
        for row in load_from_disk(
            str(EXP_ROOT / "data" / VERSION / "selection")
        )
    ]
    target_ids = {
        str(row["id"])
        for row in load_from_disk(str(EXP_ROOT / cfg["target"]["training_split"]))
    }
    starting = read_json(DEC_ROOT / "starting_solver_profile.json")

    records = []
    profiles: dict[str, dict[str, Any]] = {}
    for branch in branches:
        branch_id = str(branch["branch_id"])
        status = read_json(RECIPE_ROOT / branch_id / "status.json")
        if status.get("status") != "complete" or int(status["completed_updates"]) != goal:
            raise RuntimeError(f"branch did not reach the fixed round barrier: {branch_id}")
        endpoint_rows = read_jsonl(
            RECIPE_ROOT / branch_id / "grounded_reward" / f"update_{goal:04d}.jsonl"
        )
        profile = condition_profile(endpoint_rows, selection)
        profiles[branch_id] = profile
        scope = _summary(branch_id, "scope_probe", goal)
        record = {
            "branch_id": branch_id,
            "role": branch["role"],
            "proposal": branch.get("proposal"),
            "endpoint_profile": _compact_profile(profile),
            "scope": {
                "strict_rate": float(scope["full_pass_rate"]),
                "mean_case_fraction": float(scope["mean_case_fraction"]),
            },
            "training": {
                "updates": goal,
                "training_rollouts": int(status["training_rollouts"]),
                "training_generated_tokens": int(status["training_generated_tokens"]),
                "zero_advantage_updates": int(status["zero_advantage_updates"]),
                "fresh_optimizer_state": not bool(status["optimizer_inherited_from_parent"]),
                **_target_ignition(branch_id, target_ids, goal),
            },
        }
        records.append(record)

    direct = records[0]
    direct_profile = profiles[direct["branch_id"]]
    total_endpoint_successes = sum(record["endpoint_profile"]["counts"][4] for record in records)
    branches_with_success = sum(record["endpoint_profile"]["counts"][4] > 0 for record in records)
    transition = (
        total_endpoint_successes
        >= int(cfg["credit_transition"]["minimum_endpoint_full_passes"])
        and branches_with_success
        >= int(cfg["credit_transition"]["minimum_branches_with_endpoint_success"])
    )
    credited_one_based = 5 if transition else int(starting["bottleneck_index"])
    credited_index = credited_one_based - 1

    gains = []
    root_scope = starting["scope"]
    guard = cfg["scope_guard"]
    for record in records[1:]:
        profile = profiles[record["branch_id"]]
        gamma = float(profile["rates"][credited_index]) - float(
            direct_profile["rates"][credited_index]
        )
        paired = _paired_condition_test(profile, direct_profile, credited_index)
        strict_vs_start = float(record["scope"]["strict_rate"]) - float(
            root_scope["full_pass_rate"]
        )
        soft_vs_start = float(record["scope"]["mean_case_fraction"]) - float(
            root_scope["mean_case_fraction"]
        )
        strict_vs_direct = float(record["scope"]["strict_rate"]) - float(
            direct["scope"]["strict_rate"]
        )
        soft_vs_direct = float(record["scope"]["mean_case_fraction"]) - float(
            direct["scope"]["mean_case_fraction"]
        )
        scope_safe = (
            strict_vs_start >= -float(guard["maximum_strict_drop_vs_start"])
            and soft_vs_start >= -float(guard["maximum_case_fraction_drop_vs_start"])
            and strict_vs_direct >= -float(guard["maximum_strict_drop_vs_direct"])
            and soft_vs_direct >= -float(guard["maximum_case_fraction_drop_vs_direct"])
        )
        decisive = (
            gamma >= float(cfg["decisive_criterion"]["minimum_condition_gain"])
            and paired["candidate_wins"] > paired["direct_wins"]
            and float(paired["two_sided_exact_p"])
            <= float(cfg["decisive_criterion"]["paired_exact_p_value"])
            and scope_safe
        )
        item = {
            "branch_id": record["branch_id"],
            "credited_condition": credited_one_based,
            "signed_gamma_vs_direct": gamma,
            "paired_test": paired,
            "scope_gain_vs_start": {
                "strict": strict_vs_start,
                "mean_case_fraction": soft_vs_start,
            },
            "scope_gain_vs_direct": {
                "strict": strict_vs_direct,
                "mean_case_fraction": soft_vs_direct,
            },
            "scope_safe": scope_safe,
            "decisive": decisive,
        }
        record["comparison"] = item
        gains.append(item)

    values = [float(item["signed_gamma_vs_direct"]) for item in gains]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    advantages = [0.0] * len(values) if std < 1e-12 else [(value - mean) / std for value in values]
    for item, advantage in zip(gains, advantages):
        item["challenger_group_centered_advantage"] = advantage

    eligible = [item for item in gains if item["decisive"]]
    selected = max(eligible, key=lambda item: item["signed_gamma_vs_direct"]) if eligible else None
    selected_branch = selected["branch_id"] if selected else direct["branch_id"]
    outcome = {
        "protocol_version": cfg["protocol_version"],
        "status": "complete",
        "single_start": cfg["starting_solver"]["checkpoint"],
        "single_round": True,
        "multi_seed": False,
        "credited_condition": credited_one_based,
        "full_pass_credit_transition": transition,
        "cohort_endpoint_full_passes": total_endpoint_successes,
        "branches_with_endpoint_full_pass": branches_with_success,
        "records": records,
        "challenger_advantages": advantages,
        "decisive_condition_gain_found": bool(eligible),
        "selected_branch": selected_branch,
        "null_fallback_used": selected is None,
        "official_test_opened": False,
    }
    atomic_json(DEC_ROOT / "decision.json", outcome)
    atomic_json(
        EXP_ROOT / "manifests" / f"{VERSION}_complete.json",
        {
            "status": "complete",
            "decisive_condition_gain_found": bool(eligible),
            "selected_branch": selected_branch,
            "decision": str(DEC_ROOT / "decision.json"),
            "official_test_opened": False,
        },
    )

    lines = [
        "# Single-round decisive Mistral experiment",
        "",
        f"- Credited cumulative condition: c{credited_one_based}",
        f"- Full-pass credit transition: {transition}",
        f"- Decisive condition gain found: {bool(eligible)}",
        f"- Selected branch: `{selected_branch}`",
        f"- Null/direct fallback used: {selected is None}",
        "- Official test opened: false",
        "",
        "| branch | endpoint counts c1/c2/c3/c4/c5 | Gamma | exact p | scope safe | target train successes | generated tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, record in enumerate(records):
        comparison = record.get("comparison")
        lines.append(
            "| {branch} | {counts} | {gamma} | {p} | {safe} | {successes} | {tokens} |".format(
                branch=record["branch_id"],
                counts="/".join(map(str, record["endpoint_profile"]["counts"])),
                gamma="0 (direct)" if comparison is None else f"{comparison['signed_gamma_vs_direct']:.6f}",
                p="--" if comparison is None else f"{comparison['paired_test']['two_sided_exact_p']:.6g}",
                safe="--" if comparison is None else str(comparison["scope_safe"]),
                successes=record["training"]["target_training_successes"],
                tokens=record["training"]["training_generated_tokens"],
            )
        )
    (EXP_ROOT / "paper_outputs").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "paper_outputs" / f"RESULTS_DECISIVE_MISTRAL_{TAG.upper()}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
