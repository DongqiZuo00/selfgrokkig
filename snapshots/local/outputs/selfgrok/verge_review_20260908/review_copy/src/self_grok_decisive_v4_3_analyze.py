from __future__ import annotations

import json
import math
import os
from typing import Any

from datasets import load_from_disk

from common import EXP_ROOT, atomic_json, read_json, read_jsonl
from prepend_conditions import condition_profile


VERSION = os.environ.get(
    "SELF_GROK_DECISIVE_VERSION", "self_grok_decisive_mistral_v4_3"
)
VERSION_TAG = VERSION.rsplit("mistral_", 1)[-1]
CFG = EXP_ROOT / "manifests" / f"{VERSION}.json"
RUNTIME = EXP_ROOT / "manifests" / f"{VERSION}_runtime_branches.json"
ROOT = EXP_ROOT / "raw_results" / VERSION
RECIPE = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    low = min(successes, trials - successes)
    return min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(low + 1)) / (2**trials))


def paired(candidate: dict[str, Any], direct: dict[str, Any], index: int) -> dict[str, Any]:
    wins = losses = ties = 0
    for key, vector in candidate["keyed_vectors"].items():
        delta = int(vector[index]) - int(direct["keyed_vectors"][key][index])
        wins += int(delta > 0)
        losses += int(delta < 0)
        ties += int(delta == 0)
    return {
        "oracle_wins": wins,
        "direct_wins": losses,
        "ties": ties,
        "two_sided_exact_p": binomial_two_sided(wins, wins + losses),
    }


def compact(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "keyed_vectors"}


def main() -> None:
    cfg = read_json(CFG)
    runtime = read_json(RUNTIME)
    goal = int(cfg["solver"]["matched_updates"])
    selection = [dict(row) for row in load_from_disk(str(EXP_ROOT / cfg["target"]["selection_split"]))]
    statuses = {
        branch["branch_id"]: read_json(RECIPE / branch["branch_id"] / "status.json")
        for branch in runtime["branches"]
    }
    direct_id, oracle_id = runtime["branch_ids"]
    direct_status = statuses[direct_id]
    direct_goal = int(cfg.get("direct_completed_updates", goal))
    if (
        direct_status["status"] != "complete"
        or int(direct_status["completed_updates"]) != direct_goal
    ):
        raise RuntimeError("direct/null arm did not finish")
    direct_rows = read_jsonl(
        RECIPE / direct_id / "grounded_reward" / f"update_{direct_goal:04d}.jsonl"
    )
    direct_profile = condition_profile(direct_rows, selection)
    direct_scope = read_json(
        RECIPE
        / direct_id
        / "scope_probe"
        / f"update_{direct_goal:04d}_summary.json"
    )
    oracle_status = statuses[oracle_id]
    starting = read_json(ROOT / "starting_profile.json")
    credited = int(starting["bottleneck_index"])
    result: dict[str, Any] = {
        "protocol_version": VERSION,
        "status": "complete",
        "single_start": cfg["single_start"],
        "single_seed": True,
        "local_oracle_updates": goal,
        "direct_control_updates": direct_goal,
        "credited_condition": credited,
        "direct": {
            "profile": compact(direct_profile),
            "scope_strict": float(direct_scope["full_pass_rate"]),
            "scope_soft": float(direct_scope["mean_case_fraction"]),
            "training": direct_status,
        },
        "oracle_training": oracle_status,
        "official_test_opened": False,
    }
    minimal_positive = False
    if oracle_status["status"] == "complete" and int(oracle_status["completed_updates"]) == goal:
        oracle_rows = read_jsonl(RECIPE / oracle_id / "grounded_reward" / f"update_{goal:04d}.jsonl")
        oracle_profile = condition_profile(oracle_rows, selection)
        oracle_scope = read_json(RECIPE / oracle_id / "scope_probe" / f"update_{goal:04d}_summary.json")
        gamma = float(oracle_profile["rates"][credited - 1]) - float(direct_profile["rates"][credited - 1])
        scope_strict_gain = float(oracle_scope["full_pass_rate"]) - float(direct_scope["full_pass_rate"])
        scope_soft_gain = float(oracle_scope["mean_case_fraction"]) - float(direct_scope["mean_case_fraction"])
        scope_safe = scope_strict_gain >= -0.05 and scope_soft_gain >= -0.05
        test = paired(oracle_profile, direct_profile, credited - 1)
        minimal_positive = gamma > float(cfg["minimal_result"]["positive_condition_gain"]) and scope_safe
        result["oracle"] = {
            "profile": compact(oracle_profile),
            "scope_strict": float(oracle_scope["full_pass_rate"]),
            "scope_soft": float(oracle_scope["mean_case_fraction"]),
            "signed_gamma_vs_direct": gamma,
            "paired_test": test,
            "scope_gain_vs_direct": {"strict": scope_strict_gain, "soft": scope_soft_gain},
            "scope_safe": scope_safe,
        }
    else:
        result["oracle"] = {
            "not_evaluated": True,
            "reason": oracle_status["status"],
            "stop_event": oracle_status.get("stop_event"),
        }
    result["minimal_positive_result"] = minimal_positive
    result["selected_branch"] = oracle_id if minimal_positive else direct_id
    result["null_fallback_used"] = not minimal_positive
    atomic_json(ROOT / "decision.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / f"{VERSION}_complete.json",
        {
            "status": "complete",
            "minimal_positive_result": minimal_positive,
            "selected_branch": result["selected_branch"],
            "official_test_opened": False,
        },
    )
    lines = [
        f"# Mistral {VERSION_TAG.replace('_', '.')} minimal repaired-ladder result",
        "",
        f"- Oracle status: `{oracle_status['status']}` after {oracle_status['completed_updates']}/{goal} updates",
        f"- Credited condition: c{credited}",
        f"- Minimal positive result: {minimal_positive}",
        f"- Selected branch: `{result['selected_branch']}`",
        "- Official test opened: false",
    ]
    if "profile" in result["oracle"]:
        lines.extend(
            [
                f"- Direct endpoint counts: {direct_profile['counts']}",
                f"- Oracle endpoint counts: {result['oracle']['profile']['counts']}",
                f"- Signed Gamma: {result['oracle']['signed_gamma_vs_direct']:.6f}",
                f"- Scope safe: {result['oracle']['scope_safe']}",
                f"- Paired exact p: {result['oracle']['paired_test']['two_sided_exact_p']:.6g}",
            ]
        )
    else:
        lines.append(f"- Safe early stop: {result['oracle']['reason']}")
    (EXP_ROOT / "paper_outputs").mkdir(parents=True, exist_ok=True)
    (
        EXP_ROOT
        / "paper_outputs"
        / f"RESULTS_DECISIVE_MISTRAL_{VERSION_TAG.upper()}.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
