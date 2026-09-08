from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any

from scipy.stats import fisher_exact, t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "sg_strict_r4__direct_from_r3u100"
ORACLE = "sg_strict_r4__oracle_from_r3u100"
CHALLENGER = "sg_strict_r4__challenger_regex_from_r3u100"
BRANCHES = (DIRECT, ORACLE, CHALLENGER)


def partial(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    return int(row["passed_cases"]) / total if total else 0.0


def endpoint(branch_id: str, split: str, update: int, instances: int, samples: int):
    rows = read_jsonl(RAW_ROOT / branch_id / split / f"update_{update:04d}.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    if len(rows) != instances * samples or len(grouped) != instances:
        raise RuntimeError(
            f"invalid {split} endpoint {branch_id} u{update}: "
            f"rows={len(rows)}, instances={len(grouped)}"
        )
    if set(map(len, grouped.values())) != {samples}:
        raise RuntimeError(f"unbalanced endpoint {branch_id} {split} u{update}")
    rates = {
        key: mean(partial(row) for row in values)
        for key, values in grouped.items()
    }
    strict = sum(int(row["reward"]) for row in rows)
    return {
        "rollouts": len(rows),
        "instances": len(grouped),
        "strict_full_pass_count": strict,
        "strict_full_pass_rate": strict / len(rows),
        "soft_phi": mean(rates.values()),
        "parse_valid_rate": sum(bool(row["parse_valid"]) for row in rows) / len(rows),
        "near_strict_ge90pct_count": sum(0.9 <= partial(row) < 1.0 for row in rows),
    }, rates


def paired_effect(treatment: dict[str, float], control: dict[str, float]) -> dict[str, Any]:
    if set(treatment) != set(control):
        raise RuntimeError("paired endpoint IDs differ")
    differences = [treatment[key] - control[key] for key in sorted(control)]
    estimate = mean(differences)
    radius = 0.0
    if len(differences) > 1 and stdev(differences) > 0:
        radius = float(t.ppf(0.975, len(differences) - 1)) * stdev(differences) / math.sqrt(len(differences))
    p_value = 1.0 if all(value == 0 for value in differences) else float(
        wilcoxon(differences, alternative="greater", zero_method="wilcox", method="auto").pvalue
    )
    return {
        "estimate": estimate,
        "ci95_student_t": [estimate - radius, estimate + radius],
        "wilcoxon_one_sided_p": p_value,
        "positive_instances": sum(value > 0 for value in differences),
        "negative_instances": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def strict_effect(treatment: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    table = [
        [treatment["strict_full_pass_count"], treatment["rollouts"] - treatment["strict_full_pass_count"]],
        [control["strict_full_pass_count"], control["rollouts"] - control["strict_full_pass_count"]],
    ]
    result = fisher_exact(table, alternative="greater")
    odds = float(result.statistic)
    return {
        "rate_difference": treatment["strict_full_pass_rate"] - control["strict_full_pass_rate"],
        "odds_ratio": odds if math.isfinite(odds) else None,
        "fisher_one_sided_p": float(result.pvalue),
    }


def training_signal(branch_id: str) -> dict[str, Any]:
    records = [
        read_json(RAW_ROOT / branch_id / "train" / f"update_{update:04d}_summary.json")
        for update in range(1, 101)
    ]
    return {
        "updates": len(records),
        "rollouts": sum(int(row["rollouts"]) for row in records),
        "strict_successes": sum(int(row["successes"]) for row in records),
        "updates_with_strict_success": sum(int(row["successes"]) > 0 for row in records),
        "updates_with_active_advantages": sum(int(row["active_advantages"]) > 0 for row in records),
        "zero_advantage_updates": sum(int(row["active_advantages"]) == 0 for row in records),
        "mean_binary_reward": mean(float(row["reward_mean"]) for row in records),
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "self_grok_strict_handoff_round4.json")
    for branch_id in BRANCHES:
        status = read_json(RAW_ROOT / branch_id / "status.json")
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != 100:
            raise RuntimeError(f"incomplete branch: {branch_id}")
        if not bool(status.get("optimizer_inherited_from_parent")):
            raise RuntimeError(f"optimizer inheritance was not recorded: {branch_id}")

    endpoints: dict[str, Any] = {}
    target_rates: dict[str, dict[int, dict[str, float]]] = {}
    scope_rates: dict[str, dict[int, dict[str, float]]] = {}
    for branch_id in BRANCHES:
        endpoints[branch_id] = {"target": {}, "scope_probe": {}}
        target_rates[branch_id] = {}
        scope_rates[branch_id] = {}
        for update in (0, 100):
            target_summary, target_rate = endpoint(branch_id, "development", update, 50, 16)
            scope_summary, scope_rate = endpoint(branch_id, "scope_probe", update, 32, 8)
            endpoints[branch_id]["target"][str(update)] = target_summary
            endpoints[branch_id]["scope_probe"][str(update)] = scope_summary
            target_rates[branch_id][update] = target_rate
            scope_rates[branch_id][update] = scope_rate

    comparisons: dict[str, Any] = {}
    for name, branch_id in (("oracle", ORACLE), ("challenger_regex", CHALLENGER)):
        comparisons[name] = {}
        for update in (0, 100):
            comparisons[name][str(update)] = {
                "target_soft_phi": paired_effect(target_rates[branch_id][update], target_rates[DIRECT][update]),
                "target_strict": strict_effect(
                    endpoints[branch_id]["target"][str(update)],
                    endpoints[DIRECT]["target"][str(update)],
                ),
                "scope_soft_phi": paired_effect(scope_rates[branch_id][update], scope_rates[DIRECT][update]),
            }

    direct_final = endpoints[DIRECT]["target"]["100"]
    oracle_final = endpoints[ORACLE]["target"]["100"]
    challenger_final = endpoints[CHALLENGER]["target"]["100"]
    challenger_scope = comparisons["challenger_regex"]["100"]["scope_soft_phi"]
    oracle_reachable = oracle_final["strict_full_pass_count"] > 0
    challenger_strict_gain = (
        challenger_final["strict_full_pass_rate"] > direct_final["strict_full_pass_rate"]
    )
    scope_noninferior = challenger_scope["ci95_student_t"][0] > -0.05
    decisive_self_grok = challenger_strict_gain and scope_noninferior

    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete",
        "endpoints": endpoints,
        "comparisons_vs_matched_direct": comparisons,
        "training_signal": {branch_id: training_signal(branch_id) for branch_id in BRANCHES},
        "decision": {
            "oracle_reachable": oracle_reachable,
            "challenger_strict_gain_over_direct": challenger_strict_gain,
            "challenger_scope_noninferior": scope_noninferior,
            "decisive_single_seed_self_grok_phenomenon": decisive_self_grok,
            "confirmation_opened": False,
            "official_test_opened": False,
            "automatic_extension_submitted": False,
        },
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "self_grok_strict_handoff_round4.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / "self_grok_strict_handoff_round4_complete.json",
        {
            "status": "complete",
            **result["decision"],
            "summary": "aggregated_results/self_grok_strict_handoff_round4.json",
            "all_three_branches_complete": True,
        },
    )

    lines = [
        "# SELF-GROK strict handoff Round 4",
        "",
        f"- Direct: strict={direct_final['strict_full_pass_count']}/{direct_final['rollouts']}, soft Phi={direct_final['soft_phi']:.6f}",
        f"- Oracle Recipe: strict={oracle_final['strict_full_pass_count']}/{oracle_final['rollouts']}, soft Phi={oracle_final['soft_phi']:.6f}",
        f"- Challenger REGEX: strict={challenger_final['strict_full_pass_count']}/{challenger_final['rollouts']}, soft Phi={challenger_final['soft_phi']:.6f}",
        f"- Challenger strict rate difference vs direct={comparisons['challenger_regex']['100']['target_strict']['rate_difference']:.6f}, Fisher one-sided p={comparisons['challenger_regex']['100']['target_strict']['fisher_one_sided_p']:.6g}",
        f"- Challenger scope soft-Phi difference vs direct={challenger_scope['estimate']:.6f}, CI=[{challenger_scope['ci95_student_t'][0]:.6f}, {challenger_scope['ci95_student_t'][1]:.6f}]",
        f"- Oracle reachable={oracle_reachable}; decisive single-seed SELF-GROK phenomenon={decisive_self_grok}",
        "- Confirmation and official test remain sealed. No follow-on job was submitted automatically.",
        "",
    ]
    (EXP_ROOT / "RESULTS_SELF_GROK_STRICT_HANDOFF_ROUND4.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
