from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any

from scipy.stats import fisher_exact, t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "sg_consolidate_r5__direct"
CHALLENGER = "sg_consolidate_r5__challenger_regex"
UPDATES = (0, 40, 80, 100)


def partial(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    return int(row["passed_cases"]) / total if total else 0.0


def endpoint(branch_id: str, split: str, update: int, instances: int, samples: int):
    rows = read_jsonl(RAW_ROOT / branch_id / split / f"update_{update:04d}.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    if len(rows) != instances * samples or len(grouped) != instances:
        raise RuntimeError(f"invalid endpoint {branch_id} {split} u{update}")
    if set(map(len, grouped.values())) != {samples}:
        raise RuntimeError(f"unbalanced endpoint {branch_id} {split} u{update}")
    rates = {key: mean(partial(row) for row in values) for key, values in grouped.items()}
    strict = sum(int(row["reward"]) for row in rows)
    return {
        "rollouts": len(rows),
        "strict_full_pass_count": strict,
        "strict_full_pass_rate": strict / len(rows),
        "soft_phi": mean(rates.values()),
        "parse_valid_rate": sum(bool(row["parse_valid"]) for row in rows) / len(rows),
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
    records = [read_json(RAW_ROOT / branch_id / "train" / f"update_{u:04d}_summary.json") for u in range(1, 101)]
    successful = [row for row in records if int(row["successes"]) > 0]
    return {
        "updates": 100,
        "rollouts": sum(int(row["rollouts"]) for row in records),
        "strict_successes": sum(int(row["successes"]) for row in records),
        "updates_with_strict_success": len(successful),
        "first_strict_success_update": int(successful[0]["update"]) if successful else None,
        "updates_with_active_advantages": sum(int(row["active_advantages"]) > 0 for row in records),
        "final_replay_pool_size": int(records[-1]["success_replay_pool_size"]),
        "total_replay_examples_used": int(records[-1]["success_replay_examples_used"]),
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "self_grok_consolidation_round5.json")
    for branch_id in (DIRECT, CHALLENGER):
        status = read_json(RAW_ROOT / branch_id / "status.json")
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != 100:
            raise RuntimeError(f"incomplete branch: {branch_id}")

    endpoints: dict[str, Any] = {}
    target_rates: dict[str, dict[int, dict[str, float]]] = {}
    scope_rates: dict[str, dict[int, dict[str, float]]] = {}
    for branch_id in (DIRECT, CHALLENGER):
        endpoints[branch_id] = {"target": {}, "scope_probe": {}}
        target_rates[branch_id] = {}
        scope_rates[branch_id] = {}
        for update in UPDATES:
            target, target_rate = endpoint(branch_id, "development", update, 50, 16)
            scope, scope_rate = endpoint(branch_id, "scope_probe", update, 32, 8)
            endpoints[branch_id]["target"][str(update)] = target
            endpoints[branch_id]["scope_probe"][str(update)] = scope
            target_rates[branch_id][update] = target_rate
            scope_rates[branch_id][update] = scope_rate

    comparisons: dict[str, Any] = {}
    safe_updates: list[int] = []
    margin = float(protocol["scope_guard"]["noninferiority_margin"])
    for update in UPDATES:
        scope = paired_effect(scope_rates[CHALLENGER][update], scope_rates[DIRECT][update])
        target = paired_effect(target_rates[CHALLENGER][update], target_rates[DIRECT][update])
        strict = strict_effect(
            endpoints[CHALLENGER]["target"][str(update)],
            endpoints[DIRECT]["target"][str(update)],
        )
        scope_safe = scope["ci95_student_t"][0] > margin
        comparisons[str(update)] = {
            "target_soft_phi": target,
            "target_strict": strict,
            "scope_soft_phi": scope,
            "scope_safe": scope_safe,
        }
        if scope_safe:
            safe_updates.append(update)

    selected_update = max(safe_updates) if safe_updates else 0
    direct_selected = endpoints[DIRECT]["target"][str(selected_update)]
    challenger_selected = endpoints[CHALLENGER]["target"][str(selected_update)]
    challenger_strict_gain = (
        challenger_selected["strict_full_pass_rate"] > direct_selected["strict_full_pass_rate"]
    )
    any_dev_strict = any(
        endpoints[branch]["target"][str(update)]["strict_full_pass_count"] > 0
        for branch in (DIRECT, CHALLENGER)
        for update in UPDATES
    )
    decision = {
        "selected_scope_safe_update": selected_update,
        "challenger_strict_gain_over_direct": challenger_strict_gain,
        "decisive_single_seed_self_grok_phenomenon": challenger_strict_gain,
        "any_development_strict_success": any_dev_strict,
        "bridge_chain_required": not any_dev_strict,
        "confirmation_opened": False,
        "official_test_opened": False,
    }
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete",
        "endpoints": endpoints,
        "comparisons": comparisons,
        "training_signal": {branch: training_signal(branch) for branch in (DIRECT, CHALLENGER)},
        "decision": decision,
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "self_grok_consolidation_round5.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / "self_grok_consolidation_round5_complete.json",
        {"status": "complete", **decision, "summary": "aggregated_results/self_grok_consolidation_round5.json"},
    )
    lines = [
        "# SELF-GROK consolidation Round 5",
        "",
        f"- Selected scope-safe update: {selected_update}",
        f"- Direct strict: {direct_selected['strict_full_pass_count']}/{direct_selected['rollouts']}; soft Phi={direct_selected['soft_phi']:.6f}",
        f"- Challenger strict: {challenger_selected['strict_full_pass_count']}/{challenger_selected['rollouts']}; soft Phi={challenger_selected['soft_phi']:.6f}",
        f"- Strict rate difference: {comparisons[str(selected_update)]['target_strict']['rate_difference']:.6f}; Fisher p={comparisons[str(selected_update)]['target_strict']['fisher_one_sided_p']:.6g}",
        f"- Scope difference: {comparisons[str(selected_update)]['scope_soft_phi']['estimate']:.6f}; CI={comparisons[str(selected_update)]['scope_soft_phi']['ci95_student_t']}",
        f"- Decisive SELF-GROK phenomenon: {challenger_strict_gain}",
        f"- Bridge chain required: {not any_dev_strict}",
        "- Confirmation and official test remain sealed.",
        "",
    ]
    (EXP_ROOT / "RESULTS_SELF_GROK_CONSOLIDATION_ROUND5.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
