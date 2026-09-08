from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import fisher_exact, t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "sg_main_r3__direct_strict"
ORACLE = "sg_main_r3__oracle_has_dense40_strict60"
CANDIDATES = {
    "official_regex_dense": "sg_main_r3__challenger_regex_dense40_strict60",
    "starts_with__medium": "sg_main_r3__challenger_starts_medium_dense40_strict60",
}


def partial(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    return int(row["passed_cases"]) / total if total else 0.0


def endpoint(branch_id: str, split: str, update: int, expected_instances: int, samples: int):
    rows = read_jsonl(
        RAW_ROOT / branch_id / split / f"update_{update:04d}.jsonl"
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    expected = expected_instances * samples
    if (
        len(rows) != expected
        or len(grouped) != expected_instances
        or set(map(len, grouped.values())) != {samples}
    ):
        raise RuntimeError(
            f"invalid {split} endpoint for {branch_id} at {update}: "
            f"rows={len(rows)}, instances={len(grouped)}"
        )
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
    if len(differences) > 1 and stdev(differences) > 0:
        radius = float(t.ppf(0.975, len(differences) - 1)) * stdev(differences) / math.sqrt(len(differences))
    else:
        radius = 0.0
    p_value = 1.0 if all(value == 0 for value in differences) else float(
        wilcoxon(
            differences,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )
    return {
        "estimate": estimate,
        "ci95_student_t": [estimate - radius, estimate + radius],
        "wilcoxon_one_sided_p": p_value,
        "positive_instances": sum(value > 0 for value in differences),
        "negative_instances": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "median_instance_difference": median(differences),
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
        "fisher_one_sided_odds_ratio": odds if math.isfinite(odds) else None,
        "fisher_one_sided_p": float(result.pvalue),
    }


def training_signal(branch_id: str) -> dict[str, Any]:
    records = [
        read_json(RAW_ROOT / branch_id / "train" / f"update_{update:04d}_summary.json")
        for update in range(1, 101)
    ]
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_phase.setdefault(str(record["phase"]), []).append(record)
    return {
        phase: {
            "updates": len(values),
            "updates_with_active_advantages": sum(int(row["active_advantages"]) > 0 for row in values),
            "mean_optimization_reward": mean(float(row["optimization_reward_mean"]) for row in values),
            "mean_binary_full_pass_reward": mean(float(row["reward_mean"]) for row in values),
            "total_strict_successes": sum(int(row["successes"]) for row in values),
        }
        for phase, values in by_phase.items()
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "self_grok_main_round3.json")
    branch_ids = [DIRECT, ORACLE, *CANDIDATES.values()]
    for branch_id in branch_ids:
        status = read_json(RAW_ROOT / branch_id / "status.json")
        if int(status["completed_updates"]) != 100 or status["status"] != "complete":
            raise RuntimeError(f"incomplete branch: {branch_id}")

    endpoints: dict[str, Any] = {}
    target_rates: dict[str, dict[int, dict[str, float]]] = {}
    scope_rates: dict[str, dict[int, dict[str, float]]] = {}
    for branch_id in branch_ids:
        endpoints[branch_id] = {"target": {}, "scope_probe": {}}
        target_rates[branch_id] = {}
        scope_rates[branch_id] = {}
        for update in (40, 100):
            target_summary, target_rate = endpoint(branch_id, "development", update, 50, 16)
            scope_summary, scope_rate = endpoint(branch_id, "scope_probe", update, 32, 8)
            endpoints[branch_id]["target"][str(update)] = target_summary
            endpoints[branch_id]["scope_probe"][str(update)] = scope_summary
            target_rates[branch_id][update] = target_rate
            scope_rates[branch_id][update] = scope_rate

    comparisons: dict[str, Any] = {}
    for name, branch_id in {"oracle_has_dense": ORACLE, **CANDIDATES}.items():
        comparisons[name] = {}
        for update in (40, 100):
            treatment = endpoints[branch_id]["target"][str(update)]
            control = endpoints[DIRECT]["target"][str(update)]
            comparisons[name][str(update)] = {
                "target_soft_phi": paired_effect(target_rates[branch_id][update], target_rates[DIRECT][update]),
                "target_strict": strict_effect(treatment, control),
                "scope_soft_phi": paired_effect(scope_rates[branch_id][update], scope_rates[DIRECT][update]),
            }

    margin = float(protocol["scope_guard"]["noninferiority_margin"])
    eligible: list[tuple[float, str, str]] = []
    for candidate_id, branch_id in CANDIDATES.items():
        target = comparisons[candidate_id]["100"]["target_soft_phi"]
        scope = comparisons[candidate_id]["100"]["scope_soft_phi"]
        target_positive = bool(
            target["estimate"] > 0
            and target["ci95_student_t"][0] > 0
            and target["wilcoxon_one_sided_p"] < 0.05
        )
        scope_noninferior = bool(scope["ci95_student_t"][0] > margin)
        comparisons[candidate_id]["decision"] = {
            "target_positive": target_positive,
            "scope_noninferior": scope_noninferior,
            "eligible": target_positive and scope_noninferior,
        }
        if target_positive and scope_noninferior:
            eligible.append((float(target["estimate"]), candidate_id, branch_id))

    eligible.sort(reverse=True)
    selected = eligible[0] if eligible else None
    oracle_target = comparisons["oracle_has_dense"]["100"]["target_soft_phi"]
    oracle_strict = comparisons["oracle_has_dense"]["100"]["target_strict"]
    oracle_positive = bool(
        oracle_strict["rate_difference"] > 0
        or (
            oracle_target["estimate"] > 0
            and oracle_target["ci95_student_t"][0] > 0
            and oracle_target["wilcoxon_one_sided_p"] < 0.05
        )
    )
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "challenger_bridge_selected" if selected else "no_challenger_bridge_selected",
        "endpoints": endpoints,
        "comparisons_vs_matched_direct": comparisons,
        "training_signal": {branch_id: training_signal(branch_id) for branch_id in branch_ids},
        "decision": {
            "oracle_positive": oracle_positive,
            "selected_candidate": selected[1] if selected else None,
            "selected_branch": selected[2] if selected else DIRECT,
            "challenger_proposal_adapted_from_target_feedback": True,
            "challenger_learning_claimed": bool(selected),
            "confirmation_opened": False,
            "official_test_opened": False,
            "no_new_audit_hash_config_or_seed_search": True,
        },
    }
    output = EXP_ROOT / "aggregated_results" / "self_grok_main_round3.json"
    atomic_json(output, result)
    atomic_json(
        EXP_ROOT / "manifests" / "self_grok_main_round3_complete.json",
        {
            "status": result["status"],
            "oracle_positive": oracle_positive,
            "selected_candidate": result["decision"]["selected_candidate"],
            "summary": "aggregated_results/self_grok_main_round3.json",
            "all_four_branches_complete": True,
            "confirmation_and_official_test_sealed": True,
        },
    )
    lines = [
        "# SELF-GROK main experiment: target-aware Challenger Round 3",
        "",
        f"- Matched direct endpoint: strict={endpoints[DIRECT]['target']['100']['strict_full_pass_count']}/800, soft Phi={endpoints[DIRECT]['target']['100']['soft_phi']:.6f}",
        f"- HAS-dense oracle endpoint: strict={endpoints[ORACLE]['target']['100']['strict_full_pass_count']}/800, soft Phi={endpoints[ORACLE]['target']['100']['soft_phi']:.6f}",
        f"- Oracle target soft-Phi Gamma={oracle_target['estimate']:.6f}, CI=[{oracle_target['ci95_student_t'][0]:.6f}, {oracle_target['ci95_student_t'][1]:.6f}]",
    ]
    for candidate_id, branch_id in CANDIDATES.items():
        target = comparisons[candidate_id]["100"]["target_soft_phi"]
        scope = comparisons[candidate_id]["100"]["scope_soft_phi"]
        decision = comparisons[candidate_id]["decision"]
        summary = endpoints[branch_id]["target"]["100"]
        lines.append(
            f"- {candidate_id}: strict={summary['strict_full_pass_count']}/800, soft Phi={summary['soft_phi']:.6f}, "
            f"Gamma={target['estimate']:.6f} CI=[{target['ci95_student_t'][0]:.6f}, {target['ci95_student_t'][1]:.6f}], "
            f"scope Gamma={scope['estimate']:.6f}, eligible={decision['eligible']}"
        )
    lines.extend(
        [
            f"- Selected Challenger bridge: {result['decision']['selected_candidate']}",
            "- Confirmation and official test remain sealed.",
            "",
        ]
    )
    report = EXP_ROOT / "RESULTS_SELF_GROK_MAIN_ROUND3.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
