from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import fisher_exact, t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "softphi_r2b__direct_strict_full"
LADDER = "softphi_r2b__stochastic_strict_ladder"


def partial(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    return int(row["passed_cases"]) / total if total else 0.0


def summarize(branch_id: str) -> tuple[dict[str, Any], dict[str, float]]:
    status = read_json(RAW_ROOT / branch_id / "status.json")
    if int(status["completed_updates"]) != 100:
        raise RuntimeError(f"incomplete branch: {branch_id}")
    rows = read_jsonl(RAW_ROOT / branch_id / "development" / "update_0100.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    if len(rows) != 800 or len(grouped) != 50 or set(map(len, grouped.values())) != {16}:
        raise RuntimeError(f"invalid endpoint for {branch_id}")
    rates = {key: mean(partial(row) for row in items) for key, items in grouped.items()}
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


def training_signal(branch_id: str) -> dict[str, Any]:
    records = []
    for update in range(1, 101):
        records.append(read_json(RAW_ROOT / branch_id / "train" / f"update_{update:04d}_summary.json"))
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_phase.setdefault(str(row["phase"]), []).append(row)
    return {
        phase: {
            "updates": len(rows),
            "updates_with_active_advantages": sum(int(row["active_advantages"]) > 0 for row in rows),
            "mean_binary_reward": mean(float(row["reward_mean"]) for row in rows),
            "total_strict_successes": sum(int(row["successes"]) for row in rows),
        }
        for phase, rows in by_phase.items()
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "soft_phi_round2_stochastic_ladder.json")
    direct, direct_rates = summarize(DIRECT)
    ladder, ladder_rates = summarize(LADDER)
    differences = [ladder_rates[key] - direct_rates[key] for key in sorted(direct_rates)]
    estimate = mean(differences)
    se = stdev(differences) / math.sqrt(len(differences))
    radius = float(t.ppf(0.975, len(differences) - 1)) * se
    soft_p = 1.0 if all(value == 0 for value in differences) else float(
        wilcoxon(differences, alternative="greater", zero_method="wilcox", method="auto").pvalue
    )
    table = [
        [ladder["strict_full_pass_count"], 800 - ladder["strict_full_pass_count"]],
        [direct["strict_full_pass_count"], 800 - direct["strict_full_pass_count"]],
    ]
    fisher = fisher_exact(table, alternative="greater")
    fisher_statistic = float(fisher.statistic)
    primary = bool(
        ladder["strict_full_pass_rate"] >= 0.01
        and ladder["strict_full_pass_rate"] > direct["strict_full_pass_rate"]
        and float(fisher.pvalue) < 0.01
    )
    secondary = bool(estimate > 0 and estimate - radius > 0 and soft_p < 0.05)
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "strict_bootstrap_positive" if primary else "strict_bootstrap_not_established",
        "endpoints": {"direct": direct, "strict_ladder": ladder},
        "effects": {
            "strict_rate_difference": ladder["strict_full_pass_rate"] - direct["strict_full_pass_rate"],
            "fisher_one_sided_odds_ratio": (
                fisher_statistic if math.isfinite(fisher_statistic) else None
            ),
            "fisher_one_sided_p": float(fisher.pvalue),
            "soft_phi_gamma": estimate,
            "soft_phi_ci95_student_t": [estimate - radius, estimate + radius],
            "soft_phi_wilcoxon_one_sided_p": soft_p,
            "positive_instances": sum(value > 0 for value in differences),
            "negative_instances": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "median_instance_difference": median(differences),
        },
        "training_signal": {"direct": training_signal(DIRECT), "strict_ladder": training_signal(LADDER)},
        "decision": {
            "primary_strict_positive": primary,
            "secondary_soft_phi_positive": secondary,
            "confirmation_opened": False,
            "official_test_opened": False,
            "no_new_audit_hash_or_multiseed": True,
        },
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "soft_phi_round2_strict_ladder.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / "soft_phi_round2_complete.json",
        {
            "status": result["status"],
            "primary_strict_positive": primary,
            "secondary_soft_phi_positive": secondary,
            "summary": "aggregated_results/soft_phi_round2_strict_ladder.json",
        },
    )
    lines = [
        "# Soft-Phi self-grok Round 2: strict binary verifier ladder",
        "",
        f"- Direct: strict={direct['strict_full_pass_count']}/800, soft Phi={direct['soft_phi']:.6f}",
        f"- Ladder: strict={ladder['strict_full_pass_count']}/800, soft Phi={ladder['soft_phi']:.6f}",
        f"- Strict difference={result['effects']['strict_rate_difference']:.6f}, Fisher p={float(fisher.pvalue):.6g}",
        f"- Soft-Phi Gamma={estimate:.6f}, CI=[{estimate-radius:.6f}, {estimate+radius:.6f}], Wilcoxon p={soft_p:.6g}",
        f"- Primary strict positive: {primary}",
        f"- Secondary mechanism positive: {secondary}",
        "- Confirmation and official test remain sealed.",
        "",
    ]
    (EXP_ROOT / "RESULTS_SOFT_PHI_ROUND2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
