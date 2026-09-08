from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact, t

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "recipe__direct__has__seed42"
DENSE = "recipe__decisive__regex_dense"
UPDATE = 100


def load_endpoint(branch_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = RAW_ROOT / branch_id / "development"
    return (
        read_json(root / f"update_{UPDATE:04d}_summary.json"),
        read_jsonl(root / f"update_{UPDATE:04d}.jsonl"),
    )


def instance_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(int(row["reward"]))
    return {
        instance_id: sum(values) / len(values)
        for instance_id, values in grouped.items()
    }


def paired_t_interval(
    dense: dict[str, float],
    direct: dict[str, float],
) -> dict[str, float | int]:
    ids = sorted(set(dense) & set(direct))
    if not ids:
        raise RuntimeError("no common development instances")
    deltas = [dense[item] - direct[item] for item in ids]
    mean = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        return {
            "estimate": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "instances": len(deltas),
        }
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    sem = math.sqrt(variance / len(deltas))
    radius = float(t.ppf(0.975, len(deltas) - 1)) * sem
    return {
        "estimate": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "instances": len(deltas),
    }


def counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    successes = sum(int(row["reward"]) for row in rows)
    return successes, len(rows) - successes


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "decisive_branches.json")
    screen = read_json(EXP_ROOT / "manifests" / "recipe_screen.json")
    direct_summary, direct_rows = load_endpoint(DIRECT)
    dense_summary, dense_rows = load_endpoint(DENSE)
    expected = int(
        protocol["fixed_development_evaluation"]["total_rollouts_per_branch"]
    )
    if len(direct_rows) != expected or len(dense_rows) != expected:
        raise RuntimeError(
            f"incomplete endpoint: direct={len(direct_rows)}, dense={len(dense_rows)}"
        )

    direct_success, direct_failure = counts(direct_rows)
    dense_success, dense_failure = counts(dense_rows)
    fisher = fisher_exact(
        [[dense_success, dense_failure], [direct_success, direct_failure]],
        alternative="greater",
    )
    paired = paired_t_interval(
        instance_rates(dense_rows),
        instance_rates(direct_rows),
    )
    criterion = protocol["predeclared_decisive_criterion"]
    audit_zero = bool(
        screen["target_audit"]["full_pass_count"] == 0
        and screen["target_audit"]["rollouts"] == 4096
    )
    dense_rate = dense_success / len(dense_rows)
    direct_rate = direct_success / len(direct_rows)
    decisive = bool(
        audit_zero
        and dense_rate >= float(criterion["dense_endpoint_min_full_pass_rate"])
        and dense_rate > direct_rate
        and float(fisher.pvalue)
        < float(criterion["one_sided_fisher_exact_p_below"])
    )
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "positive" if decisive else "null_under_frozen_recipe",
        "audit_zero": audit_zero,
        "endpoint_update": UPDATE,
        "endpoint": {
            "direct": {
                "successes": direct_success,
                "rollouts": len(direct_rows),
                "full_pass_rate": direct_rate,
                "summary": direct_summary,
            },
            "dense_regex": {
                "successes": dense_success,
                "rollouts": len(dense_rows),
                "full_pass_rate": dense_rate,
                "summary": dense_summary,
            },
        },
        "effect": {
            "absolute_rate_difference": dense_rate - direct_rate,
            "rate_ratio": (
                dense_rate / direct_rate if direct_rate > 0 else None
            ),
            "one_sided_fisher_exact_odds_ratio": float(fisher.statistic),
            "one_sided_fisher_exact_p": float(fisher.pvalue),
            "paired_instance_t_interval": paired,
        },
        "decision": {
            "decisive_transfer_witness": decisive,
            "criterion": criterion,
            "official_test_remains_sealed": True,
            "no_seed_or_hash_search": True,
        },
    }
    atomic_json(
        EXP_ROOT / "aggregated_results" / "decisive_summary.json",
        result,
    )
    atomic_json(
        EXP_ROOT / "manifests" / "decisive_run_complete.json",
        {"status": result["status"], "summary": "aggregated_results/decisive_summary.json"},
    )
    if decisive:
        atomic_json(
            EXP_ROOT / "manifests" / "decisive_experiment_complete.json",
            {
                "status": "positive",
                "claim_scope": protocol["scope"]["claim_if_positive"],
                "official_test_remains_sealed": True,
            },
        )

    lines = [
        "# Decisive HAS transfer experiment",
        "",
        f"- Target audit: {screen['target_audit']['full_pass_count']}/"
        f"{screen['target_audit']['rollouts']} strict full passes",
        f"- Direct HAS after {UPDATE} updates: {direct_success}/{len(direct_rows)} "
        f"({direct_rate:.6f})",
        f"- Dense REGEX curriculum after {UPDATE} updates: "
        f"{dense_success}/{len(dense_rows)} ({dense_rate:.6f})",
        f"- Absolute difference: {dense_rate - direct_rate:.6f}",
        f"- One-sided Fisher exact p: {float(fisher.pvalue):.6g}",
        f"- Paired-instance difference: {paired['estimate']:.6f} "
        f"[{paired['ci95_low']:.6f}, {paired['ci95_high']:.6f}]",
        f"- Decisive transfer witness: {decisive}",
        "",
        "The sealed official test was not opened. No seed or hash search was run.",
        "",
    ]
    (EXP_ROOT / "RESULTS_DECISIVE.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
