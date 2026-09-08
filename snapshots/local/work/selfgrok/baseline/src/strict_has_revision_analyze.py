from __future__ import annotations

import math
from typing import Any

from scipy.stats import fisher_exact, t

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "recipe__direct__has__seed42"
DENSE = "recipe__decisive__regex_dense"
DIRECT_UPDATE = 100
DENSE_UPDATE = 200


def load_endpoint(
    branch_id: str,
    update: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = RAW_ROOT / branch_id / "development"
    return (
        read_json(root / f"update_{update:04d}_summary.json"),
        read_jsonl(root / f"update_{update:04d}.jsonl"),
    )


def instance_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(int(row["reward"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def paired_interval(
    dense: dict[str, float],
    direct: dict[str, float],
) -> dict[str, float | int]:
    ids = sorted(set(dense) & set(direct))
    deltas = [dense[key] - direct[key] for key in ids]
    mean = sum(deltas) / len(deltas)
    variance = (
        sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        if len(deltas) > 1
        else 0.0
    )
    radius = (
        float(t.ppf(0.975, len(deltas) - 1))
        * math.sqrt(variance / len(deltas))
        if len(deltas) > 1
        else 0.0
    )
    return {
        "estimate": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "instances": len(deltas),
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "strict_has_revision.json")
    direct_summary, direct_rows = load_endpoint(DIRECT, DIRECT_UPDATE)
    dense_summary, dense_rows = load_endpoint(DENSE, DENSE_UPDATE)
    expected = int(
        protocol["fixed_development_evaluation"]["total_rollouts_per_branch"]
    )
    if len(direct_rows) != expected or len(dense_rows) != expected:
        raise RuntimeError(
            f"incomplete endpoint: direct={len(direct_rows)}, dense={len(dense_rows)}"
        )
    direct_success = sum(int(row["reward"]) for row in direct_rows)
    dense_success = sum(int(row["reward"]) for row in dense_rows)
    direct_rate = direct_success / len(direct_rows)
    dense_rate = dense_success / len(dense_rows)
    fisher = fisher_exact(
        [
            [dense_success, len(dense_rows) - dense_success],
            [direct_success, len(direct_rows) - direct_success],
        ],
        alternative="greater",
    )
    paired = paired_interval(
        instance_rates(dense_rows),
        instance_rates(direct_rows),
    )
    criterion = protocol["predeclared_positive_criterion"]
    positive = bool(
        dense_rate
        >= float(criterion["dense_strict_has_endpoint_min_full_pass_rate"])
        and dense_rate > direct_rate
        and float(fisher.pvalue)
        < float(criterion["one_sided_fisher_exact_p_below"])
    )
    odds_ratio = float(fisher.statistic)
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "positive" if positive else "null_under_single_revision",
        "endpoints": {
            "direct_exact_stall_counterfactual": {
                "observed_update": DIRECT_UPDATE,
                "counterfactual_update": DENSE_UPDATE,
                "successes": direct_success,
                "rollouts": len(direct_rows),
                "full_pass_rate": direct_rate,
                "summary": direct_summary,
            },
            "dense_regex_then_strict_has": {
                "update": DENSE_UPDATE,
                "successes": dense_success,
                "rollouts": len(dense_rows),
                "full_pass_rate": dense_rate,
                "summary": dense_summary,
            },
        },
        "effect": {
            "absolute_rate_difference": dense_rate - direct_rate,
            "one_sided_fisher_exact_odds_ratio": (
                odds_ratio if math.isfinite(odds_ratio) else None
            ),
            "one_sided_fisher_exact_p": float(fisher.pvalue),
            "paired_instance_t_interval": paired,
        },
        "decision": {
            "positive_strict_has_bootstrap": positive,
            "criterion": criterion,
            "no_further_automatic_extension_if_null": True,
            "official_test_remains_sealed": True,
            "no_seed_hash_or_configuration_search": True,
        },
    }
    atomic_json(
        EXP_ROOT / "aggregated_results" / "strict_has_revision_summary.json",
        result,
    )
    atomic_json(
        EXP_ROOT / "manifests" / "strict_has_revision_run_complete.json",
        {
            "status": result["status"],
            "summary": "aggregated_results/strict_has_revision_summary.json",
        },
    )
    if positive:
        atomic_json(
            EXP_ROOT / "manifests" / "strict_has_revision_experiment_complete.json",
            {
                "status": "positive",
                "claim_scope": protocol["fixed_development_evaluation"]["claim_scope"],
                "official_test_remains_sealed": True,
            },
        )
    lines = [
        "# Strict HAS bootstrap: single frozen revision",
        "",
        f"- Direct exact-stall counterfactual: {direct_success}/{len(direct_rows)} "
        f"({direct_rate:.6f})",
        f"- Dense REGEX + strict HAS endpoint: {dense_success}/{len(dense_rows)} "
        f"({dense_rate:.6f})",
        f"- Absolute difference: {dense_rate - direct_rate:.6f}",
        f"- One-sided Fisher exact p: {float(fisher.pvalue):.6g}",
        f"- Positive strict-HAS bootstrap: {positive}",
        "",
        "No target partial reward, seed/hash/configuration search, or official test was used.",
        "",
    ]
    (EXP_ROOT / "RESULTS_STRICT_HAS_REVISION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
