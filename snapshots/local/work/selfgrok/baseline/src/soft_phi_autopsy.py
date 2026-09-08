from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "high_power_strict_assay_v1"
INPUTS = {
    "direct": RAW_ROOT / "direct_u0100.jsonl",
    "curriculum": RAW_ROOT / "curriculum_u0200.jsonl",
}


def score(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    if total <= 0:
        raise RuntimeError("non-positive total_cases")
    return int(row["passed_cases"]) / total


def validate(rows: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    expected_total = int(expected["expected_rollouts_per_arm"])
    expected_instances = int(expected["expected_instances"])
    expected_per_instance = int(expected["expected_rollouts_per_instance"])
    if len(rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} rows, found {len(rows)}")
    counts = Counter(str(row["instance_id"]) for row in rows)
    if len(counts) != expected_instances:
        raise RuntimeError(
            f"expected {expected_instances} instances, found {len(counts)}"
        )
    if set(counts.values()) != {expected_per_instance}:
        raise RuntimeError(f"unexpected per-instance counts: {sorted(counts.values())}")
    keys = [(str(row["instance_id"]), int(row["rollout_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate instance/rollout keys")


def failure_bins(rows: list[dict[str, Any]]) -> dict[str, int]:
    bins = Counter()
    for row in rows:
        value = score(row)
        if not bool(row["parse_valid"]):
            bins["parse_invalid"] += 1
        elif value == 0.0:
            bins["parsed_zero_cases"] += 1
        elif value <= 0.25:
            bins["parsed_gt0_le25pct"] += 1
        elif value <= 0.50:
            bins["parsed_gt25_le50pct"] += 1
        elif value <= 0.75:
            bins["parsed_gt50_le75pct"] += 1
        elif value < 1.0:
            bins["parsed_gt75_lt100pct"] += 1
        else:
            bins["strict_full_pass"] += 1
    order = [
        "parse_invalid",
        "parsed_zero_cases",
        "parsed_gt0_le25pct",
        "parsed_gt25_le50pct",
        "parsed_gt50_le75pct",
        "parsed_gt75_lt100pct",
        "strict_full_pass",
    ]
    return {key: int(bins[key]) for key in order}


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if bool(row["parse_valid"])]
    values = [score(row) for row in rows]
    parsed_values = [score(row) for row in parsed]
    strict = sum(int(row["reward"]) for row in rows)
    parse_errors = Counter(
        str(row["verifier_message"])
        for row in rows
        if not bool(row["parse_valid"])
    )
    return {
        "rollouts": len(rows),
        "strict_full_pass_count": strict,
        "strict_full_pass_rate": strict / len(rows),
        "parse_valid_count": len(parsed),
        "parse_valid_rate": len(parsed) / len(rows),
        "soft_phi_mean": mean(values),
        "soft_phi_median": median(values),
        "conditional_semantic_mean": mean(parsed_values) if parsed_values else None,
        "conditional_semantic_median": median(parsed_values) if parsed_values else None,
        "near_strict_ge90pct_count": sum(0.9 <= value < 1.0 for value in values),
        "failure_bins": failure_bins(rows),
        "top_parse_errors": [
            {"message": message, "count": count}
            for message, count in parse_errors.most_common(12)
        ],
    }


def per_instance(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    result = {}
    for instance_id, items in grouped.items():
        parsed_values = [score(row) for row in items if bool(row["parse_valid"])]
        values = [score(row) for row in items]
        result[instance_id] = {
            "parse_valid_count": len(parsed_values),
            "parse_rate": len(parsed_values) / len(items),
            "soft_phi": mean(values),
            "conditional_semantic": (
                mean(parsed_values) if parsed_values else None
            ),
            "best_partial_score": max(values),
            "near_strict_ge90pct_count": sum(0.9 <= value < 1.0 for value in values),
        }
    return result


def paired_inference(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    metric: str,
    ids: list[str],
) -> dict[str, Any]:
    differences = [float(left[key][metric]) - float(right[key][metric]) for key in ids]
    estimate = mean(differences)
    if len(differences) > 1:
        standard_error = stdev(differences) / math.sqrt(len(differences))
        radius = float(t.ppf(0.975, len(differences) - 1)) * standard_error
        ci = [estimate - radius, estimate + radius]
    else:
        ci = [estimate, estimate]
    if all(value == 0.0 for value in differences):
        pvalue = 1.0
    else:
        pvalue = float(
            wilcoxon(
                differences,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    return {
        "estimate": estimate,
        "ci95_student_t": ci,
        "wilcoxon_one_sided_p": pvalue,
        "instances": len(ids),
        "positive_instances": sum(value > 0.0 for value in differences),
        "negative_instances": sum(value < 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "median_instance_difference": median(differences),
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "soft_phi_autopsy.json")
    rows = {name: read_jsonl(path) for name, path in INPUTS.items()}
    for items in rows.values():
        validate(items, protocol["inputs"])
    direct_keys = {
        (str(row["instance_id"]), int(row["rollout_index"])) for row in rows["direct"]
    }
    curriculum_keys = {
        (str(row["instance_id"]), int(row["rollout_index"]))
        for row in rows["curriculum"]
    }
    if direct_keys != curriculum_keys:
        raise RuntimeError("arms do not contain identical instance/rollout keys")

    arm = {name: summarize_arm(items) for name, items in rows.items()}
    instance = {name: per_instance(items) for name, items in rows.items()}
    ids = sorted(instance["direct"])
    soft_effect = paired_inference(
        instance["curriculum"], instance["direct"], "soft_phi", ids
    )
    parse_effect = paired_inference(
        instance["curriculum"], instance["direct"], "parse_rate", ids
    )
    minimum_parsed = int(
        protocol["predeclared_admissibility"][
            "conditional_min_parse_valid_rollouts_per_arm_per_instance"
        ]
    )
    conditional_ids = [
        key
        for key in ids
        if instance["direct"][key]["parse_valid_count"] >= minimum_parsed
        and instance["curriculum"][key]["parse_valid_count"] >= minimum_parsed
    ]
    conditional_effect = paired_inference(
        instance["curriculum"],
        instance["direct"],
        "conditional_semantic",
        conditional_ids,
    )
    criteria = protocol["predeclared_admissibility"]
    checks = {
        "both_strict_zero": (
            arm["direct"]["strict_full_pass_count"] == 0
            and arm["curriculum"]["strict_full_pass_count"] == 0
        ),
        "curriculum_parse_rate": (
            arm["curriculum"]["parse_valid_rate"]
            >= float(criteria["curriculum_parse_rate_at_least"])
        ),
        "soft_phi_effect_size": (
            soft_effect["estimate"] >= float(criteria["soft_phi_gain_at_least"])
        ),
        "soft_phi_ci": (
            soft_effect["ci95_student_t"][0]
            > float(criteria["soft_phi_ci95_low_above"])
        ),
        "soft_phi_wilcoxon": (
            soft_effect["wilcoxon_one_sided_p"]
            < float(criteria["soft_phi_wilcoxon_p_below"])
        ),
        "soft_phi_positive_instances": (
            soft_effect["positive_instances"]
            >= int(criteria["soft_phi_positive_instances_at_least"])
        ),
        "conditional_common_instances": (
            conditional_effect["instances"]
            >= int(criteria["conditional_common_instances_at_least"])
        ),
        "conditional_effect_size": (
            conditional_effect["estimate"]
            >= float(criteria["conditional_semantic_gain_at_least"])
        ),
        "conditional_ci": (
            conditional_effect["ci95_student_t"][0]
            > float(criteria["conditional_semantic_ci95_low_above"])
        ),
        "conditional_wilcoxon": (
            conditional_effect["wilcoxon_one_sided_p"]
            < float(criteria["conditional_semantic_wilcoxon_p_below"])
        ),
        "conditional_direction_count": (
            conditional_effect["positive_instances"]
            > conditional_effect["negative_instances"]
        ),
    }
    admissible = all(checks.values())
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "admissible_soft_phi" if admissible else "soft_phi_not_admissible",
        "arm_summaries": arm,
        "paired_effects": {
            "curriculum_minus_direct_parse_rate": parse_effect,
            "curriculum_minus_direct_soft_phi": soft_effect,
            "curriculum_minus_direct_conditional_semantics": conditional_effect,
        },
        "conditional_instance_filter": {
            "minimum_parse_valid_rollouts_per_arm": minimum_parsed,
            "included_instances": conditional_ids,
        },
        "decision": {
            "soft_phi_admissible_as_online_search_instrument": admissible,
            "checks": checks,
            "strict_full_pass_remains_only_endpoint_metric": True,
            "launch_main_loop": False,
            "official_test_remains_sealed": True,
        },
    }
    atomic_json(
        EXP_ROOT / "aggregated_results" / "soft_phi_autopsy.json",
        result,
    )
    atomic_json(
        EXP_ROOT / "manifests" / "soft_phi_autopsy_complete.json",
        {
            "status": result["status"],
            "summary": "aggregated_results/soft_phi_autopsy.json",
        },
    )
    semantic = conditional_effect
    lines = [
        "# Soft-Phi autopsy",
        "",
        f"- Strict full pass: direct {arm['direct']['strict_full_pass_count']}/6400; "
        f"curriculum {arm['curriculum']['strict_full_pass_count']}/6400",
        f"- Parse valid: direct {arm['direct']['parse_valid_rate']:.6f}; "
        f"curriculum {arm['curriculum']['parse_valid_rate']:.6f}",
        f"- Unconditional soft Phi: direct {arm['direct']['soft_phi_mean']:.6f}; "
        f"curriculum {arm['curriculum']['soft_phi_mean']:.6f}",
        f"- Paired soft-Phi gain: {soft_effect['estimate']:.6f} "
        f"[{soft_effect['ci95_student_t'][0]:.6f}, "
        f"{soft_effect['ci95_student_t'][1]:.6f}], "
        f"Wilcoxon p={soft_effect['wilcoxon_one_sided_p']:.6g}",
        f"- Conditional semantic gain: {semantic['estimate']:.6f} "
        f"[{semantic['ci95_student_t'][0]:.6f}, "
        f"{semantic['ci95_student_t'][1]:.6f}], "
        f"Wilcoxon p={semantic['wilcoxon_one_sided_p']:.6g}, "
        f"instances={semantic['instances']}",
        f"- Conditional directions: +{semantic['positive_instances']} "
        f"-{semantic['negative_instances']} ties={semantic['ties']}",
        f"- Soft Phi admissible: {admissible}",
        "- Strict full pass remains the only endpoint metric; official test remains sealed.",
        "",
    ]
    (EXP_ROOT / "RESULTS_SOFT_PHI_AUTOPSY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
