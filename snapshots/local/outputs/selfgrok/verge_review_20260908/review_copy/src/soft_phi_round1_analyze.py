from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import t, wilcoxon

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def partial(row: dict[str, Any]) -> float:
    total = int(row["total_cases"])
    return int(row["passed_cases"]) / total if total > 0 else 0.0


def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    if len(rows) != 800 or len(grouped) != 50 or set(map(len, grouped.values())) != {16}:
        raise RuntimeError(
            f"invalid fixed evaluation: rows={len(rows)}, instances={len(grouped)}"
        )
    rates = {
        key: mean(partial(row) for row in items) for key, items in grouped.items()
    }
    return (
        {
            "rollouts": len(rows),
            "instances": len(grouped),
            "soft_phi": mean(rates.values()),
            "strict_full_pass_count": sum(int(row["reward"]) for row in rows),
            "strict_full_pass_rate": sum(int(row["reward"]) for row in rows) / len(rows),
            "parse_valid_rate": sum(bool(row["parse_valid"]) for row in rows) / len(rows),
            "near_strict_ge90pct_count": sum(0.9 <= partial(row) < 1.0 for row in rows),
        },
        rates,
    )


def paired_effect(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if len(ids) != 50:
        raise RuntimeError(f"expected 50 paired instances, found {len(ids)}")
    differences = [left[key] - right[key] for key in ids]
    estimate = mean(differences)
    standard_error = stdev(differences) / math.sqrt(len(differences))
    radius = float(t.ppf(0.975, len(differences) - 1)) * standard_error
    pvalue = (
        1.0
        if all(value == 0.0 for value in differences)
        else float(
            wilcoxon(
                differences,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    )
    return {
        "gamma_phi": estimate,
        "ci95_student_t": [estimate - radius, estimate + radius],
        "wilcoxon_one_sided_p": pvalue,
        "positive_instances": sum(value > 0 for value in differences),
        "negative_instances": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "median_instance_difference": median(differences),
    }


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "soft_phi_rollout_round1.json")
    summaries: dict[str, Any] = {}
    rates: dict[str, dict[str, float]] = {}
    for branch in protocol["branches"]:
        branch_id = str(branch["branch_id"])
        status_path = RAW_ROOT / branch_id / "status.json"
        if not status_path.exists() or int(read_json(status_path)["completed_updates"]) != 100:
            raise RuntimeError(f"branch is incomplete: {branch_id}")
        path = RAW_ROOT / branch_id / "development" / "update_0100.jsonl"
        summaries[branch_id], rates[branch_id] = summarize(read_jsonl(path))

    direct_id = "softphi_r1__direct"
    comparisons = {}
    eligible = []
    for branch in protocol["branches"]:
        if branch["kind"] != "candidate":
            continue
        branch_id = str(branch["branch_id"])
        effect = paired_effect(rates[branch_id], rates[direct_id])
        effect["candidate_id"] = branch["candidate_id"]
        effect["eligible"] = bool(
            effect["gamma_phi"] > 0
            and effect["ci95_student_t"][0] > 0
            and effect["wilcoxon_one_sided_p"] < 0.05
        )
        comparisons[branch_id] = effect
        if effect["eligible"]:
            eligible.append((effect["gamma_phi"], str(branch["candidate_id"]), branch_id))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    selected_branch = eligible[0][2] if eligible else direct_id
    selected_candidate = comparisons[selected_branch]["candidate_id"] if eligible else None
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "candidate_selected" if eligible else "direct_retained",
        "branch_summaries": summaries,
        "candidate_vs_direct": comparisons,
        "decision": {
            "selected_branch": selected_branch,
            "selected_candidate": selected_candidate,
            "selection_uses_soft_phi": True,
            "strict_full_pass_remains_endpoint": True,
            "confirmation_opened": False,
            "official_test_opened": False,
            "challenger_learning_claimed": False,
        },
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "soft_phi_round1.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / "challenger_feedback_round1.json",
        {
            "round": 1,
            "feedback": [
                {
                    "branch_id": key,
                    "candidate_id": value["candidate_id"],
                    "gamma_phi": value["gamma_phi"],
                    "eligible": value["eligible"],
                }
                for key, value in comparisons.items()
            ],
            "selected_branch": selected_branch,
            "selected_candidate": selected_candidate,
            "next_round_not_automatically_launched": True,
        },
    )
    atomic_json(
        EXP_ROOT / "manifests" / "soft_phi_round1_complete.json",
        {
            "status": result["status"],
            "selected_branch": selected_branch,
            "summary": "aggregated_results/soft_phi_round1.json",
        },
    )
    lines = [
        "# Soft-Phi self-grok rollout: round 1",
        "",
        f"- Direct soft Phi: {summaries[direct_id]['soft_phi']:.6f}",
    ]
    for branch_id, effect in comparisons.items():
        summary = summaries[branch_id]
        lines.append(
            f"- {effect['candidate_id']}: soft Phi={summary['soft_phi']:.6f}, "
            f"Gamma={effect['gamma_phi']:.6f}, "
            f"CI=[{effect['ci95_student_t'][0]:.6f}, "
            f"{effect['ci95_student_t'][1]:.6f}], "
            f"p={effect['wilcoxon_one_sided_p']:.6g}, "
            f"strict={summary['strict_full_pass_count']}/800, "
            f"eligible={effect['eligible']}"
        )
    lines.extend(
        [
            f"- Selected branch: {selected_branch}",
            f"- Selected candidate: {selected_candidate}",
            "- Confirmation and official test remain sealed.",
            "",
        ]
    )
    (EXP_ROOT / "RESULTS_SOFT_PHI_ROUND1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
