from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


AGG = EXP_ROOT / "aggregated_results"
PAPER = EXP_ROOT / "paper_outputs"
FIG = EXP_ROOT / "figures"


def tex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def branch_paths(branch_id: str, budget: int) -> tuple[Path, Path, Path]:
    root = EXP_ROOT / "raw_results" / "branches" / branch_id
    return (
        root / "status.json",
        root / "confirmation" / f"budget_{budget}_summary.json",
        root / "official_test" / f"budget_{budget}_summary.json",
    )


def instance_rates(branch_id: str, budget: int) -> np.ndarray:
    path = (
        EXP_ROOT
        / "raw_results"
        / "branches"
        / branch_id
        / "official_test"
        / f"budget_{budget}.jsonl"
    )
    rows = read_jsonl(path)
    groups: dict[int, list[int]] = {}
    for row in rows:
        groups.setdefault(int(row["instance_index"]), []).append(int(row["reward"]))
    return np.asarray([np.mean(groups[index]) for index in sorted(groups)], dtype=float)


def bootstrap_gain(
    candidate_vectors: dict[int, np.ndarray],
    direct_vectors: dict[int, np.ndarray],
    rng: np.random.Generator,
    samples: int = 10000,
) -> tuple[float, float]:
    size = len(next(iter(candidate_vectors.values())))
    values = np.empty(samples, dtype=float)
    seeds = sorted(candidate_vectors)
    for iteration in range(samples):
        chosen = rng.integers(0, size, size=size)
        values[iteration] = np.mean(
            [
                np.mean(candidate_vectors[seed][chosen] - direct_vectors[seed][chosen])
                for seed in seeds
            ]
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def claim_assessment(frame: pd.DataFrame) -> tuple[bool, str]:
    uncertainty_span = float(frame["solver_uncertainty"].max() - frame["solver_uncertainty"].min())
    nonpositive_high = bool(
        (frame.nlargest(max(1, len(frame) // 2), "solver_uncertainty")["mean_target_gain"] <= 0).any()
    )
    positive = bool((frame["ci_low"] > 0).any())
    stable_positive = bool(
        frame.groupby("candidate_id")["all_seed_gains_positive"].first().any()
    )
    supported = uncertainty_span <= 0.35 and nonpositive_high and positive and stable_positive
    if supported:
        reason = (
            "Selected candidates occupied a comparable uncertainty band while including both "
            "a high-uncertainty non-improving transfer and a bootstrap-positive transfer whose "
            "paired gain was positive in all three seeds."
        )
    else:
        reason = (
            "The prespecified support rule was not fully met. Results are reported without "
            "candidate replacement; the table and confidence intervals identify which condition failed."
        )
    return supported, reason


def make_figures(gains: pd.DataFrame, branch_frame: pd.DataFrame, frozen: dict, budget: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = gains.sort_values("solver_uncertainty", ascending=False).reset_index(drop=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    ax = axes[0]
    for index, row in order.iterrows():
        ax.errorbar(
            row["solver_uncertainty"],
            row["mean_target_gain"],
            yerr=[[row["mean_target_gain"] - row["ci_low"]], [row["ci_high"] - row["mean_target_gain"]]],
            fmt="o",
            color=colors[index],
            capsize=2,
            markersize=4,
        )
        ax.annotate(row["family"], (row["solver_uncertainty"], row["mean_target_gain"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Pre-training Solver uncertainty")
    ax.set_ylabel("Paired PREPEND target gain")
    ax.set_title("A. Uncertainty versus target gain", loc="left", fontweight="bold")
    ax = axes[1]
    x = np.arange(len(order))
    ax.bar(x, order["mean_target_gain"], color=colors, width=0.72, edgecolor="black", linewidth=0.4)
    for index, row in order.iterrows():
        seed_rows = gains[gains["candidate_id"] == row["candidate_id"]]
        for _, seed_row in seed_rows.iterrows():
            for gain in seed_row["seed_gains"]:
                ax.plot(index, gain, "o", color="black", markersize=2.5, alpha=0.75)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [row["family"] for _, row in order.iterrows()], rotation=45, ha="right")
    ax.set_ylabel("Paired PREPEND target gain")
    ax.set_title("B. Comparable uncertainty, different transfer", loc="left", fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"uncertainty_vs_target_gain.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    curves: list[dict[str, Any]] = []
    labels = {"direct": []}
    selected_ids = frozen["selected_candidate_ids"]
    for branch in frozen["branches"]:
        label = "direct" if branch["kind"] == "direct" else branch["candidate_id"]
        for path in sorted(
            (EXP_ROOT / "raw_results" / "branches" / branch["branch_id"] / "development").glob(
                "update_*_summary.json"
            )
        ):
            row = read_json(path)
            curves.append(
                {
                    "label": label,
                    "seed": branch["seed"],
                    "update": row["update"],
                    "pass1": row["full_pass_rate"],
                }
            )
    curve_frame = pd.DataFrame(curves)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    for index, label in enumerate(["direct"] + selected_ids):
        subset = curve_frame[curve_frame["label"] == label]
        grouped = subset.groupby("update")["pass1"]
        mean = grouped.mean()
        low = grouped.min()
        high = grouped.max()
        name = "PREPEND-only" if label == "direct" else label.split("__")[0]
        color = "black" if label == "direct" else plt.get_cmap("tab10")(index - 1)
        ax.plot(mean.index, mean.values, label=name, color=color, linewidth=1.2)
        ax.fill_between(mean.index, low.values, high.values, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Global update")
    ax.set_ylabel("PREPEND development Pass@1")
    ax.set_title("Compute-matched target learning curves")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"target_learning_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    AGG.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frozen = read_json(EXP_ROOT / "manifests" / "frozen_manifest.json")
    budget = int(read_json(EXP_ROOT / "manifests" / "final_common_budget.json")["budget"])
    candidate_table = pd.DataFrame(frozen["complete_candidate_table"])
    candidate_table.to_csv(AGG / "candidate_uncertainty.csv", index=False)
    branch_rows = []
    test_vectors: dict[str, np.ndarray] = {}
    for branch in frozen["branches"]:
        status_path, confirmation_path, test_path = branch_paths(branch["branch_id"], budget)
        status = read_json(status_path)
        confirmation = read_json(confirmation_path)
        test = read_json(test_path)
        test_vectors[branch["branch_id"]] = instance_rates(branch["branch_id"], budget)
        branch_rows.append(
            {
                **branch,
                "completed_updates": status["completed_updates"],
                "training_rollouts": status["training_rollouts"],
                "training_generated_tokens": status["training_generated_tokens"],
                "first_target_success_update": status["first_target_success_update"],
                "confirmation_pass1": confirmation["full_pass_rate"],
                "confirmed_confirmation_pass1": confirmation["confirmed_full_pass_rate"],
                "official_test_pass1": test["full_pass_rate"],
                "official_test_parse_valid_rate": test["parse_valid_rate"],
                "official_test_successes": test["full_pass_count"],
            }
        )
    branch_frame = pd.DataFrame(branch_rows)
    branch_frame.to_csv(AGG / "branch_level_results.csv", index=False)
    branch_frame[
        [
            "branch_id",
            "kind",
            "candidate_id",
            "seed",
            "official_test_pass1",
            "official_test_parse_valid_rate",
            "official_test_successes",
        ]
    ].to_csv(AGG / "test_results.csv", index=False)

    direct = {
        int(row.seed): test_vectors[row.branch_id]
        for row in branch_frame.itertuples()
        if row.kind == "direct"
    }
    seed_rows = []
    gain_rows = []
    rng = np.random.default_rng(20260826)
    selected_map = {row["candidate_id"]: row for row in frozen["selected_candidates"]}
    for candidate_id in frozen["selected_candidate_ids"]:
        candidate_branches = branch_frame[branch_frame["candidate_id"] == candidate_id]
        vectors = {
            int(row.seed): test_vectors[row.branch_id] for row in candidate_branches.itertuples()
        }
        seed_gains = []
        for seed in sorted(vectors):
            candidate_pass = float(vectors[seed].mean())
            direct_pass = float(direct[seed].mean())
            gain = candidate_pass - direct_pass
            seed_gains.append(gain)
            seed_rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": selected_map[candidate_id]["family"],
                    "seed": seed,
                    "candidate_pass1": candidate_pass,
                    "direct_pass1": direct_pass,
                    "paired_target_gain": gain,
                }
            )
        low, high = bootstrap_gain(vectors, direct, rng)
        gain_rows.append(
            {
                "candidate_id": candidate_id,
                "family": selected_map[candidate_id]["family"],
                "solver_uncertainty": selected_map[candidate_id]["solver_uncertainty"],
                "seed_gains": seed_gains,
                "mean_target_gain": float(np.mean(seed_gains)),
                "ci_low": low,
                "ci_high": high,
                "all_seed_gains_positive": all(value > 0 for value in seed_gains),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(AGG / "seed_level_target_gain.csv", index=False)
    gains = pd.DataFrame(gain_rows)
    gains["uncertainty_rank"] = gains["solver_uncertainty"].rank(ascending=False, method="min")
    gains["target_gain_rank"] = gains["mean_target_gain"].rank(ascending=False, method="min")
    rho, pvalue = spearmanr(gains["solver_uncertainty"], gains["mean_target_gain"])
    supported, assessment = claim_assessment(gains)
    total_updates = int(branch_frame["completed_updates"].sum())
    total_rollouts = int(branch_frame["training_rollouts"].sum())
    total_tokens = int(branch_frame["training_generated_tokens"].sum())
    statistical = {
        "common_budget": budget,
        "branches": len(branch_frame),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
        "bootstrap_samples": 10000,
        "bootstrap_unit": "official PREPEND test instance with seed pairing preserved",
        "claim_supported": supported,
        "claim_assessment": assessment,
        "total_training_updates": total_updates,
        "total_training_rollouts": total_rollouts,
        "total_training_generated_tokens": total_tokens,
        "candidates": json.loads(gains.to_json(orient="records")),
    }
    atomic_json(AGG / "statistical_summary.json", statistical)
    make_figures(gains, branch_frame, frozen, budget)

    table_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Candidate family & Uncertainty & Test Pass@1 & Paired gain & 95\% CI \\",
        r"\midrule",
    ]
    for row in gains.sort_values("solver_uncertainty", ascending=False).itertuples():
        candidate_mean = float(
            seed_frame[seed_frame["candidate_id"] == row.candidate_id]["candidate_pass1"].mean()
        )
        table_lines.append(
            f"{tex(row.family)} & {row.solver_uncertainty:.3f} & {candidate_mean:.3f} & "
            f"{row.mean_target_gain:+.3f} & [{row.ci_low:+.3f}, {row.ci_high:+.3f}] \\\\"
        )
    table_lines += [r"\midrule", f"Spearman $\rho$ & \multicolumn{{4}}{{r}}{{{rho:.3f} ($p={pvalue:.3g}$)}} \\\\", r"\bottomrule", r"\end{tabular}"]
    (PAPER / "main_result_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    caption = (
        "Solver uncertainty versus compute-matched PREPEND transfer. Panel A compares "
        "pre-training verified uncertainty with official-test paired target gain; error bars "
        "are 95\% paired instance-bootstrap intervals over three seeds. Panel B retains all "
        "eight frozen candidates and shows seed-level paired gains. All branches use the same "
        f"{budget}-update budget."
    )
    (PAPER / "main_figure_caption.tex").write_text(caption + "\n", encoding="utf-8")
    strongest = gains.sort_values("mean_target_gain", ascending=False).iloc[0]
    weakest = gains.sort_values("mean_target_gain", ascending=True).iloc[0]
    paragraph = (
        f"We compared eight target-blind Manufactoria distributions selected by Solver "
        f"uncertainty while fixing the Qwen3.5-4B initialization, binary verifier reward, "
        f"LoRA--GRPO configuration, and a common budget of {budget} updates. "
        f"On the sealed PREPEND test set, paired target gains ranged from "
        f"{weakest.mean_target_gain:+.3f} ({tex(weakest.family)}) to "
        f"{strongest.mean_target_gain:+.3f} ({tex(strongest.family)}); the rank correlation "
        f"between uncertainty and gain was $\\rho={rho:.3f}$ ($p={pvalue:.3g}$). "
        + (
            "The prespecified evidence rule was satisfied, showing that frontier proximity alone did not determine progress toward the fixed target."
            if supported
            else "The prespecified evidence rule was not satisfied in this setting, so these results do not support the proposed phenomenon."
        )
    )
    (PAPER / "result_paragraph.tex").write_text(paragraph + "\n", encoding="utf-8")
    protocol = (
        "We evaluated the official post-trained Qwen/Qwen3.5-4B text-only policy with "
        "thinking disabled. Candidate distributions were selected without PREPEND feedback "
        "from 13 non-target generator families using 32 instances and eight verifier-scored "
        "rollouts per configuration. Each of 27 LoRA--GRPO branches used eight prompts and "
        "eight rollouts per update, strict full-pass rewards, and one B200; two independent "
        "branches ran concurrently. Candidate branches used 60 candidate updates followed by "
        f"PREPEND, and all branches stopped at the common budget of {budget} updates."
    )
    (PAPER / "experiment_protocol.tex").write_text(protocol + "\n", encoding="utf-8")

    candidate_md = candidate_table[
        ["candidate_id", "family", "full_pass_rate", "parse_valid_rate", "solver_uncertainty", "selected"]
    ].to_markdown(index=False)
    gain_md = gains[
        ["candidate_id", "solver_uncertainty", "mean_target_gain", "ci_low", "ci_high"]
    ].to_markdown(index=False)
    branch_md = branch_frame[
        ["branch_id", "completed_updates", "official_test_pass1", "first_target_success_update"]
    ].to_markdown(index=False)
    results = f"""# Uncertainty-to-target-gain experiment results

## Outcome

* Final zero-success target stratum: {frozen['target_stratum']['generator_parameters']}
* Common budget: {budget} updates for all 27 branches
* Total training updates: {total_updates}
* Total training rollouts: {total_rollouts}
* Total generated training tokens: {total_tokens}
* Spearman uncertainty/gain correlation: rho={rho:.4f}, p={pvalue:.4g}
* Core phenomenon supported: **{str(supported).lower()}**
* Assessment: {assessment}

## Complete target-blind candidate pool

{candidate_md}

## Selected candidates and paired official-test target gain

{gain_md}

## Branch completion and official test

{branch_md}

## Compute and implementation

GPU 0 and GPU 1 each ran one complete independent branch at a time; no tensor
parallelism or DDP was used. ROLL was replaced by Transformers/PEFT native
generation because its stable Qwen3.5 examples conflict with the fixed
one-branch-per-GPU layout and the available vLLM wheel required an incompatible
PyTorch/CUDA ABI. The backbone, prompts, full verifier, binary rewards, budgets,
selection, sampling, and evaluation protocol were unchanged.
"""
    (EXP_ROOT / "RESULTS.md").write_text(results, encoding="utf-8")
    report = f"""# Claim evidence report

**Supported:** {supported}

{assessment}

Uncertainty-to-gain Spearman rho was {rho:.4f} (p={pvalue:.4g}). The full
candidate table, all zero/negative transfers, three paired seeds, hidden
confirmation checks, and sealed official test results are retained.
"""
    (PAPER / "claim_evidence_report.md").write_text(report, encoding="utf-8")
    print(statistical, flush=True)


if __name__ == "__main__":
    main()
