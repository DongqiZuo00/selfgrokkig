from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EXP_ROOT, read_json


def optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_message_figure(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def discovery_figure(rows: list[dict[str, str]]) -> None:
    path = EXP_ROOT / "figures" / "discovery_uncertainty_vs_gain.pdf"
    if not rows:
        save_message_figure(path, "Discovery uncertainty vs. gain", "Discovery has not completed.")
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    for row in rows:
        axis.scatter(
            float(row["solver_uncertainty"]),
            float(row["target_gain"]),
            s=55,
            label=row["candidate_id"],
        )
        axis.annotate(
            row["candidate_id"],
            (float(row["solver_uncertainty"]), float(row["target_gain"])),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Base solver uncertainty U = 4p(1-p)")
    axis.set_ylabel("Target gain over seed-matched direct")
    axis.set_title("Pre-frozen matched candidates: discovery seed 42")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def confirmed_figure(rows: list[dict[str, str]]) -> None:
    path = EXP_ROOT / "figures" / "confirmed_matched_pair.pdf"
    if not rows:
        save_message_figure(path, "Confirmed matched pair", "Independent confirmation was not run.")
        return
    seeds = [int(row["seed"]) for row in rows]
    a = [float(row["candidate_a_target_gain"]) for row in rows]
    b = [float(row["candidate_b_target_gain"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 5))
    for index, seed in enumerate(seeds):
        axis.plot([0, 1], [a[index], b[index]], marker="o", alpha=0.8, label=f"seed {seed}")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks([0, 1], ["Candidate A", "Candidate B"])
    axis.set_ylabel("Target gain over seed-matched direct")
    axis.set_title("Independent confirmation across three new seeds")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def learning_curve_figure() -> None:
    path = EXP_ROOT / "figures" / "confirmed_target_learning_curves.pdf"
    branch_manifest = optional_json(EXP_ROOT / "manifests" / "fast_branches.json")
    pair = optional_json(EXP_ROOT / "manifests" / "confirmed_pair_manifest.json")
    base = optional_json(
        EXP_ROOT
        / "raw_results"
        / "protocol_v2"
        / "base_shared"
        / "development"
        / "update_0000_summary.json"
    )
    if not branch_manifest or not pair or not base:
        save_message_figure(path, "Target learning curves", "Confirmation learning curves were not run.")
        return
    groups: dict[str, list[tuple[int, float]]] = {"A": [], "B": [], "Direct": []}
    for branch in branch_manifest["branches"]:
        if branch["stage"] != "confirmation":
            continue
        if branch["kind"] == "direct":
            label = "Direct"
        elif branch["candidate_id"] == pair["candidate_a"]:
            label = "A"
        elif branch["candidate_id"] == pair["candidate_b"]:
            label = "B"
        else:
            continue
        monitor = optional_json(
            EXP_ROOT
            / "raw_results"
            / "protocol_v2"
            / "branches"
            / branch["branch_id"]
            / "development_monitor"
            / "update_0060_summary.json"
        )
        endpoint = optional_json(
            EXP_ROOT
            / "raw_results"
            / "protocol_v2"
            / "branches"
            / branch["branch_id"]
            / "confirmation"
            / "update_0100_summary.json"
        )
        if monitor and endpoint:
            groups[label].append((branch["seed"], float(monitor["full_pass_rate"])))
            groups[label].append((branch["seed"] + 1_000_000, float(endpoint["full_pass_rate"])))
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = {"A": "#1f77b4", "B": "#d62728", "Direct": "#555555"}
    base_rate = float(base["full_pass_rate"])
    for label, values in groups.items():
        seeds = sorted({seed % 1_000_000 for seed, _ in values})
        for seed in seeds:
            at60 = next((value for key, value in values if key == seed), None)
            at100 = next((value for key, value in values if key == seed + 1_000_000), None)
            if at60 is not None and at100 is not None:
                axis.plot(
                    [0, 60, 100],
                    [base_rate, at60, at100],
                    color=colors[label],
                    alpha=0.35,
                )
        means = []
        for update in (60, 100):
            offset = 0 if update == 60 else 1_000_000
            selected = [value for key, value in values if key >= offset and key < offset + 1_000_000]
            means.append(sum(selected) / len(selected) if selected else 0)
        axis.plot([0, 60, 100], [base_rate, *means], color=colors[label], marker="o", label=label)
    axis.set_xlabel("Update")
    axis.set_ylabel("Verified full-pass rate")
    axis.set_title("Sparse target monitoring at frozen checkpoints")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def tex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_latex(
    discovery: list[dict[str, str]],
    seeds: list[dict[str, str]],
    statistical: dict[str, Any] | None,
) -> None:
    table = EXP_ROOT / "paper_outputs" / "main_result_table.tex"
    with table.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrr}\n")
        handle.write("\\toprule\nCandidate / seed & Uncertainty & Target gain & A--B difference \\\\\n")
        handle.write("\\midrule\n")
        if seeds:
            for row in seeds:
                handle.write(
                    f"{tex_escape(row['seed'])} & -- & {float(row['candidate_a_target_gain']):.4f} / "
                    f"{float(row['candidate_b_target_gain']):.4f} & {float(row['a_minus_b']):.4f} \\\\\n"
                )
        elif discovery:
            for row in discovery:
                handle.write(
                    f"{tex_escape(row['candidate_id'])} & {float(row['solver_uncertainty']):.4f} & "
                    f"{float(row['target_gain']):.4f} & -- \\\\\n"
                )
        else:
            handle.write("Not completed & -- & -- & -- \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
    caption = EXP_ROOT / "paper_outputs" / "main_figure_caption.tex"
    caption.write_text(
        "Target gains for two candidates paired using only base-solver success and "
        "uncertainty. Zero and negative transfer outcomes are retained.\n",
        encoding="utf-8",
    )
    supported = bool(statistical and statistical.get("confirmation_supported"))
    if supported:
        ci = statistical["hierarchical_paired_bootstrap_95_ci"]
        paragraph = (
            "The pre-frozen matched pair reproduced the discovery direction across all "
            f"three new seeds (hierarchical paired-bootstrap 95\\% CI "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]). This supports the scoped claim that tasks "
            "with matched solver uncertainty can yield reliably different target gains."
        )
    elif statistical:
        paragraph = (
            "The fast protocol did not independently confirm the discovery claim. "
            f"The recorded outcome was: {statistical.get('reason', 'confirmation criteria were not met')}. "
            "No unsupported general conclusion about solver uncertainty is drawn."
        )
    else:
        paragraph = "The experiment is still running; no result claim is made."
    (EXP_ROOT / "paper_outputs" / "result_paragraph.tex").write_text(
        paragraph + "\n", encoding="utf-8"
    )
    protocol = (
        "Candidates were screened without target outcomes using 64 initial verified "
        "rollouts, with eight finalists expanded to 256 rollouts and paired by "
        "$|\\Delta p|\\leq0.05$ and $|\\Delta U|\\leq0.05$, where "
        "$U=4p(1-p)$. Discovery used seed 42 and a fixed 100-update endpoint; "
        "confirmation used seeds 123, 2026, and 31415. Rewards were binary full-pass only."
    )
    (EXP_ROOT / "paper_outputs" / "experiment_protocol.tex").write_text(
        protocol + "\n", encoding="utf-8"
    )


def write_results(
    throughput: dict[str, Any] | None,
    discovery: list[dict[str, str]],
    seeds: list[dict[str, str]],
    statistical: dict[str, Any] | None,
) -> None:
    branch_manifest = optional_json(EXP_ROOT / "manifests" / "fast_branches.json") or {}
    complete = optional_json(EXP_ROOT / "manifests" / "fast_experiment_complete.json")
    candidate_manifest = optional_json(EXP_ROOT / "manifests" / "discovery_candidates.json")
    pair = optional_json(EXP_ROOT / "manifests" / "confirmed_pair_manifest.json")
    branches = branch_manifest.get("branches", [])
    statuses = []
    for branch in branches:
        status_path = (
            EXP_ROOT
            / "raw_results"
            / "protocol_v2"
            / "branches"
            / branch["branch_id"]
            / "status.json"
        )
        statuses.append(
            f"- {branch['branch_id']}: "
            + ("complete" if status_path.exists() else "not started/in progress")
        )
    lines = [
        "# RESULTS_FAST",
        "",
        "## Timing and throughput",
        "",
        "- Original ETA: 10–15 days for the superseded 27-branch protocol.",
    ]
    if throughput:
        base = throughput["baseline"]
        optimized = throughput["optimized"]
        lines.extend(
            [
                f"- Before: {base['seconds_per_rollout']:.4f} s/rollout; "
                f"{base['output_tokens_per_second']:.1f} output tokens/s.",
                f"- After: {optimized['seconds_per_rollout']:.4f} s/rollout; "
                f"{optimized['output_tokens_per_second']:.1f} output tokens/s.",
                f"- Completion mean / P95: {optimized['mean_completion_tokens']:.1f} / "
                f"{optimized['p95_completion_tokens']} tokens.",
                f"- Thinking truly disabled: {throughput['thinking_disabled_argument']['observed_think_tags_optimized'] == 0}.",
                f"- Active concurrency: {optimized['active_concurrency_submitted']}; "
                f"mean GPU utilization: {optimized['mean_gpu_utilization']}.",
                "- Revised ETA is recalibrated from the first completed training update; "
                "the fixed protocol never extends beyond update 100.",
            ]
        )
    lines.extend(
        [
            "",
            "## Protocol state",
            "",
            "- Legacy audit retained: 272 rollouts from job 40336174.",
            "- Audit reuse: data split and verifier retained; outputs rerun because the "
            "completion cap/stopping protocol changed from 4096-token unconstrained generation to 2048 with a legal program terminator.",
            f"- Frozen eight candidates: {json.dumps([x['candidate_id'] for x in candidate_manifest['finalists']] if candidate_manifest else [])}.",
            f"- Frozen discovery pairs: {json.dumps(candidate_manifest['discovery_batches'] if candidate_manifest else [])}.",
            f"- Second discovery batch triggered: {bool(discovery and max(int(row['batch']) for row in discovery) == 2)}.",
            f"- Confirmed pair: {json.dumps(pair) if pair else 'none'}.",
            "",
            "## Branch status",
            "",
            *statuses,
            "",
            "## Outcome",
            "",
            f"- Stage 1 complete: {bool(complete and complete.get('stage1_complete'))}.",
            f"- Stage 2 complete: {bool(complete and complete.get('stage2_complete'))}.",
            f"- Official test complete: {bool(complete and complete.get('official_test_complete'))}.",
            f"- Core phenomenon: {complete.get('outcome') if complete else 'inconclusive (running)'}.",
            "- Protocol deviations: the legacy 4096-token audit was not mixed with protocol-v2 "
            "uncertainty; vLLM uses two independent single-GPU servers; all zero and negative gains remain visible.",
        ]
    )
    (EXP_ROOT / "RESULTS_FAST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    throughput = optional_json(EXP_ROOT / "aggregated_results" / "throughput_report.json")
    discovery = csv_rows(EXP_ROOT / "aggregated_results" / "discovery_branch_results.csv")
    seeds = csv_rows(EXP_ROOT / "aggregated_results" / "seed_level_target_gain.csv")
    statistical = optional_json(EXP_ROOT / "aggregated_results" / "statistical_summary.json")
    discovery_figure(discovery)
    confirmed_figure(seeds)
    learning_curve_figure()
    write_latex(discovery, seeds, statistical)
    write_results(throughput, discovery, seeds, statistical)


if __name__ == "__main__":
    main()
