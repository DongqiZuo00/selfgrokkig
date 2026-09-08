"""Deterministic scientific result exports; no sampling or selection changes.

CSV files are flat analysis intermediates for the figure and manuscript table.
All quantities come from the frozen round and produced endpoint artifacts.
"""
import csv
import json
from pathlib import Path


def result_rows(frozen, branches, decision):
    credit_rung = decision["reward_rung_zero_based"]
    selection_rung = decision["selection_rung_zero_based"]
    evaluated = {r["index"]: r for r in decision["evaluated_candidates"]}
    output = []
    endpoints = [("start", frozen["initial"], None)] + [
        ("direct" if i == 0 else f"candidate_{i}", branches[i], i) for i in sorted(branches)]
    for name, endpoint, index in endpoints:
        target, scope = endpoint["target"], endpoint["scope"]
        training = endpoint.get("training", {})
        item = evaluated.get(index, {})
        output.append({
            "branch": name,
            "source": "round_frozen.json:initial" if index is None else f"branches/{index}/complete.json",
            "checkpoint": endpoint["checkpoint"],
            "loss_tokens": training.get("train_tokens", 0),
            "generated_training_tokens": training.get("generated_tokens", 0),
            "target_loss_tokens": training.get("target_loss_tokens", 0),
            "nonzero_advantage_tokens": training.get("nonzero_advantage_tokens", 0),
            "update_units": training.get("update", 0),
            "optimizer_steps": training.get("optimizer_steps", 0),
            "training_target_successes": training.get("target_successes", 0),
            "endpoint_draws": target["rollouts"],
            "endpoint_full_successes": target["counts"][-1],
            "endpoint_full_rate": target["rates"][-1],
            "credit_condition": target["condition_names"][credit_rung],
            "credit_condition_count": target["counts"][credit_rung],
            "credit_condition_rate": target["rates"][credit_rung],
            "selection_condition": target["condition_names"][selection_rung],
            "selection_condition_count": target["counts"][selection_rung],
            "selection_condition_rate": target["rates"][selection_rung],
            "raw_observed_credit_gain": item.get("observed_endpoint_reward"),
            "credit_gain": item.get("reward"),
            "raw_observed_selection_gain": item.get("observed_endpoint_gain"),
            "selection_gain": item.get("selection_gain"),
            "selection_ci95_low": item.get("paired_95_interval", [None, None])[0],
            "selection_ci95_high": item.get("paired_95_interval", [None, None])[1],
            "identity_zero_update": item.get("identical_policy_by_zero_update_provenance"),
            "scope_draws": scope["rollouts"],
            "scope_full_successes": scope["full_pass_count"],
            "scope_full_rate": scope["full_pass_rate"],
            "scope_mean_case_fraction": scope["mean_case_fraction"],
            "scope_safe": item.get("scope_safe"),
            "eligible": item.get("eligible"),
            "retained": name == decision["selected"]["name"],
        })
    return output


def condition_rows(frozen, branches):
    output = []
    endpoints = [("start", frozen["initial"])] + [
        ("direct" if i == 0 else f"candidate_{i}", branches[i]) for i in sorted(branches)]
    for branch, endpoint in endpoints:
        target = endpoint["target"]
        for rung, (name, count, rate) in enumerate(zip(
                target["condition_names"], target["counts"], target["rates"])):
            output.append({"branch": branch, "rung_zero_based": rung, "condition": name,
                           "successes": count, "draws": target["rollouts"], "rate": rate})
    return output


def write_csv(path, rows):
    assert rows and all(set(r) == set(rows[0]) for r in rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def latex_table(rows, decision, completion, acceptance=False, book_label=None):
    title = "Engineering acceptance only" if acceptance else tex_escape(book_label) if book_label else "One repaired Mistral outer round"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Branch & Optim. steps & Active tokens & Target pass & Credit pass & Scope pass \\",
             r"\hline"]
    for row in rows:
        lines.append(" & ".join([
            tex_escape(row["branch"]), str(row["optimizer_steps"]), str(row["nonzero_advantage_tokens"]),
            f"{row['endpoint_full_successes']}/{row['endpoint_draws']}",
            f"{row['credit_condition_count']}/{row['endpoint_draws']}",
            f"{row['scope_full_successes']}/{row['scope_draws']}"]) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
        r"\caption{" + title + ". Each trained branch receives " + str(rows[1]["loss_tokens"]) +
        " loss tokens. Active tokens have nonzero binary-reward group advantages; optimizer steps may also act through KL and momentum. " +
        "Target and credit counts use fresh fixed-checkpoint draws. Credit condition: " +
        tex_escape(rows[0]["credit_condition"]) + ". Retained Solver: " +
        tex_escape(decision["selected"]["name"]) + ". Its separate post-selection check yields " +
        f"{completion['counts'][-1]}/{completion['rollouts']} full passes. " +
        "Scope preservation is an observed check, not a population guarantee.}",
        r"\label{tab:verge-repaired-round" + ("-" + book_label.replace(" ", "-").replace(":", "").replace(",", "").replace("/", "-") if book_label else "") + "}", r"\end{table}", ""]
    return "\n".join(lines)


def make_figure(root, rows, conditions, decision, acceptance, book_label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = [r["branch"].replace("candidate_", "C") for r in rows]
    colors = ["#7a8593", "#343e4d", "#176a9e", "#3c8576", "#ad6642"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    matrix = np.array([r["rate"] for r in conditions]).reshape(len(rows), -1) * 100
    ax = axes[0, 0]
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(rows)), names)
    ax.set_xticks(range(11), ["Parse", "Exec."] + [f"LCP{i}" for i in range(1, 5)] +
                  [f"Exact{i}" for i in range(1, 5)] + ["Full"], rotation=45, ha="right")
    ax.set_title("A  Endpoint cumulative conditions (%)", loc="left")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7,
                color="white" if value > 50 else "#20364a")
    fig.colorbar(im, ax=ax, shrink=.75)
    ax = axes[0, 1]
    ax.axvline(0, color="#83909c", linestyle="--", linewidth=1)
    for n, row in enumerate(rows[2:]):
        low, high, gain = (row[k] * 100 for k in ("selection_ci95_low", "selection_ci95_high", "selection_gain"))
        ax.plot([low, high], [n, n], color=colors[n + 2], linewidth=2)
        ax.scatter([gain], [n], color=colors[n + 2], s=45)
        suffix = " (identity)" if row["identity_zero_update"] else ""
        ax.text(.98, n + .17, f"{gain:+.2f} [{low:+.2f}, {high:+.2f}]{suffix}",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8)
    ax.set_yticks(range(3), names[2:])
    ax.set_ylim(-.5, 2.6)
    span = max(.5, max(abs(r[k] * 100) for r in rows[2:] for k in ("selection_ci95_low", "selection_ci95_high")))
    ax.set_xlim(-span * 1.3, span * 1.7)
    ax.set_xlabel("Selection-condition gain vs. direct (percentage points)")
    ax.set_title("B  Paired exploratory 95% intervals", loc="left")
    ax = axes[1, 0]
    values = [r["endpoint_full_rate"] * 100 for r in rows]
    ax.bar(names, values, color=colors)
    ax.set_ylim(0, max(1, max(values) * 1.3))
    for i, row in enumerate(rows):
        ax.annotate(f"{row['endpoint_full_successes']}/{row['endpoint_draws']}", (i, values[i]),
                    xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_ylabel("Full-pass rate (%)")
    ax.set_title("C  Fixed-checkpoint target outcomes", loc="left")
    ax = axes[1, 1]
    values = [r["nonzero_advantage_tokens"] / 1000 for r in rows[1:]]
    ax.bar(names[1:], values, color=colors[1:])
    ax.set_ylim(0, max(1, max(values) * 1.3))
    for i, row in enumerate(rows[1:]):
        ax.annotate(f"{row['optimizer_steps']} optimizer steps", (i, values[i]),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("Nonzero-advantage loss tokens (thousands)")
    ax.set_title("D  Actual training signal", loc="left")
    fig.suptitle("Engineering acceptance — not scientific results" if acceptance else book_label or
                 "VERGE: one start, three curricula, one equal-token direct control", fontsize=12)
    fig.savefig(root / "main_figure.png", dpi=180)
    plt.close(fig)


def export(root, frozen, branches, decision, teacher, completion):
    root = Path(root)
    rows = result_rows(frozen, branches, decision)
    conditions = condition_rows(frozen, branches)
    acceptance = frozen["config"].get("acceptance_only", False)
    write_csv(root / "branch_results.csv", rows)
    write_csv(root / "condition_profiles.csv", conditions)
    book_label = None
    if frozen["config"].get("book_suite"):
        cfg = frozen["config"]
        book_label = f"Experiment book: {cfg['book_arm']}, round {cfg['book_round'] + 1}/6"
    (root / "result_table.tex").write_text(latex_table(rows, decision, completion, acceptance, book_label), encoding="utf-8")
    make_figure(root, rows, conditions, decision, acceptance, book_label)
    metadata = {"analysis_only": True, "new_model_draws": 0, "new_training_seeds": 0,
        "rate_units": "fraction [0,1] in CSV; percent or percentage points in figure",
        "missing_values": "empty CSV fields mean not applicable, not zero",
        "endpoint_sampling": "fresh draws at each fixed endpoint; no trajectory pooling",
        "paired_interval": decision["confidence_note"],
        "identity_override": "If both branches made zero optimizer steps, signed gain and interval are zero by training provenance; raw observed estimates remain in CSV",
        "proposal_source": "round_frozen.json:generation; all three proposals, no replacement",
        "teacher_updated": teacher["updated"], "acceptance_only": acceptance,
        "scope_note": "Both scope metrics must satisfy the frozen observed non-decrease rule; no population guarantee"}
    (root / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ["branch_results.csv", "condition_profiles.csv", "result_table.tex", "main_figure.png", "analysis_metadata.json"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exp = Path(__file__).resolve().parents[1]
    source, destination = (exp / args.root).resolve(), (exp / args.output).resolve()
    if not source.is_relative_to(exp) or not destination.is_relative_to(exp):
        raise ValueError("Analysis paths must stay inside this experiment")
    frozen = json.loads((source / "round_frozen.json").read_text())
    archive = json.loads((source / "archive.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    print(json.dumps(export(destination, frozen, {int(k): v for k, v in archive["branch_endpoints"].items()},
                            archive["decision"], archive["challenger_update"], archive["completion"])))
