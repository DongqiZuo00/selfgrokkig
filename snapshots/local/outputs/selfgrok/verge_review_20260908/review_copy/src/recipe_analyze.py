from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
BRANCHES = {
    "direct": "recipe__direct__has__seed42",
    "regex": "recipe__candidate__regex__seed42",
    "compr": "recipe__candidate__compr__seed42",
}


def instance_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance_id"]), []).append(int(row["reward"]))
    return {
        instance_id: sum(values) / len(values)
        for instance_id, values in grouped.items()
    }


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    seed: int,
    samples: int = 10000,
) -> dict[str, float]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise RuntimeError("no common development instances")
    observed = sum(left[item] - right[item] for item in ids) / len(ids)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        drawn = [ids[rng.randrange(len(ids))] for _ in ids]
        values.append(sum(left[item] - right[item] for item in drawn) / len(drawn))
    values.sort()
    return {
        "estimate": observed,
        "ci95_low": values[int(0.025 * samples)],
        "ci95_high": values[int(0.975 * samples)],
        "instances": len(ids),
        "bootstrap_samples": samples,
    }


def load_endpoint(branch_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = RAW_ROOT / branch_id / "development"
    return (
        read_json(root / "update_0500_summary.json"),
        read_jsonl(root / "update_0500.jsonl"),
    )


def main() -> None:
    screen = read_json(EXP_ROOT / "manifests" / "recipe_screen.json")
    summaries: dict[str, Any] = {}
    rates: dict[str, dict[str, float]] = {}
    for name, branch_id in BRANCHES.items():
        summary, rows = load_endpoint(branch_id)
        summaries[name] = summary
        rates[name] = instance_rates(rows)

    regex_vs_direct = paired_bootstrap(rates["regex"], rates["direct"], 2026090201)
    compr_vs_direct = paired_bootstrap(rates["compr"], rates["direct"], 2026090202)
    regex_vs_compr = paired_bootstrap(rates["regex"], rates["compr"], 2026090203)

    positive_discovery = (
        bool(screen["regime"]["theorem_instantiation_preconditions_met"])
        and regex_vs_direct["estimate"] > 0.0
        and regex_vs_compr["estimate"] > 0.0
    )
    result = {
        "protocol_version": "recipe_witness_v1",
        "stage": "single_fixed_run",
        "screen": screen,
        "endpoint_summaries": summaries,
        "paired_effects": {
            "regex_minus_direct": regex_vs_direct,
            "compr_minus_direct": compr_vs_direct,
            "regex_minus_compr": regex_vs_compr,
        },
        "decision": {
            "positive_discovery": positive_discovery,
            "requires_multi_seed_confirmation": False,
            "official_test_remains_sealed": True,
            "paper_claim_status": (
                "single_fixed_run_positive_evidence"
                if positive_discovery
                else "frozen_single_run_null_or_precondition_failure"
            ),
        },
    }
    atomic_json(
        EXP_ROOT / "aggregated_results" / "recipe_witness_summary.json",
        result,
    )

    lines = [
        "# HAS transfer witness: frozen single-seed discovery",
        "",
        f"- Target pass@128 audit is zero: {screen['regime']['target_pass_at_128_is_zero']}",
        f"- REGEX/COMPR strict difficulty match: {screen['matching']['strict_pair_match']}",
        f"- Direct HAS endpoint pass rate: {summaries['direct']['full_pass_rate']:.6f}",
        f"- REGEX curriculum endpoint pass rate: {summaries['regex']['full_pass_rate']:.6f}",
        f"- COMPR curriculum endpoint pass rate: {summaries['compr']['full_pass_rate']:.6f}",
        f"- REGEX minus direct: {regex_vs_direct['estimate']:.6f} "
        f"[{regex_vs_direct['ci95_low']:.6f}, {regex_vs_direct['ci95_high']:.6f}]",
        f"- COMPR minus direct: {compr_vs_direct['estimate']:.6f} "
        f"[{compr_vs_direct['ci95_low']:.6f}, {compr_vs_direct['ci95_high']:.6f}]",
        f"- REGEX minus COMPR: {regex_vs_compr['estimate']:.6f} "
        f"[{regex_vs_compr['ci95_low']:.6f}, {regex_vs_compr['ci95_high']:.6f}]",
        f"- Positive discovery: {positive_discovery}",
        "",
        "The official test was not opened. A positive discovery advances to the frozen "
        "single fixed-run confirmation policy; no seed or hash search.",
        "",
    ]
    output = EXP_ROOT / "RESULTS_RECIPE.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
