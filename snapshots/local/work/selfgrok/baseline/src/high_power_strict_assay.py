from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from scipy.stats import beta, binomtest, fisher_exact

from common import DATA_ROOT, EXP_ROOT, atomic_json, read_json
from generation import summarize_rollouts
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


OUTPUT_ROOT = EXP_ROOT / "raw_results" / "high_power_strict_assay_v1"
DIRECT_CHECKPOINT = (
    EXP_ROOT
    / "checkpoints"
    / "recipe__direct__has__seed42"
    / "resume_u0100"
)
CURRICULUM_CHECKPOINT = (
    EXP_ROOT
    / "checkpoints"
    / "recipe__decisive__regex_dense"
    / "resume_u0200"
)


def exact_interval(successes: int, trials: int) -> dict[str, float]:
    low = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    return {"low": low, "high": high}


def instance_success(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row["instance_id"])
        result[key] = max(result.get(key, 0), int(row["reward"]))
    return result


def score(
    client: VLLMRolloutClient,
    rows: list[dict[str, Any]],
    seed: int,
    output: Path,
) -> list[dict[str, Any]]:
    return client.score_rows(rows, 128, seed, output)


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "high_power_strict_assay.json")
    development = [
        dict(row)
        for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))
    ]
    if len(development) != 50:
        raise RuntimeError(f"expected 50 fixed development instances: {len(development)}")
    for checkpoint in (DIRECT_CHECKPOINT, CURRICULUM_CHECKPOINT):
        if not (checkpoint / "adapter_config.json").exists():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")

    with VLLMServerPool(gpus=(0, 1), base_port=8400) as urls:
        direct_client = VLLMRolloutClient(urls[0], 0)
        curriculum_client = VLLMRolloutClient(urls[1], 1)
        direct_client.load_lora("strict_assay_direct_u100", DIRECT_CHECKPOINT)
        curriculum_client.load_lora(
            "strict_assay_curriculum_u200",
            CURRICULUM_CHECKPOINT,
        )
        tasks = [
            (
                direct_client,
                development,
                int(protocol["no_search"]["fixed_operational_rng"]["direct"]),
                OUTPUT_ROOT / "direct_u0100.jsonl",
            ),
            (
                curriculum_client,
                development,
                int(protocol["no_search"]["fixed_operational_rng"]["curriculum"]),
                OUTPUT_ROOT / "curriculum_u0200.jsonl",
            ),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            direct_rows, curriculum_rows = list(
                executor.map(lambda values: score(*values), tasks)
            )
        direct_client.use_base()
        curriculum_client.use_base()

    expected = 6400
    if len(direct_rows) != expected or len(curriculum_rows) != expected:
        raise RuntimeError(
            f"incomplete assay: direct={len(direct_rows)}, "
            f"curriculum={len(curriculum_rows)}"
        )
    direct_summary = summarize_rollouts(direct_rows)
    curriculum_summary = summarize_rollouts(curriculum_rows)
    direct_success = int(direct_summary["full_pass_count"])
    curriculum_success = int(curriculum_summary["full_pass_count"])
    direct_instances = instance_success(direct_rows)
    curriculum_instances = instance_success(curriculum_rows)
    common = sorted(set(direct_instances) & set(curriculum_instances))
    curriculum_only = sum(
        curriculum_instances[key] == 1 and direct_instances[key] == 0
        for key in common
    )
    direct_only = sum(
        direct_instances[key] == 1 and curriculum_instances[key] == 0
        for key in common
    )
    discordant = curriculum_only + direct_only
    mcnemar_p = (
        float(
            binomtest(
                curriculum_only,
                discordant,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    fisher = fisher_exact(
        [
            [curriculum_success, expected - curriculum_success],
            [direct_success, expected - direct_success],
        ],
        alternative="greater",
    )
    criterion = protocol["predeclared_decisive_criterion"]
    curriculum_distinct = sum(curriculum_instances.values())
    positive = bool(
        curriculum_success
        >= int(criterion["curriculum_min_strict_successes"])
        and curriculum_distinct
        >= int(criterion["curriculum_min_distinct_success_instances"])
        and curriculum_success > direct_success
        and float(fisher.pvalue)
        < float(criterion["one_sided_fisher_exact_p_below"])
        and mcnemar_p
        < float(criterion["paired_instance_pass_at_128_mcnemar_p_below"])
    )
    all_zero = direct_success == 0 and curriculum_success == 0
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": (
            "positive"
            if positive
            else "all_zero"
            if all_zero
            else "nonzero_not_decisive"
        ),
        "strict_rollouts_per_checkpoint": expected,
        "direct": {
            **direct_summary,
            "strict_rate_ci95_exact": exact_interval(direct_success, expected),
            "distinct_success_instances": sum(direct_instances.values()),
        },
        "curriculum": {
            **curriculum_summary,
            "strict_rate_ci95_exact": exact_interval(curriculum_success, expected),
            "distinct_success_instances": curriculum_distinct,
        },
        "comparison": {
            "absolute_strict_rate_difference": (
                curriculum_success - direct_success
            )
            / expected,
            "fisher_exact_one_sided_p": float(fisher.pvalue),
            "fisher_odds_ratio": (
                float(fisher.statistic)
                if math.isfinite(float(fisher.statistic))
                else None
            ),
            "curriculum_only_pass_at_128_instances": curriculum_only,
            "direct_only_pass_at_128_instances": direct_only,
            "paired_mcnemar_exact_one_sided_p": mcnemar_p,
        },
        "decision": {
            "decisive_nonzero_gamma": positive,
            "criterion": criterion,
            "do_not_increase_k_if_all_zero": all_zero,
            "official_test_remains_sealed": True,
        },
    }
    atomic_json(
        EXP_ROOT / "aggregated_results" / "high_power_strict_assay.json",
        result,
    )
    atomic_json(
        EXP_ROOT / "manifests" / "high_power_strict_assay_complete.json",
        {
            "status": result["status"],
            "summary": "aggregated_results/high_power_strict_assay.json",
        },
    )
    lines = [
        "# High-power strict Gamma assay",
        "",
        f"- Direct strict successes: {direct_success}/{expected}",
        f"- Curriculum strict successes: {curriculum_success}/{expected}",
        f"- Curriculum distinct pass@128 instances: {curriculum_distinct}/50",
        f"- One-sided Fisher exact p: {float(fisher.pvalue):.6g}",
        f"- Paired instance McNemar p: {mcnemar_p:.6g}",
        f"- Decisive nonzero Gamma: {positive}",
        f"- Assay status: {result['status']}",
        "",
        "No partial target reward, training, seed/configuration search, confirmation, "
        "or official test was used.",
        "",
    ]
    (EXP_ROOT / "RESULTS_HIGH_POWER_ASSAY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
