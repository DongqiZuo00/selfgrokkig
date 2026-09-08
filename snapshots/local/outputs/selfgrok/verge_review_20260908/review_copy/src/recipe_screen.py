from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, load_config, uncertainty_matched
from generation import summarize_rollouts
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


SCREEN_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "screen"


def score(
    client: VLLMRolloutClient,
    rows: list[dict[str, Any]],
    samples: int,
    seed: int,
    output: Path,
) -> list[dict[str, Any]]:
    return client.score_rows(rows, samples, seed, output)


def main() -> None:
    config = load_config()
    root_update = int(config["training"]["root_updates"])
    adapter = (
        EXP_ROOT
        / "checkpoints"
        / "recipe__root__basic__seed42"
        / f"resume_u{root_update:04d}"
    )
    if not (adapter / "adapter_config.json").exists():
        raise RuntimeError(f"root adapter missing: {adapter}")

    target = [
        dict(row)
        for row in load_from_disk(str(DATA_ROOT / "target_splits" / "initial_audit"))
    ]
    regex = [
        dict(row)
        for row in load_from_disk(
            str(DATA_ROOT / "candidate_splits" / "regex" / "audit")
        )
    ]
    compr = [
        dict(row)
        for row in load_from_disk(
            str(DATA_ROOT / "candidate_splits" / "compr" / "audit")
        )
    ]

    with VLLMServerPool(gpus=(0, 1), base_port=8100) as urls:
        clients = [
            VLLMRolloutClient(urls[0], 0),
            VLLMRolloutClient(urls[1], 1),
        ]
        for index, client in enumerate(clients):
            client.load_lora(f"recipe_root_screen_gpu{index}", adapter)

        midpoint = len(target) // 2
        tasks = [
            (
                clients[0],
                target[:midpoint],
                int(config["audit"]["target_rollouts_per_instance"]),
                2026090101,
                SCREEN_ROOT / "target_audit_part0.jsonl",
            ),
            (
                clients[1],
                target[midpoint:],
                int(config["audit"]["target_rollouts_per_instance"]),
                2026090102,
                SCREEN_ROOT / "target_audit_part1.jsonl",
            ),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            target_parts = list(executor.map(lambda values: score(*values), tasks))
        target_scored = target_parts[0] + target_parts[1]

        candidate_tasks = [
            (
                clients[0],
                regex,
                16,
                2026090111,
                SCREEN_ROOT / "regex_audit.jsonl",
            ),
            (
                clients[1],
                compr,
                16,
                2026090112,
                SCREEN_ROOT / "compr_audit.jsonl",
            ),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            candidate_parts = list(
                executor.map(lambda values: score(*values), candidate_tasks)
            )
        regex_scored, compr_scored = candidate_parts
        for client in clients:
            client.use_base()

    target_summary = summarize_rollouts(target_scored)
    regex_summary = summarize_rollouts(regex_scored)
    compr_summary = summarize_rollouts(compr_scored)
    regex_p = float(regex_summary["full_pass_rate"])
    compr_p = float(compr_summary["full_pass_rate"])
    success_gap = abs(regex_p - compr_p)
    uncertainty_gap = abs(
        uncertainty_matched(regex_p) - uncertainty_matched(compr_p)
    )
    target_is_zero = int(target_summary["full_pass_count"]) == 0
    pair_is_matched = success_gap <= 0.05 and uncertainty_gap <= 0.05

    result = {
        "protocol_version": "recipe_witness_v1",
        "checkpoint": str(adapter),
        "target_family": "has",
        "target_audit": target_summary,
        "candidates": {
            "regex": {
                **regex_summary,
                "uncertainty": uncertainty_matched(regex_p),
            },
            "compr": {
                **compr_summary,
                "uncertainty": uncertainty_matched(compr_p),
            },
        },
        "matching": {
            "success_rate_gap": success_gap,
            "uncertainty_gap": uncertainty_gap,
            "max_success_rate_gap": 0.05,
            "max_uncertainty_gap": 0.05,
            "strict_pair_match": pair_is_matched,
        },
        "regime": {
            "target_pass_at_128_is_zero": target_is_zero,
            "theorem_instantiation_preconditions_met": target_is_zero
            and pair_is_matched,
        },
        "selection": {
            "target_development_used": False,
            "target_confirmation_used": False,
            "official_test_used": False,
            "branches_continue_regardless_of_screen_result": True,
            "reason": "REGEX and COMPR are frozen positive controls from RL Grok Recipe",
        },
    }
    atomic_json(EXP_ROOT / "manifests" / "recipe_screen.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
