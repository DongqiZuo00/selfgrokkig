from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from common import (
    DATA_ROOT,
    EXP_ROOT,
    atomic_json,
    branch_id,
    load_config,
    package_versions,
    read_json,
    uncertainty_rzero,
)
from generation import InferenceEngine, summarize_rollouts
from prepare_data import generate_distribution, materialize_selected_training


def dataset_rows(path: Path) -> list[dict]:
    return [dict(row) for row in load_from_disk(str(path))]


def audit_target(engine: InferenceEngine, config: dict) -> dict:
    manifest_path = EXP_ROOT / "manifests" / "target_stratum.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    rows = dataset_rows(DATA_ROOT / "target_splits" / "initial_audit")
    audit_root = EXP_ROOT / "raw_results" / "target_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    level = 0
    while True:
        raw_path = audit_root / f"stratum_{level}.jsonl"
        scored = engine.score_rows(
            rows,
            int(config["audit"]["target_rollouts_per_instance"]),
            int(config["data"]["split_seed"]) + level,
            raw_path,
            int(config["audit"]["generation_batch_size"]),
        )
        summary = summarize_rollouts(scored)
        if summary["full_pass_count"] == 0:
            result = {
                "stratum_index": level,
                "source": (
                    "manufactoria/prepend_sequence_train fixed initial-audit split"
                    if level == 0
                    else "official PrependSequenceGenerator escalated stratum"
                ),
                "generator_parameters": (
                    {"official_dataset_difficulty": "hard", "official_rows": 32}
                    if level == 0
                    else {
                        "sequence_lengths": [6 + 2 * level, 8 + 2 * level],
                        "pattern_mutations": True,
                        "color_mode": "two_color",
                    }
                ),
                "audit_data_path": str(
                    DATA_ROOT / "target_splits" / "initial_audit"
                    if level == 0
                    else DATA_ROOT / "target_escalation" / f"stratum_{level}"
                ),
                **summary,
                "criterion": "No audited solution passes the complete verifier.",
            }
            atomic_json(manifest_path, result)
            return result
        level += 1
        candidate_config = {
            "name": f"target_escalation_{level}",
            "color_mode": "two_color",
            "sequence_lengths": [6 + 2 * level, 8 + 2 * level],
            "count_thresholds": [1, 4],
            "numerical_thresholds": [1, 8],
            "regex_max_pattern_length": [1, 1],
            "prepend_pattern_mutations": True,
        }
        rows = generate_distribution(
            "prepend_sequence",
            candidate_config,
            int(config["data"]["initial_audit"]),
            int(config["data"]["split_seed"]) + level,
            f"target_escalation_{level}",
        )
        path = DATA_ROOT / "target_escalation" / f"stratum_{level}"
        from datasets import Dataset

        Dataset.from_list(rows).save_to_disk(str(path))


def audit_candidates(engine: InferenceEngine, config: dict) -> tuple[list[dict], list[dict]]:
    pool = read_json(EXP_ROOT / "manifests" / "candidate_pool_generation_manifest.json")
    result_path = EXP_ROOT / "aggregated_results" / "candidate_uncertainty_full.json"
    if result_path.exists():
        full_table = read_json(result_path)
    else:
        full_table = []
        for entry_index, entry in enumerate(pool["entries"]):
            rows = dataset_rows(Path(entry["audit_data_path"]))
            raw_path = (
                EXP_ROOT / "raw_results" / "candidate_audits" / f"{entry['candidate_id']}.jsonl"
            )
            scored = engine.score_rows(
                rows,
                int(config["audit"]["candidate_rollouts_per_instance"]),
                int(config["data"]["candidate_generation_seed"]) + entry_index,
                raw_path,
                int(config["audit"]["generation_batch_size"]),
            )
            summary = summarize_rollouts(scored)
            summary.update(
                {
                    "candidate_id": entry["candidate_id"],
                    "family": entry["family"],
                    "generator_parameters": entry["configuration"],
                    "instances": len(rows),
                    "rollouts_per_instance": int(
                        config["audit"]["candidate_rollouts_per_instance"]
                    ),
                    "solver_uncertainty": uncertainty_rzero(summary["full_pass_rate"]),
                    "selected": False,
                }
            )
            full_table.append(summary)
            atomic_json(result_path, full_table)
    family_winners = []
    for family in sorted({row["family"] for row in full_table}):
        alternatives = [row for row in full_table if row["family"] == family]
        alternatives.sort(
            key=lambda row: (
                -row["solver_uncertainty"],
                abs(row["full_pass_rate"] - 0.5),
                row["candidate_id"],
            )
        )
        family_winners.append(alternatives[0])
    family_winners.sort(
        key=lambda row: (-row["solver_uncertainty"], row["candidate_id"])
    )
    selected = family_winners[: int(config["candidate_pool"]["selected_families"])]
    selected_ids = {row["candidate_id"] for row in selected}
    for row in full_table:
        row["selected"] = row["candidate_id"] in selected_ids
    atomic_json(result_path, full_table)
    return full_table, selected


def freeze_manifest(config: dict, target: dict, full_table: list[dict], selected: list[dict]) -> dict:
    frozen_path = EXP_ROOT / "manifests" / "frozen_manifest.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if existing["selected_candidate_ids"] != [row["candidate_id"] for row in selected]:
            raise RuntimeError("refusing to replace frozen candidate selection")
        return existing
    selected_entries = []
    for row in selected:
        training_path = materialize_selected_training(row["candidate_id"], 480)
        selected_entries.append(
            {
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "solver_uncertainty": row["solver_uncertainty"],
                "full_pass_rate": row["full_pass_rate"],
                "generator_parameters": row["generator_parameters"],
                "training_data_path": str(training_path),
            }
        )
    branches = []
    for seed in config["training"]["seeds"]:
        branches.append(
            {
                "branch_id": branch_id("direct", int(seed)),
                "kind": "direct",
                "seed": int(seed),
                "candidate_id": None,
                "priority": 0,
            }
        )
        for rank, entry in enumerate(selected_entries):
            branches.append(
                {
                    "branch_id": branch_id("candidate", int(seed), entry["candidate_id"]),
                    "kind": "candidate",
                    "seed": int(seed),
                    "candidate_id": entry["candidate_id"],
                    "priority": 1 + rank,
                }
            )
    frozen = {
        "target_stratum": target,
        "uncertainty_definition": {
            "formula": config["audit"]["uncertainty_definition"],
            "source": "R-Zero paper Solver uncertainty definition; official repository commit 5699329 does not expose a standalone selector function",
        },
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "selected_candidates": selected_entries,
        "complete_candidate_table": full_table,
        "training_config": config["training"],
        "sampling_config": config["sampling"],
        "initial_common_budget": int(config["training"]["initial_total_updates"]),
        "branches": sorted(branches, key=lambda row: (row["priority"], row["seed"])),
        "software": package_versions(),
        "upstream_commits": {
            "rl-grok-recipe": "8500bec",
            "ROLL": "5304da2",
            "R-Zero": "5699329",
        },
        "official_prepend_test_accessed": False,
    }
    atomic_json(frozen_path, frozen)
    return frozen


def main() -> None:
    config = load_config()
    if (EXP_ROOT / "manifests" / "audit_and_selection_complete.json").exists():
        print("audit and selection already complete", flush=True)
        return
    engine = InferenceEngine()
    target = audit_target(engine, config)
    full_table, selected = audit_candidates(engine, config)
    frozen = freeze_manifest(config, target, full_table, selected)
    atomic_json(
        EXP_ROOT / "manifests" / "audit_and_selection_complete.json",
        {
            "selected_candidate_ids": frozen["selected_candidate_ids"],
            "full_candidate_configurations": len(full_table),
            "non_prepend_families": len({row["family"] for row in full_table}),
        },
    )
    print("frozen candidates:", frozen["selected_candidate_ids"], flush=True)


if __name__ == "__main__":
    main()
