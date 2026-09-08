from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from common import (
    DATA_ROOT,
    EXP_ROOT,
    MODEL_REPO_ID,
    WORK_ROOT,
    atomic_json,
    canonical_sha256,
    ensure_dirs,
    load_config,
)


SHARED_DATA = WORK_ROOT / "data" / "official_train"
PROTOCOL_VERSION = "recipe_witness_v1"


def shuffled_indices(size: int, seed: int, label: str) -> list[int]:
    rng = random.Random(canonical_sha256({"seed": seed, "label": label}))
    indices = list(range(size))
    rng.shuffle(indices)
    return indices


def materialize(source, indices: list[int], destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.select(indices).save_to_disk(str(destination))


def dataset_ids(dataset) -> list[str]:
    return [str(value) for value in dataset["id"]]


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = int(config["data"]["split_seed"])

    target = load_from_disk(str(SHARED_DATA / "has_train"))
    regex = load_from_disk(str(SHARED_DATA / "regex_train"))
    compr = load_from_disk(str(SHARED_DATA / "compr_train"))
    basic = load_from_disk(str(SHARED_DATA / "basic_mix_train"))

    required = {
        "target_train": int(config["data"]["target_train"]),
        "initial_audit": int(config["data"]["initial_audit"]),
        "development": int(config["data"]["development"]),
        "confirmation": int(config["data"]["confirmation"]),
    }
    if sum(required.values()) != len(target):
        raise RuntimeError(
            f"HAS split contract uses {sum(required.values())} rows, source has {len(target)}"
        )

    target_order = shuffled_indices(len(target), seed, "has")
    cursor = 0
    target_splits: dict[str, list[int]] = {}
    for name, count in required.items():
        target_splits[name] = target_order[cursor : cursor + count]
        cursor += count

    candidate_splits: dict[str, dict[str, list[int]]] = {}
    for family, dataset in (("regex", regex), ("compr", compr)):
        order = shuffled_indices(len(dataset), seed, family)
        audit_count = 64
        candidate_splits[family] = {
            "audit": order[:audit_count],
            "train": order[audit_count:],
        }

    for name, indices in target_splits.items():
        materialize(target, indices, DATA_ROOT / "target_splits" / name)
    for family, dataset in (("regex", regex), ("compr", compr)):
        for name, indices in candidate_splits[family].items():
            materialize(dataset, indices, DATA_ROOT / "candidate_splits" / family / name)
    materialize(basic, list(range(len(basic))), DATA_ROOT / "root" / "basic_mix")

    split_manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "split_seed": seed,
        "sources": {
            "has": str(SHARED_DATA / "has_train"),
            "regex": str(SHARED_DATA / "regex_train"),
            "compr": str(SHARED_DATA / "compr_train"),
            "basic_mix": str(SHARED_DATA / "basic_mix_train"),
        },
        "target": {},
        "candidates": {},
        "root": {
            "count": len(basic),
            "id_sha256": canonical_sha256(dataset_ids(basic)),
        },
        "sealed_official_test_touched": False,
    }
    for name, indices in target_splits.items():
        subset = target.select(indices)
        split_manifest["target"][name] = {
            "count": len(subset),
            "id_sha256": canonical_sha256(dataset_ids(subset)),
            "path": str(DATA_ROOT / "target_splits" / name),
        }
    for family, dataset in (("regex", regex), ("compr", compr)):
        split_manifest["candidates"][family] = {}
        for name, indices in candidate_splits[family].items():
            subset = dataset.select(indices)
            split_manifest["candidates"][family][name] = {
                "count": len(subset),
                "id_sha256": canonical_sha256(dataset_ids(subset)),
                "path": str(DATA_ROOT / "candidate_splits" / family / name),
            }
    split_manifest["manifest_sha256"] = canonical_sha256(split_manifest)
    atomic_json(EXP_ROOT / "manifests" / "recipe_data.json", split_manifest)

    root_id = "recipe__root__basic__seed42"
    branches = [
        {
            "branch_id": root_id,
            "kind": "root",
            "seed": 42,
            "goal_updates": int(config["training"]["root_updates"]),
            "training_data_path": str(DATA_ROOT / "root" / "basic_mix"),
        },
        {
            "branch_id": "recipe__direct__has__seed42",
            "kind": "direct",
            "seed": 42,
            "goal_updates": int(config["training"]["initial_total_updates"]),
            "root_branch_id": root_id,
        },
        {
            "branch_id": "recipe__candidate__regex__seed42",
            "kind": "candidate",
            "candidate_id": "regex",
            "seed": 42,
            "goal_updates": int(config["training"]["initial_total_updates"]),
            "candidate_updates": int(config["training"]["candidate_updates"]),
            "training_data_path": str(DATA_ROOT / "candidate_splits" / "regex" / "train"),
            "root_branch_id": root_id,
        },
        {
            "branch_id": "recipe__candidate__compr__seed42",
            "kind": "candidate",
            "candidate_id": "compr",
            "seed": 42,
            "goal_updates": int(config["training"]["initial_total_updates"]),
            "candidate_updates": int(config["training"]["candidate_updates"]),
            "training_data_path": str(DATA_ROOT / "candidate_splits" / "compr" / "train"),
            "root_branch_id": root_id,
        },
    ]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "claim_layer": "objective_and_loop_with_positive_control",
        "target_family": "has",
        "hard_boundary_family": "prepend_preserved_from_protocol_v2",
        "branches": branches,
        "common_root": {
            "family": "basic_mix",
            "updates": int(config["training"]["root_updates"]),
            "reward": "strict_full_pass",
        },
        "post_root": {
            "direct": "HAS for 500 updates",
            "regex": "REGEX for 100 updates, then HAS for 400 updates",
            "compr": "COMPR for 100 updates, then HAS for 400 updates",
            "reward": "strict_full_pass",
        },
        "fairness": {
            "same_root_adapter": True,
            "fresh_optimizer_per_post_root_branch": True,
            "same_post_root_update_budget": True,
            "budget_unit": "optimizer_updates",
            "prompts_per_update": int(config["training"]["prompts_per_update"]),
            "rollouts_per_prompt": int(config["training"]["rollouts_per_prompt"]),
            "sigma_hat_zero_advantage": "all advantages exactly zero",
            "weight_decay": float(config["training"]["weight_decay"]),
            "target_dev_used_for_selection": False,
            "sealed_official_test_used": False,
        },
        "screen": {
            "target_audit_instances": int(config["data"]["initial_audit"]),
            "target_rollouts_per_instance": int(config["audit"]["target_rollouts_per_instance"]),
            "candidate_audit_instances": 64,
            "candidate_rollouts_per_instance": 16,
            "max_success_rate_gap": 0.05,
            "max_uncertainty_gap": 0.05,
        },
        "recipe_alignment": {
            "inherited": [
                MODEL_REPO_ID,
                "Manufactoria official verifier",
                "8192 response-token cap",
                "temperature 1.0",
                "beta 0",
                "strict full-pass reward after shared root",
            ],
            "deliberate_resource_adaptations": [
                "2 B200 maximum instead of the paper's multi-node run",
                "LoRA rank 64 rather than full-model training",
                "16 prompts x 8 rollouts per update rather than 48 x 16",
                "common easy-task root rather than per-test dense reward on HAS",
            ],
        },
    }
    protocol["manifest_sha256"] = canonical_sha256(protocol)
    atomic_json(EXP_ROOT / "manifests" / "recipe_branches.json", protocol)
    print(f"prepared {len(branches)} frozen branches under {DATA_ROOT}", flush=True)


if __name__ == "__main__":
    main()
