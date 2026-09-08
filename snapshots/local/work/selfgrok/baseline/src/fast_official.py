from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, load_config, read_json, stable_int, verify_full, write_jsonl
from evaluate_branch import hidden_cases
from generation import summarize_rollouts
from train_branch import latest_checkpoint
from vllm_runtime import VLLMRolloutClient


RAW_ROOT = EXP_ROOT / "raw_results" / "protocol_v2" / "branches"


def prepare_sealed() -> Path:
    marker = EXP_ROOT / "manifests" / "official_test_unsealed_fast.json"
    if marker.exists():
        return Path(read_json(marker)["path"])
    statistical = read_json(EXP_ROOT / "aggregated_results" / "statistical_summary.json")
    if not statistical.get("official_test_authorized"):
        raise RuntimeError("confirmation did not authorize sealed official test")
    config = load_config()
    dataset = load_dataset(
        "manufactoria/prepend_sequence_test",
        cache_dir=str(DATA_ROOT.parent / "caches" / "huggingface" / "datasets"),
    )["train"]
    if len(dataset) != int(config["data"]["official_test"]):
        raise RuntimeError(f"expected 108 official rows, received {len(dataset)}")
    path = DATA_ROOT / "official_sealed_fast" / "prepend_sequence_test"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    atomic_json(
        marker,
        {
            "repository": "manufactoria/prepend_sequence_test",
            "rows": len(dataset),
            "dataset_fingerprint": dataset._fingerprint,
            "unsealed_after_confirmation": True,
            "path": str(path),
        },
    )
    return path


def get_branch(branch_id: str) -> dict[str, Any]:
    branches = read_json(EXP_ROOT / "manifests" / "fast_branches.json")["branches"]
    return next(item for item in branches if item["branch_id"] == branch_id)


def evaluate_official(branch_id: str, server_url: str, gpu: int) -> dict[str, Any]:
    marker = EXP_ROOT / "manifests" / "official_test_unsealed_fast.json"
    if not marker.exists():
        raise RuntimeError("official test remains sealed")
    branch = get_branch(branch_id)
    checkpoint_update, checkpoint = latest_checkpoint(branch_id, 100)
    if checkpoint is None or checkpoint_update != 100:
        raise RuntimeError(f"missing update-100 checkpoint for {branch_id}")
    rows = [
        dict(row)
        for row in load_from_disk(
            str(DATA_ROOT / "official_sealed_fast" / "prepend_sequence_test")
        )
    ]
    client = VLLMRolloutClient(server_url, gpu)
    try:
        client.load_lora(f"official_{branch_id}", checkpoint)
        raw_path = RAW_ROOT / branch_id / "official_test" / "update_0100.jsonl"
        scored = client.score_rows(
            rows,
            int(load_config()["evaluation"]["rollouts_per_instance"]),
            stable_int("fast_official", branch["seed"], branch.get("candidate_id")),
            raw_path,
        )
        hidden_count = int(load_config()["evaluation"]["hidden_tests_per_confirmed_program"])
        augmented = []
        confirmed = 0
        for item in scored:
            hidden_reward = 0
            if item["reward"]:
                row = rows[int(item["instance_index"])]
                hidden_reward = verify_full(
                    item["completion"], hidden_cases(row, hidden_count)
                ).reward
            value = {
                **item,
                "hidden_full_pass": hidden_reward,
                "confirmed_reward": int(item["reward"] and hidden_reward),
            }
            augmented.append(value)
            confirmed += value["confirmed_reward"]
        hidden_path = raw_path.with_name("update_0100_hidden.jsonl")
        write_jsonl(hidden_path, augmented)
        summary = {
            "branch_id": branch_id,
            "seed": branch["seed"],
            "kind": branch["kind"],
            "candidate_id": branch.get("candidate_id"),
            "split": "official_test",
            "update": 100,
            **summarize_rollouts(scored),
            "confirmed_full_pass_count": confirmed,
            "confirmed_full_pass_rate": confirmed / len(scored),
            "hidden_tests_per_official_success": hidden_count,
        }
        atomic_json(raw_path.with_name("update_0100_summary.json"), summary)
        return summary
    finally:
        client.use_base()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-sealed", action="store_true")
    parser.add_argument("--branch-id")
    parser.add_argument("--server-url")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.prepare_sealed:
        print(prepare_sealed(), flush=True)
        return
    if args.branch_id is None or args.server_url is None or args.gpu is None:
        parser.error("branch evaluation requires --branch-id, --server-url, and --gpu")
    print(evaluate_official(args.branch_id, args.server_url, args.gpu), flush=True)


if __name__ == "__main__":
    main()
