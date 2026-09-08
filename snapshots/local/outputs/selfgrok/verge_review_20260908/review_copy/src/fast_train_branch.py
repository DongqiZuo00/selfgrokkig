from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from common import (
    DATA_ROOT,
    EXP_ROOT,
    apply_chat_template,
    atomic_json,
    group_advantages,
    load_config,
    read_json,
    stable_int,
)
from generation import summarize_rollouts
from train_branch import (
    build_model,
    build_optimizer,
    latest_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    target_order,
    trajectory_loss,
)
from vllm_runtime import VLLMRolloutClient


RAW_ROOT = EXP_ROOT / "raw_results" / "protocol_v2" / "branches"


def get_branch(branch_id: str) -> dict[str, Any]:
    manifest = read_json(EXP_ROOT / "manifests" / "fast_branches.json")
    return next(item for item in manifest["branches"] if item["branch_id"] == branch_id)


def rows_for_update(
    branch: dict[str, Any],
    update: int,
    target_dataset,
    target_indices: list[int],
    candidate_dataset,
    prompts_per_update: int,
) -> tuple[list[dict[str, Any]], str]:
    if branch["kind"] == "candidate" and update <= 60:
        start = (update - 1) * prompts_per_update
        return [
            dict(candidate_dataset[index]) for index in range(start, start + prompts_per_update)
        ], "candidate"
    start = (update - 1) * prompts_per_update
    indices = target_indices[start : start + prompts_per_update]
    return [dict(target_dataset[index]) for index in indices], "prepend"


def sync_adapter(
    model,
    client: VLLMRolloutClient,
    branch_id: str,
    update: int,
) -> tuple[Path, float]:
    root = EXP_ROOT / "checkpoints" / "runtime_lora" / branch_id
    path = root / f"u{update:04d}"
    started = time.perf_counter()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    client.load_lora(f"{branch_id}_u{update:04d}", path)
    for old in sorted(root.glob("u*")):
        if old != path:
            resolved = old.resolve()
            if root.resolve() not in resolved.parents:
                raise RuntimeError(f"refusing to remove runtime adapter outside root: {resolved}")
            shutil.rmtree(old)
    return path, time.perf_counter() - started


def evaluate(
    client: VLLMRolloutClient,
    branch: dict[str, Any],
    split: str,
    update: int,
    rows: list[dict[str, Any]],
    samples: int,
) -> dict[str, Any]:
    output = RAW_ROOT / branch["branch_id"] / split / f"update_{update:04d}.jsonl"
    scored = client.score_rows(
        rows,
        samples,
        stable_int("fast_eval", branch["seed"], branch.get("candidate_id"), split, update),
        output,
    )
    summary = {
        "branch_id": branch["branch_id"],
        "kind": branch["kind"],
        "seed": branch["seed"],
        "candidate_id": branch.get("candidate_id"),
        "split": split,
        "update": update,
        "instances": len(rows),
        "samples_per_instance": samples,
        **summarize_rollouts(scored),
    }
    atomic_json(output.with_name(f"update_{update:04d}_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    config = load_config()
    branch = get_branch(args.branch_id)
    goal = 100
    start_update, checkpoint = latest_checkpoint(args.branch_id, goal)
    tokenizer, model = build_model(int(branch["seed"]), checkpoint)
    optimizer, scheduler = build_optimizer(model, config)
    cumulative = {
        "training_rollouts": 0,
        "training_generated_tokens": 0,
        "rollout_wall_seconds": 0.0,
        "learner_wall_seconds": 0.0,
        "lora_sync_wall_seconds": 0.0,
    }
    if checkpoint is not None:
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cumulative.update(state["runtime_state"]["cumulative"])
    target_dataset = load_from_disk(str(DATA_ROOT / "target_splits" / "target_train"))
    candidate_dataset = (
        load_from_disk(branch["training_data_path"]) if branch["kind"] == "candidate" else None
    )
    prompts_per_update = int(config["training"]["prompts_per_update"])
    target_indices = target_order(
        int(branch["seed"]), goal, prompts_per_update, len(target_dataset)
    )
    development = [
        dict(row) for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))
    ]
    client = VLLMRolloutClient(args.server_url, args.gpu)
    science_updates = {0, 60, 100}
    if start_update == 0:
        client.use_base()
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.branch_id,
            0,
            {"cumulative": cumulative},
        )
        prune_checkpoints(args.branch_id, science_updates)
    else:
        _, sync_seconds = sync_adapter(model, client, args.branch_id, start_update)
        cumulative["lora_sync_wall_seconds"] += sync_seconds
    train_root = RAW_ROOT / args.branch_id / "train"
    try:
        for update in range(start_update + 1, goal + 1):
            rows, phase = rows_for_update(
                branch,
                update,
                target_dataset,
                target_indices,
                candidate_dataset,
                prompts_per_update,
            )
            rollout_path = train_root / f"update_{update:04d}.jsonl"
            rollout_started = time.perf_counter()
            scored = client.score_rows(
                rows,
                int(config["training"]["rollouts_per_prompt"]),
                stable_int("fast_train", branch["seed"], branch.get("candidate_id"), update),
                rollout_path,
            )
            rollout_wall = time.perf_counter() - rollout_started
            rewards = [int(item["reward"]) for item in scored]
            advantages = group_advantages(
                rewards, int(config["training"]["rollouts_per_prompt"])
            )
            cumulative["training_rollouts"] += len(scored)
            cumulative["training_generated_tokens"] += sum(
                int(item["completion_tokens"]) for item in scored
            )
            cumulative["rollout_wall_seconds"] += rollout_wall

            learner_started = time.perf_counter()
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            kl_values: list[float] = []
            losses: list[float] = []
            for item, advantage in zip(scored, advantages):
                row = rows[int(item["instance_index"])]
                prompt_ids = tokenizer(
                    apply_chat_template(tokenizer, row["messages"]),
                    add_special_tokens=False,
                ).input_ids
                loss, kl = trajectory_loss(
                    model,
                    prompt_ids,
                    [int(value) for value in item["completion_token_ids"]],
                    advantage,
                    float(config["training"]["grpo_clip_epsilon"]),
                    float(config["training"]["kl_beta"]),
                )
                (loss / len(scored)).backward()
                losses.append(float(loss.detach().cpu()))
                kl_values.append(kl)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["training"]["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            learner_wall = time.perf_counter() - learner_started
            cumulative["learner_wall_seconds"] += learner_wall

            adapter_path, sync_wall = sync_adapter(model, client, args.branch_id, update)
            cumulative["lora_sync_wall_seconds"] += sync_wall
            monitor = None
            endpoint = None
            if update == 60:
                monitor = evaluate(
                    client,
                    branch,
                    "development_monitor",
                    update,
                    development[: int(config["evaluation"]["monitor_instances"])],
                    int(config["evaluation"]["monitor_rollouts_per_instance"]),
                )
            if update == 100:
                endpoint_split = str(branch["endpoint_split"])
                endpoint_rows = [
                    dict(row)
                    for row in load_from_disk(
                        str(DATA_ROOT / "target_splits" / endpoint_split)
                    )
                ]
                endpoint = evaluate(
                    client,
                    branch,
                    endpoint_split,
                    update,
                    endpoint_rows,
                    int(config["evaluation"]["rollouts_per_instance"]),
                )
            metric = {
                "branch_id": args.branch_id,
                "seed": int(branch["seed"]),
                "candidate_id": branch.get("candidate_id"),
                "update": update,
                "phase": phase,
                "rollouts": len(scored),
                "successes": sum(rewards),
                "reward_mean": sum(rewards) / len(rewards),
                "reward_variance": float(
                    torch.tensor(rewards, dtype=torch.float32).var(unbiased=False)
                ),
                "loss": sum(losses) / len(losses),
                "kl": sum(kl_values) / len(kl_values),
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "rollout_wall_seconds_update": rollout_wall,
                "learner_wall_seconds_update": learner_wall,
                "lora_sync_wall_seconds_update": sync_wall,
                "monitor": monitor,
                "endpoint": endpoint,
                **cumulative,
            }
            atomic_json(train_root / f"update_{update:04d}_summary.json", metric)
            if (
                update % int(config["training"]["resume_interval"]) == 0
                or update in science_updates
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    args.branch_id,
                    update,
                    {"cumulative": cumulative},
                )
                prune_checkpoints(args.branch_id, science_updates)
            print(json.dumps(metric), flush=True)
        atomic_json(
            RAW_ROOT / args.branch_id / "status.json",
            {
                "branch_id": args.branch_id,
                "completed_updates": 100,
                "fixed_budget": True,
                "extended": False,
                "status": "complete",
                **cumulative,
            },
        )
    finally:
        client.use_base()


if __name__ == "__main__":
    main()
