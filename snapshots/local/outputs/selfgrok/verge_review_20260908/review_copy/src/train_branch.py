from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from common import (
    DATA_ROOT,
    EXP_ROOT,
    MODEL_ROOT,
    apply_chat_template,
    atomic_json,
    group_advantages,
    load_config,
    read_json,
    seed_everything,
    stable_int,
)
from generation import InferenceEngine, summarize_rollouts
from modeling import assert_adapter_backbone, load_text_model, load_tokenizer, lora_target_modules


def get_branch(branch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = read_json(EXP_ROOT / "manifests" / "frozen_manifest.json")
    branch = next(item for item in frozen["branches"] if item["branch_id"] == branch_id)
    return frozen, branch


def target_order(seed: int, required_updates: int, prompts_per_update: int, size: int) -> list[int]:
    rng = random.Random(stable_int("target_order", seed))
    result: list[int] = []
    while len(result) < required_updates * prompts_per_update:
        block = list(range(size))
        rng.shuffle(block)
        result.extend(block)
    return result


def latest_checkpoint(branch_id: str, goal: int) -> tuple[int, Path | None]:
    root = EXP_ROOT / "checkpoints" / branch_id
    options: list[tuple[int, Path]] = []
    if root.exists():
        for path in root.glob("resume_u*"):
            try:
                update = int(path.name.replace("resume_u", ""))
            except ValueError:
                continue
            if update <= goal and (path / "adapter_config.json").exists() and (path / "state.pt").exists():
                options.append((update, path))
    return max(options, default=(0, None), key=lambda item: item[0])


def build_model(seed: int, checkpoint: Path | None):
    seed_everything(seed)
    tokenizer = load_tokenizer("left")
    base = load_text_model()
    if checkpoint is None:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=lora_target_modules("all-linear"),
            bias="none",
        )
        model = get_peft_model(base, lora)
    else:
        assert_adapter_backbone(checkpoint)
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return tokenizer, model


def build_optimizer(model, config: dict[str, Any]):
    training = config["training"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=tuple(float(x) for x in training["betas"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup = max(1, round(int(training["initial_total_updates"]) * float(training["warmup_ratio"])))

    def multiplier(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup))

    scheduler = LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def selected_candidate(frozen: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return next(item for item in frozen["selected_candidates"] if item["candidate_id"] == candidate_id)


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
        indices = list(range(start, start + prompts_per_update))
        return [dict(candidate_dataset[index]) for index in indices], "candidate"
    start = (update - 1) * prompts_per_update
    indices = target_indices[start : start + prompts_per_update]
    return [dict(target_dataset[index]) for index in indices], "prepend"


def chosen_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    normalizer = torch.logsumexp(logits.float(), dim=-1)
    return chosen - normalizer


def trajectory_loss(
    model,
    prompt_ids: list[int],
    completion_ids: list[int],
    advantage: float,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[torch.Tensor, float]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    targets = input_ids[:, 1:]
    start = len(prompt_ids) - 1
    with torch.no_grad():
        with model.disable_adapter():
            reference_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1, :]
            reference_logp = chosen_log_probs(reference_logits, targets)[:, start:]
        del reference_logits
    current_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits[:, :-1, :]
    current_logp = chosen_log_probs(current_logits, targets)[:, start:]
    del current_logits
    old_logp = current_logp.detach()
    log_ratio = current_logp - old_logp
    ratio = torch.exp(log_ratio)
    scalar_advantage = torch.tensor(float(advantage), device=device)
    unclipped = ratio * scalar_advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * scalar_advantage
    surrogate = torch.minimum(unclipped, clipped)
    ref_delta = reference_logp - current_logp
    kl = torch.exp(ref_delta) - ref_delta - 1.0
    loss = -(surrogate - kl_beta * kl).mean()
    return loss, float(kl.detach().mean().cpu())


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    branch_id: str,
    update: int,
    runtime_state: dict[str, Any],
) -> Path:
    path = EXP_ROOT / "checkpoints" / branch_id / f"resume_u{update:04d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "runtime_state": runtime_state,
            "update": update,
        },
        path / "state.pt",
    )
    atomic_json(
        EXP_ROOT / "checkpoints" / branch_id / "latest.json",
        {"update": update, "path": str(path)},
    )
    return path


def prune_checkpoints(
    branch_id: str,
    science_updates: set[int],
    resumable_to_keep: int = 2,
) -> None:
    """Bound disk use while retaining every adapter needed for evaluation.

    Science checkpoints keep their adapter files. Only the newest rolling
    checkpoints keep optimizer/scheduler state, which is much larger and is
    needed solely to resume an interrupted run.
    """
    root = EXP_ROOT / "checkpoints" / branch_id
    root_resolved = root.resolve()
    entries: list[tuple[int, Path]] = []
    for path in root.glob("resume_u*"):
        try:
            update = int(path.name.replace("resume_u", ""))
        except ValueError:
            continue
        resolved = path.resolve()
        if root_resolved not in resolved.parents:
            raise RuntimeError(f"Refusing to prune outside checkpoint root: {resolved}")
        entries.append((update, path))
    entries.sort(key=lambda item: item[0])

    resumable = [update for update, path in entries if (path / "state.pt").exists()]
    keep_state = set(resumable[-resumable_to_keep:])
    for update, path in entries:
        if update in science_updates:
            state_path = path / "state.pt"
            if update not in keep_state and state_path.exists():
                state_path.unlink()
        elif update not in keep_state:
            shutil.rmtree(path)


def evaluate_development(
    model,
    tokenizer,
    branch_id: str,
    update: int,
    seed: int,
    cumulative: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    output_path = (
        EXP_ROOT
        / "raw_results"
        / "branches"
        / branch_id
        / "development"
        / f"update_{update:04d}.jsonl"
    )
    rows = [dict(row) for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))]
    engine = InferenceEngine(model=model, tokenizer=tokenizer)
    scored = engine.score_rows(
        rows,
        int(config["evaluation"]["rollouts_per_instance"]),
        stable_int("dev", seed, update),
        output_path,
        int(config["training"]["generation_batch_size"]),
    )
    summary = summarize_rollouts(scored)
    summary.update(
        {
            "branch_id": branch_id,
            "update": update,
            "reward_variance": float(
                torch.tensor([row["reward"] for row in scored], dtype=torch.float32).var(
                    unbiased=False
                )
            ),
            "target_successes": summary["full_pass_count"],
            "cumulative_training_rollouts": cumulative["training_rollouts"],
            "cumulative_training_generated_tokens": cumulative["training_generated_tokens"],
        }
    )
    atomic_json(
        EXP_ROOT
        / "raw_results"
        / "branches"
        / branch_id
        / "development"
        / f"update_{update:04d}_summary.json",
        summary,
    )
    model.train()
    model.config.use_cache = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--goal-updates", required=True, type=int)
    args = parser.parse_args()
    config = load_config()
    frozen, branch = get_branch(args.branch_id)
    start_update, checkpoint = latest_checkpoint(args.branch_id, args.goal_updates)
    tokenizer, model = build_model(int(branch["seed"]), checkpoint)
    optimizer, scheduler = build_optimizer(model, config)
    cumulative = {"training_rollouts": 0, "training_generated_tokens": 0}
    first_target_success_update = None
    if checkpoint is not None:
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cumulative = state["runtime_state"]["cumulative"]
        first_target_success_update = state["runtime_state"].get("first_target_success_update")
    target_dataset = load_from_disk(str(DATA_ROOT / "target_splits" / "target_train"))
    candidate_dataset = None
    if branch["kind"] == "candidate":
        entry = selected_candidate(frozen, branch["candidate_id"])
        candidate_dataset = load_from_disk(entry["training_data_path"])
    prompts_per_update = int(config["training"]["prompts_per_update"])
    target_indices = target_order(
        int(branch["seed"]), args.goal_updates, prompts_per_update, len(target_dataset)
    )
    required_evals = set(int(x) for x in config["evaluation"]["checkpoints"])
    if args.goal_updates > int(config["training"]["initial_total_updates"]):
        required_evals.update(
            range(
                int(config["training"]["initial_total_updates"])
                + int(config["training"]["extension_block_updates"]),
                args.goal_updates + 1,
                int(config["training"]["extension_block_updates"]),
            )
        )
    science_updates = required_evals | {0}
    if start_update == 0:
        initial_summary = evaluate_development(
            model, tokenizer, args.branch_id, 0, int(branch["seed"]), cumulative, config
        )
        if initial_summary["full_pass_count"] and first_target_success_update is None:
            first_target_success_update = 0
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.branch_id,
            0,
            {
                "cumulative": cumulative,
                "first_target_success_update": first_target_success_update,
            },
        )
        prune_checkpoints(args.branch_id, science_updates)
    train_root = EXP_ROOT / "raw_results" / "branches" / args.branch_id / "train"
    for update in range(start_update + 1, args.goal_updates + 1):
        rows, phase = rows_for_update(
            branch,
            update,
            target_dataset,
            target_indices,
            candidate_dataset,
            prompts_per_update,
        )
        model.eval()
        model.config.use_cache = True
        engine = InferenceEngine(model=model, tokenizer=tokenizer)
        rollout_path = train_root / f"update_{update:04d}.jsonl"
        scored = engine.score_rows(
            rows,
            int(config["training"]["rollouts_per_prompt"]),
            stable_int("train", branch["seed"], branch.get("candidate_id"), update),
            rollout_path,
            int(config["training"]["generation_batch_size"]),
        )
        rewards = [int(item["reward"]) for item in scored]
        advantages = group_advantages(rewards, int(config["training"]["rollouts_per_prompt"]))
        cumulative["training_rollouts"] += len(scored)
        cumulative["training_generated_tokens"] += sum(
            int(item["completion_tokens"]) for item in scored
        )
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        kl_values: list[float] = []
        losses: list[float] = []
        for item, advantage in zip(scored, advantages):
            row = rows[int(item["instance_index"])]
            prompt_ids = tokenizer(
                apply_chat_template(tokenizer, row["messages"]), add_special_tokens=False
            ).input_ids
            loss, kl = trajectory_loss(
                model,
                prompt_ids,
                [int(x) for x in item["completion_token_ids"]],
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
        metric = {
            "branch_id": args.branch_id,
            "seed": int(branch["seed"]),
            "candidate_id": branch.get("candidate_id"),
            "update": update,
            "phase": phase,
            "rollouts": len(scored),
            "successes": sum(rewards),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_variance": float(torch.tensor(rewards, dtype=torch.float32).var(unbiased=False)),
            "loss": sum(losses) / len(losses),
            "kl": sum(kl_values) / len(kl_values),
            "grad_norm": float(grad_norm),
            "learning_rate": scheduler.get_last_lr()[0],
            **cumulative,
        }
        atomic_json(train_root / f"update_{update:04d}_summary.json", metric)
        if update in required_evals:
            dev = evaluate_development(
                model,
                tokenizer,
                args.branch_id,
                update,
                int(branch["seed"]),
                cumulative,
                config,
            )
            if dev["full_pass_count"] and first_target_success_update is None:
                first_target_success_update = update
        if (
            update % int(config["training"]["resume_interval"]) == 0
            or update in required_evals
            or update == args.goal_updates
        ):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.branch_id,
                update,
                {
                    "cumulative": cumulative,
                    "first_target_success_update": first_target_success_update,
                },
            )
            prune_checkpoints(args.branch_id, science_updates)
        print(json.dumps(metric), flush=True)
    atomic_json(
        EXP_ROOT / "raw_results" / "branches" / args.branch_id / "status.json",
        {
            "branch_id": args.branch_id,
            "completed_updates": args.goal_updates,
            "common_budget": args.goal_updates,
            "first_target_success_update": first_target_success_update,
            **cumulative,
            "status": "complete_to_budget",
        },
    )


if __name__ == "__main__":
    main()
