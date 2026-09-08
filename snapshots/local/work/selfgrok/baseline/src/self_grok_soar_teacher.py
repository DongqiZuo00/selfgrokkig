from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

from common import EXP_ROOT, MODEL_ROOT, apply_chat_template, atomic_json, load_config, read_json, stable_int
from recipe_train_branch import build_recipe_model, recipe_loss
from self_grok_soar_loop import (
    PROTOCOL_PATH,
    decision_path,
    dispatch_students,
    generation_path,
    register_branches,
)
from train_branch import build_optimizer, prune_checkpoints, save_checkpoint
from vllm_runtime import VLLMRolloutClient, VLLMServerPool, http_json
from modeling import load_tokenizer


TEACHER_BRANCH_ID = "mistral_self_grok_soar_teacher"


def teacher_checkpoint(update: int) -> Path:
    return EXP_ROOT / "checkpoints" / TEACHER_BRANCH_ID / f"resume_u{update:04d}"


def teacher_messages(cfg: dict[str, Any]) -> list[dict[str, str]]:
    families = ", ".join(cfg["teacher"]["allowed_families"])
    return [
        {
            "role": "system",
            "content": (
                "You design compact training curricula for a student that writes Manufactoria DSL programs. "
                "You never see the hidden target tasks. Your only learning signal is whether training on your "
                "curriculum later improves the student's verified performance on those hidden tasks."
            ),
        },
        {
            "role": "user",
            "content": (
                "Propose one diverse curriculum specification. Select one to three components from the "
                f"following official task families: {families}. Each component must have a level chosen from "
                "low, medium, high and an integer weight from 1 through 4. Favor combinations that may teach "
                "reusable program structure rather than superficial formatting. Do not mention or infer the "
                "hidden target. Output exactly one JSON object with this schema and no additional keys: "
                '{"components":[{"family":"starts_with","level":"low","weight":1}]}'
            ),
        },
    ]


def parse_spec(text: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    parsed = None
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "components" in value:
            parsed = value
            break
    if parsed is None or set(parsed) != {"components"}:
        return None
    components = parsed["components"]
    maximum = int(cfg["teacher"]["maximum_components_per_curriculum"])
    if not isinstance(components, list) or not 1 <= len(components) <= maximum:
        return None
    allowed = set(cfg["teacher"]["allowed_families"])
    levels = set(cfg["teacher"]["levels"])
    normalized = []
    seen = set()
    for item in components:
        if not isinstance(item, dict) or set(item) != {"family", "level", "weight"}:
            return None
        family = str(item["family"])
        level = str(item["level"])
        try:
            weight = int(item["weight"])
        except (TypeError, ValueError):
            return None
        key = (family, level)
        if family not in allowed or level not in levels or not 1 <= weight <= 4 or key in seen:
            return None
        seen.add(key)
        normalized.append({"family": family, "level": level, "weight": weight})
    return {"components": normalized}


def initialize_teacher() -> Path:
    path = teacher_checkpoint(0)
    if path.exists() and (path / "state.pt").exists():
        return path
    tokenizer, model = build_recipe_model(42, None)
    config = load_config()
    optimizer, scheduler = build_optimizer(model, config)
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        TEACHER_BRANCH_ID,
        0,
        {"role": "target-blind curriculum teacher", "completed_outer_updates": 0},
    )
    del tokenizer, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return path


def update_teacher(outer: int) -> Path:
    if outer == 0:
        return initialize_teacher()
    destination = teacher_checkpoint(outer)
    if destination.exists() and (destination / "state.pt").exists():
        return destination
    previous = teacher_checkpoint(outer - 1)
    if not (previous / "state.pt").exists():
        raise RuntimeError(f"missing prior teacher checkpoint: {previous}")
    cfg = read_json(PROTOCOL_PATH)
    generation = read_json(generation_path(outer - 1))
    decision = read_json(decision_path(outer - 1))
    rewards = [float(value) for value in decision["teacher_rewards"]]
    if len(rewards) != len(generation["candidates"]):
        raise RuntimeError("teacher reward count does not match generated candidates")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    advantages = [0.0 for _ in rewards]
    if std >= 1e-6:
        advantages = [(value - mean) / (std + 1e-6) for value in rewards]

    tokenizer, model = build_recipe_model(42, previous)
    train_config = load_config()
    train_config["training"]["learning_rate"] = float(cfg["teacher"]["learning_rate"])
    optimizer, scheduler = build_optimizer(model, train_config)
    prior_state = torch.load(previous / "state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(prior_state["optimizer"])
    scheduler.load_state_dict(prior_state["scheduler"])
    optimizer.zero_grad(set_to_none=True)
    active = sum(abs(value) > 1e-12 for value in advantages)
    losses = []
    if active:
        prompt_ids = tokenizer(
            apply_chat_template(tokenizer, generation["teacher_messages"]),
            add_special_tokens=False,
        ).input_ids
        for candidate, advantage in zip(generation["candidates"], advantages):
            if abs(advantage) <= 1e-12:
                continue
            loss, _ = recipe_loss(
                model,
                prompt_ids,
                [int(value) for value in candidate["completion_token_ids"]],
                advantage,
                float(cfg["teacher"]["kl_beta"]),
            )
            (loss / active).backward()
            losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        TEACHER_BRANCH_ID,
        outer,
        {
            "role": "target-blind curriculum teacher",
            "completed_outer_updates": outer,
            "source_outer": outer - 1,
            "rewards": rewards,
            "advantages": advantages,
            "zero_advantage": active == 0,
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
        },
    )
    prune_checkpoints(TEACHER_BRANCH_ID, {outer}, resumable_to_keep=2)
    del tokenizer, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def generate_candidates(outer: int, checkpoint: Path, port: int) -> dict[str, Any]:
    path = generation_path(outer)
    if path.exists():
        return read_json(path)
    cfg = read_json(PROTOCOL_PATH)
    messages = teacher_messages(cfg)
    tokenizer = load_tokenizer("left")
    prompt = apply_chat_template(tokenizer, messages)
    required = int(cfg["teacher"]["candidates_per_outer_step"])
    candidates: list[dict[str, Any]] = []
    rejected = 0
    with VLLMServerPool(gpus=(0,), base_port=port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora(f"soar_teacher_o{outer:03d}", checkpoint)
        for attempt in range(int(cfg["teacher"]["maximum_generation_attempts"])):
            response = http_json(
                urls[0] + "/v1/completions",
                {
                    "model": client.model_name,
                    "prompt": [prompt],
                    "n": max(4, required - len(candidates)),
                    "max_tokens": int(cfg["teacher"]["completion_tokens"]),
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": stable_int("soar_teacher_generation", outer, attempt),
                },
                timeout=3600,
            )
            for choice in sorted(response["choices"], key=lambda item: int(item["index"])):
                completion = str(choice["text"])
                spec = parse_spec(completion, cfg)
                if spec is None:
                    rejected += 1
                    continue
                candidates.append(
                    {
                        "candidate_index": len(candidates),
                        "spec": spec,
                        "completion": completion,
                        "completion_token_ids": [
                            int(value)
                            for value in tokenizer(completion, add_special_tokens=False).input_ids
                        ],
                        "generation_attempt": attempt,
                    }
                )
                if len(candidates) == required:
                    break
            if len(candidates) == required:
                break
        client.use_base()
    if len(candidates) != required:
        raise RuntimeError(
            f"teacher produced only {len(candidates)}/{required} valid curriculum specs "
            f"after rejecting {rejected} outputs"
        )
    branch_ids = register_branches(outer, candidates)
    result = {
        "outer": outer,
        "teacher_checkpoint": str(checkpoint.relative_to(EXP_ROOT)),
        "teacher_messages": messages,
        "candidates": candidates,
        "branch_ids": branch_ids,
        "rejected_format_outputs": rejected,
        "target_visible_to_teacher": False,
    }
    atomic_json(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    checkpoint = update_teacher(args.outer)
    result = generate_candidates(args.outer, checkpoint, args.port)
    submission = dispatch_students(args.outer)
    print(json.dumps({"generation": result, "submission": submission}, indent=2), flush=True)


if __name__ == "__main__":
    main()
