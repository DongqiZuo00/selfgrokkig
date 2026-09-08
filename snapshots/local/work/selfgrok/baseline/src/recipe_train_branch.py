from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from common import (
    DATA_ROOT,
    EXP_ROOT,
    MODEL_ROOT,
    apply_chat_template,
    atomic_json,
    group_advantages,
    load_config,
    read_json,
    read_jsonl,
    seed_everything,
    stable_int,
)
from fast_train_branch import sync_adapter
from generation import summarize_rollouts
from train_branch import (
    build_optimizer,
    chosen_log_probs,
    latest_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from vllm_runtime import VLLMRolloutClient
from modeling import (
    assert_adapter_backbone,
    load_text_model,
    load_tokenizer,
    lora_target_modules,
)


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"


def get_branch(branch_id: str) -> dict[str, Any]:
    paths = [
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_6_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_5_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_4_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_3_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_2_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_1_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v5_0_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_9_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_8_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_7_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_6_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_5_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_4_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_3_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_2_bootstrap.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_v4_2_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_decisive_mistral_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_soar_mistral_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_soar_runtime_branches.json",
        EXP_ROOT / "manifests" / "self_grok_strict_handoff_round4.json",
        EXP_ROOT / "manifests" / "self_grok_consolidation_round5.json",
        EXP_ROOT / "manifests" / "self_grok_bridge_chain_round6.json",
        EXP_ROOT / "manifests" / "self_grok_main_round3.json",
        EXP_ROOT / "manifests" / "soft_phi_round2_stochastic_ladder.json",
        EXP_ROOT / "manifests" / "soft_phi_round2_strict_ladder.json",
        EXP_ROOT / "manifests" / "soft_phi_rollout_round1.json",
        EXP_ROOT / "manifests" / "strict_has_revision.json",
        EXP_ROOT / "manifests" / "decisive_branches.json",
        EXP_ROOT / "manifests" / "recipe_branches.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        manifest = read_json(path)
        branches = manifest.get("branches", [manifest.get("branch")])
        for item in branches:
            if item and item["branch_id"] == branch_id:
                return item
    raise KeyError(f"unknown branch: {branch_id}")


def root_checkpoint(branch: dict[str, Any], config: dict[str, Any]) -> Path:
    explicit_parent = branch.get("parent_checkpoint")
    if explicit_parent:
        path = EXP_ROOT / str(explicit_parent)
        if not (path / "adapter_config.json").exists():
            raise RuntimeError(f"explicit parent checkpoint is not ready: {path}")
        return path
    update = int(config["training"]["root_updates"])
    path = (
        EXP_ROOT
        / "checkpoints"
        / str(branch["root_branch_id"])
        / f"resume_u{update:04d}"
    )
    if not (path / "adapter_config.json").exists():
        raise RuntimeError(f"common root checkpoint is not ready: {path}")
    return path


def build_recipe_model(seed: int, adapter: Path | None):
    seed_everything(seed)
    tokenizer = load_tokenizer("left")
    base = load_text_model()
    config = load_config()
    if adapter is None:
        training = config["training"]
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(training["lora_rank"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            target_modules=lora_target_modules(str(training["lora_target"])),
            bias="none",
        )
        model = get_peft_model(base, lora)
    else:
        assert_adapter_backbone(adapter)
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return tokenizer, model


def build_reference_model(seed: int, adapter: Path):
    """Load an immutable reference policy, including the starting Solver LoRA."""
    seed_everything(seed)
    assert_adapter_backbone(adapter)
    base = load_text_model()
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cycle_rows(
    dataset,
    update: int,
    count: int,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    first = (update - 1) * count
    rows: list[dict[str, Any]] = []
    orders: dict[int, list[int]] = {}
    for position in range(first, first + count):
        epoch, within = divmod(position, len(dataset))
        if epoch not in orders:
            order = list(range(len(dataset)))
            random.Random(stable_int("recipe_cycle", seed, label, epoch)).shuffle(order)
            orders[epoch] = order
        rows.append(dict(dataset[orders[epoch][within]]))
    return rows


def rows_for_update(
    branch: dict[str, Any],
    update: int,
    target,
    source,
    prompts_per_update: int,
) -> tuple[list[dict[str, Any]], str]:
    if branch["kind"] == "root":
        return (
            cycle_rows(source, update, prompts_per_update, int(branch["seed"]), "basic"),
            "basic_mix",
        )
    candidate_updates = int(branch.get("candidate_updates", 0))
    if branch["kind"] in {"candidate", "candidate_dense"} and update <= candidate_updates:
        return (
            cycle_rows(
                source,
                update,
                prompts_per_update,
                int(branch["seed"]),
                str(branch["candidate_id"]),
            ),
            str(branch["candidate_id"]),
        )
    if branch["kind"] == "target_ladder":
        rows = cycle_rows(
            target,
            update,
            prompts_per_update,
            int(branch["seed"]),
            "has",
        )
        test_count: int | None = None
        for stage in branch["strict_test_ladder"]:
            if update <= int(stage["through_update"]):
                raw_count = stage["tests"]
                test_count = None if raw_count == "all" else int(raw_count)
                break
        if test_count is None:
            return rows, "has_strict_all"
        transformed = []
        for row in rows:
            cases = list(row["ground_truth"])
            positive = [i for i, case in enumerate(cases) if bool(case["expected_accepted"])]
            negative = [i for i, case in enumerate(cases) if not bool(case["expected_accepted"])]
            selected = positive[: test_count // 2] + negative[: test_count // 2]
            if len(selected) < test_count:
                selected_set = set(selected)
                selected.extend(i for i in range(len(cases)) if i not in selected_set)
            selected = sorted(selected[:test_count])
            item = dict(row)
            item["ground_truth"] = [cases[i] for i in selected]
            transformed.append(item)
        return transformed, f"has_strict_{test_count}"
    if branch["kind"] == "stochastic_target_ladder":
        rows = cycle_rows(
            target,
            update,
            prompts_per_update,
            int(branch["seed"]),
            "has",
        )
        test_count = ladder_test_count(branch, update)
        return rows, "has_strict_all" if test_count is None else f"has_strict_{test_count}"
    if branch["kind"] == "target_dense_then_strict":
        rows = cycle_rows(
            target,
            update,
            prompts_per_update,
            int(branch["seed"]),
            "has",
        )
        phase = (
            "has_dense_all_tests"
            if update <= int(branch["dense_updates"])
            else "has_strict_full_pass"
        )
        return rows, phase
    if branch["kind"] == "bridge_chain":
        for stage_index, stage in enumerate(branch["stages"]):
            if update > int(stage["through_update"]):
                continue
            if stage["kind"] == "target":
                target_update = update + int(branch.get("target_update_offset", 0))
                return (
                    cycle_rows(
                        target,
                        target_update,
                        prompts_per_update,
                        int(branch["seed"]),
                        "has",
                    ),
                    f"target_stage_{stage_index + 1}",
                )
            stage_start = 1 if stage_index == 0 else int(branch["stages"][stage_index - 1]["through_update"]) + 1
            stage_update = update - stage_start + 1
            target_fraction = float(stage.get("target_mix_fraction", 0.0))
            if target_fraction > 0.0:
                target_count = int(round(prompts_per_update * target_fraction))
                target_count = min(prompts_per_update - 1, max(1, target_count))
                bridge_count = prompts_per_update - target_count
                target_rows = cycle_rows(
                    target,
                    update + int(branch.get("target_update_offset", 0)),
                    target_count,
                    int(branch["seed"]),
                    f"target_mix_stage_{stage_index + 1}",
                )
                bridge_rows = cycle_rows(
                    source[str(stage_index)],
                    stage_update,
                    bridge_count,
                    int(branch["seed"]),
                    str(stage["name"]),
                )
                if stage.get("verification_case_counts") is not None:
                    bridge_rows = [
                        {**row, "_coverage_bridge_verifier": True}
                        for row in bridge_rows
                    ]
                # Preserve fixed group sizes while avoiding a systematic position
                # effect between target and bridge prompts.
                rows = target_rows + bridge_rows
                random.Random(
                    stable_int(
                        "bridge_target_mix",
                        branch["seed"],
                        branch.get("randomness_branch_id", branch["branch_id"]),
                        update,
                    )
                ).shuffle(rows)
                return rows, (
                    f"{stage['name']}__target_mix_{target_fraction:g}"
                )
            return (
                cycle_rows(
                    source[str(stage_index)],
                    stage_update,
                    prompts_per_update,
                    int(branch["seed"]),
                    str(stage["name"]),
                ),
                str(stage["name"]),
            )
        raise RuntimeError(f"bridge chain has no stage for update {update}")
    if branch["kind"] == "grounded_meta_candidate":
        return (
            cycle_rows(
                source,
                update,
                prompts_per_update,
                int(branch["seed"]),
                str(branch["candidate_id"]),
            ),
            "grounded_meta_curriculum",
        )
    target_update = (
        update
        if branch["kind"] == "direct" or branch.get("target_prompt_stream") == "global"
        else update - candidate_updates
    )
    target_update += int(branch.get("target_update_offset", 0))
    return (
        cycle_rows(target, target_update, prompts_per_update, int(branch["seed"]), "has"),
        "has",
    )


def ladder_test_count(branch: dict[str, Any], update: int) -> int | None:
    for stage in branch["strict_test_ladder"]:
        if update <= int(stage["through_update"]):
            return None if stage["tests"] == "all" else int(stage["tests"])
    raise RuntimeError(f"strict ladder has no stage for update {update}")


def bridge_verification_case_sets(
    branch: dict[str, Any],
    stage_index: int,
    stage: dict[str, Any],
    update: int,
    rows: list[dict[str, Any]],
    rollouts_per_prompt: int,
) -> list[list[list[dict[str, Any]]]] | None:
    """Assign strict verifier subsets per rollout for an annealed coverage bridge."""
    raw_counts = stage.get("verification_case_counts")
    if raw_counts is None:
        return None
    counts = [int(value) for value in raw_counts]
    if len(counts) != rollouts_per_prompt:
        raise RuntimeError(
            "verification_case_counts must contain exactly one entry per rollout"
        )
    result: list[list[list[dict[str, Any]]]] = []
    for instance_index, row in enumerate(rows):
        cases = list(row["ground_truth"])
        if not bool(row.get("_coverage_bridge_verifier", False)):
            result.append([cases for _ in counts])
            continue
        assigned = counts.copy()
        random.Random(
            int(branch["seed"]) * 1_000_003
            + update * 10_007
            + stage_index * 1_009
            + instance_index * 257
        ).shuffle(assigned)
        instance_sets: list[list[dict[str, Any]]] = []
        occurrences: dict[int, int] = {}
        subset_mode = str(stage.get("verification_subset_mode", "prefix"))
        for count in assigned:
            if count < 1 or count > len(cases):
                raise RuntimeError(
                    f"invalid verifier case count {count} for {len(cases)} cases"
                )
            if subset_mode == "prefix" or count == len(cases):
                indices = tuple(range(count))
            elif subset_mode == "rotating_combinations":
                choices = list(itertools.combinations(range(len(cases)), count))
                occurrence = occurrences.get(count, 0)
                offset = (
                    int(branch["seed"])
                    + update * 101
                    + stage_index * 17
                    + instance_index
                ) % len(choices)
                indices = choices[(offset + occurrence) % len(choices)]
                occurrences[count] = occurrence + 1
            elif subset_mode == "required_anchor_combinations":
                required = int(stage["verification_required_case_index"])
                if required < 0 or required >= len(cases):
                    raise RuntimeError(
                        f"invalid required verifier case index {required} for "
                        f"{len(cases)} cases"
                    )
                other_indices = [index for index in range(len(cases)) if index != required]
                choices = [
                    tuple(sorted((required, *others)))
                    for others in itertools.combinations(other_indices, count - 1)
                ]
                occurrence = occurrences.get(count, 0)
                offset = (
                    int(branch["seed"])
                    + update * 101
                    + stage_index * 17
                    + instance_index
                ) % len(choices)
                indices = choices[(offset + occurrence) % len(choices)]
                occurrences[count] = occurrence + 1
            else:
                raise RuntimeError(f"unknown verification_subset_mode: {subset_mode}")
            instance_sets.append([cases[index] for index in indices])
        result.append(instance_sets)
    return result


def rollout_verification_cases(
    branch: dict[str, Any],
    update: int,
    rows: list[dict[str, Any]],
    rollouts_per_prompt: int,
) -> list[list[list[dict[str, Any]]]] | None:
    if branch["kind"] == "bridge_chain":
        stage_info = bridge_stage_at_update(branch, update)
        if stage_info is None:
            return None
        stage_index, stage = stage_info
        return bridge_verification_case_sets(
            branch,
            stage_index,
            stage,
            update,
            rows,
            rollouts_per_prompt,
        )
    if branch["kind"] != "stochastic_target_ladder":
        return None
    test_count = ladder_test_count(branch, update)
    if test_count is None:
        return None
    result = []
    for instance_index, row in enumerate(rows):
        cases = list(row["ground_truth"])
        positive = [i for i, case in enumerate(cases) if bool(case["expected_accepted"])]
        negative = [i for i, case in enumerate(cases) if not bool(case["expected_accepted"])]
        instance_sets = []
        for rollout_index in range(rollouts_per_prompt):
            arithmetic_seed = (
                int(branch["seed"]) * 1_000_003
                + update * 10_007
                + instance_index * 257
                + rollout_index
            )
            rng = random.Random(arithmetic_seed)
            pos = positive.copy()
            neg = negative.copy()
            rng.shuffle(pos)
            rng.shuffle(neg)
            selected = pos[: test_count // 2] + neg[: test_count // 2]
            if len(selected) < test_count:
                selected_set = set(selected)
                remaining = [i for i in range(len(cases)) if i not in selected_set]
                rng.shuffle(remaining)
                selected.extend(remaining[: test_count - len(selected)])
            instance_sets.append([cases[i] for i in sorted(selected[:test_count])])
        result.append(instance_sets)
    return result


def annealed_partial_weight(branch: dict[str, Any], update: int) -> float:
    if branch["kind"] in {"grounded_meta_candidate", "grounded_meta_direct"}:
        return float(branch.get("partial_weight", 0.0))
    if branch["kind"] == "bridge_chain":
        for stage in branch["stages"]:
            if update <= int(stage["through_update"]):
                return float(stage["partial_weight"])
        raise RuntimeError(f"bridge chain reward schedule has no stage for update {update}")
    if branch["kind"] != "annealed_target_continuation":
        return 0.0
    for stage in branch["reward_schedule"]:
        if update <= int(stage["through_update"]):
            return float(stage["partial_weight"])
    raise RuntimeError(f"reward schedule has no stage for update {update}")


def bootstrap_success_replay(
    source_branch_id: str,
    replay_dataset,
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): dict(row) for row in replay_dataset}
    replay: list[dict[str, Any]] = []
    seen: set[str] = set()
    train_root = RAW_ROOT / source_branch_id / "train"
    for path in sorted(train_root.glob("update_*.jsonl")):
        if ".telemetry." in path.name:
            continue
        for item in read_jsonl(path):
            instance_id = str(item["instance_id"])
            if int(item["reward"]) != 1 or instance_id in seen:
                continue
            row = by_id.get(instance_id)
            if row is None:
                raise RuntimeError(f"replay instance is absent from target data: {instance_id}")
            seen.add(instance_id)
            replay.append(
                {
                    "instance_id": instance_id,
                    "messages": row["messages"],
                    "completion_token_ids": [int(value) for value in item["completion_token_ids"]],
                    "source_branch_id": source_branch_id,
                    "source_update": int(path.stem.replace("update_", "")),
                }
            )
    return replay


def recipe_loss(
    model,
    prompt_ids: list[int],
    completion_ids: list[int],
    advantage: float,
    kl_beta: float,
    reference_model=None,
) -> tuple[torch.Tensor, float]:
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [prompt_ids + completion_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    targets = input_ids[:, 1:]
    start = len(prompt_ids) - 1
    current_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits[:, :-1, :]
    current_logp = chosen_log_probs(current_logits, targets)[:, start:]
    del current_logits
    policy_loss = -float(advantage) * current_logp.mean()
    if kl_beta == 0.0:
        return policy_loss, 0.0
    if reference_model is None:
        raise RuntimeError("positive KL beta requires an immutable reference policy")
    with torch.no_grad():
        reference_logits = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1, :]
        reference_logp = chosen_log_probs(reference_logits, targets)[:, start:]
        del reference_logits
    ref_delta = reference_logp - current_logp
    kl = (torch.exp(ref_delta) - ref_delta - 1.0).mean()
    return policy_loss + kl_beta * kl, float(kl.detach().cpu())


def evaluate(
    client: VLLMRolloutClient,
    branch: dict[str, Any],
    update: int,
    rows: list[dict[str, Any]],
    samples: int,
    split: str = "development",
) -> dict[str, Any]:
    output = RAW_ROOT / branch["branch_id"] / split / f"update_{update:04d}.jsonl"
    scored = client.score_rows(
        rows,
        samples,
        stable_int(
            "recipe_eval",
            branch["seed"],
            branch.get("evaluation_rng_key", branch.get("candidate_id")),
            split,
            0 if branch.get("align_evaluation_across_updates") else update,
        ),
        output,
        sampling_seed_key=str(
            branch.get(
                "evaluation_rng_key",
                branch.get("candidate_id") or branch["branch_id"],
            )
        ),
    )
    summary = {
        "branch_id": branch["branch_id"],
        "kind": branch["kind"],
        "candidate_id": branch.get("candidate_id"),
        "seed": int(branch["seed"]),
        "split": split,
        "update": update,
        "instances": len(rows),
        "samples_per_instance": samples,
        **summarize_rollouts(scored),
        "mean_case_fraction": sum(
            float(item["passed_cases"]) / max(1, int(item["total_cases"]))
            for item in scored
        )
        / max(1, len(scored)),
    }
    atomic_json(output.with_name(f"update_{update:04d}_summary.json"), summary)
    return summary


def bridge_stage_at_update(branch: dict[str, Any], update: int):
    if branch.get("kind") != "bridge_chain":
        return None
    for index, stage in enumerate(branch["stages"]):
        if update <= int(stage["through_update"]):
            return (index, stage)
    return None


def run_stage_learnability_gate(
    client: VLLMRolloutClient,
    branch: dict[str, Any],
    stage_index: int,
    stage: dict[str, Any],
    dataset,
    update: int,
) -> dict[str, Any]:
    """Reject a bridge stage before training when its GRPO reward is degenerate."""
    gate = dict(branch["stage_learnability_gate"])
    instances = min(int(gate["instances"]), len(dataset))
    rollouts = int(gate["rollouts_per_instance"])
    rows = [dict(dataset[index]) for index in range(instances)]
    if stage.get("verification_case_counts") is not None:
        rows = [{**row, "_coverage_bridge_verifier": True} for row in rows]
    output = (
        RAW_ROOT
        / branch["branch_id"]
        / "stage_gates"
        / f"stage_{stage_index:02d}_before_update_{update:04d}.jsonl"
    )
    gate_seed_branch_id = str(stage.get("gate_seed_branch_id", branch["branch_id"]))
    gate_seed_stage_index = int(stage.get("gate_seed_stage_index", stage_index))
    scored = client.score_rows(
        rows,
        rollouts,
        stable_int(
            "recipe_stage_gate",
            branch["seed"],
            gate_seed_branch_id,
            gate_seed_stage_index,
        ),
        output,
        verification_case_sets=bridge_verification_case_sets(
            branch,
            stage_index,
            stage,
            update,
            rows,
            rollouts,
        ),
        sampling_seed_key=f"{gate_seed_branch_id}__stage_gate_{gate_seed_stage_index}",
    )
    successes = sum(int(item["reward"]) for item in scored)
    rate = successes / len(scored)
    by_instance: dict[int, list[int]] = {}
    for item in scored:
        by_instance.setdefault(int(item["instance_index"]), []).append(int(item["reward"]))
    mixed_groups = sum(len(set(values)) > 1 for values in by_instance.values())
    trainable = (
        rate >= float(gate["minimum_success_rate"])
        and rate <= float(gate["maximum_success_rate"])
        and mixed_groups >= int(gate["minimum_mixed_groups"])
    )
    mastered = rate >= float(gate.get("mastered_success_rate", 0.95))
    accepted = trainable or mastered
    decision = {
        "stage_index": stage_index,
        "stage": stage,
        "before_update": update,
        "rollouts": len(scored),
        "successes": successes,
        "success_rate": rate,
        "mixed_groups": mixed_groups,
        "classification": "trainable" if trainable else ("mastered" if mastered else "too_hard"),
        "accepted": accepted,
        "thresholds": gate,
    }
    atomic_json(output.with_suffix(".decision.json"), decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    branch = get_branch(args.branch_id)
    goal = int(branch["goal_updates"])
    start_update, local_checkpoint = latest_checkpoint(args.branch_id, goal)
    if local_checkpoint is not None:
        initial_adapter = local_checkpoint
    elif branch["kind"] == "root":
        initial_adapter = None
    else:
        initial_adapter = root_checkpoint(branch, config)

    tokenizer, model = build_recipe_model(int(branch["seed"]), initial_adapter)
    optimizer_config = copy.deepcopy(config)
    if "learning_rate" in branch:
        optimizer_config["training"]["learning_rate"] = float(branch["learning_rate"])
    if "warmup_updates" in branch:
        optimizer_config["training"]["initial_total_updates"] = int(branch["warmup_updates"])
        optimizer_config["training"]["warmup_ratio"] = 1.0
    optimizer, scheduler = build_optimizer(model, optimizer_config)
    kl_beta = float(branch.get("kl_beta", config["training"]["kl_beta"]))
    reference_model = None
    if kl_beta > 0.0:
        reference_path = EXP_ROOT / str(
            branch.get("reference_checkpoint", branch.get("parent_checkpoint"))
        )
        if not (reference_path / "adapter_config.json").exists():
            raise RuntimeError(f"immutable reference checkpoint is not ready: {reference_path}")
        reference_model = build_reference_model(int(branch["seed"]), reference_path)
    optimizer_parent_loaded = False
    cumulative = {
        "training_rollouts": 0,
        "training_generated_tokens": 0,
        "rollout_wall_seconds": 0.0,
        "learner_wall_seconds": 0.0,
        "lora_sync_wall_seconds": 0.0,
        "zero_advantage_updates": 0,
        "success_replay_examples_used": 0,
    }
    success_replay: list[dict[str, Any]] = []
    if local_checkpoint is not None:
        state = torch.load(
            local_checkpoint / "state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cumulative.update(state["runtime_state"]["cumulative"])
        success_replay = list(state["runtime_state"].get("success_replay", []))
    elif branch.get("inherit_optimizer_state"):
        parent = root_checkpoint(branch, config)
        parent_state_path = parent / "state.pt"
        if not parent_state_path.exists():
            raise RuntimeError(
                f"optimizer inheritance requested but parent state is missing: {parent_state_path}"
            )
        state = torch.load(
            parent_state_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_parent_update = branch.get("parent_scientific_update")
        if expected_parent_update is not None and int(state["update"]) != int(expected_parent_update):
            raise RuntimeError(
                f"parent update mismatch: expected {expected_parent_update}, got {state['update']}"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        optimizer_parent_loaded = True

    target = None
    if branch["kind"] != "root":
        target_path = branch.get("target_training_path", "data/target_splits/target_train")
        target = load_from_disk(str(EXP_ROOT / str(target_path)))
    target_ids = {str(row["id"]) for row in target} if target is not None else set()
    if (
        start_update == 0
        and not success_replay
        and branch.get("bootstrap_replay_from_branch")
    ):
        replay_dataset = target
        if branch.get("bootstrap_replay_data_path"):
            replay_dataset = load_from_disk(
                str(EXP_ROOT / str(branch["bootstrap_replay_data_path"]))
            )
        success_replay = bootstrap_success_replay(
            str(branch["bootstrap_replay_from_branch"]),
            replay_dataset,
        )
    source = None
    if branch["kind"] in {
        "root",
        "candidate",
        "candidate_dense",
        "grounded_meta_candidate",
    }:
        source = load_from_disk(str(branch["training_data_path"]))
    elif branch["kind"] == "bridge_chain":
        source = {
            str(index): load_from_disk(str(EXP_ROOT / str(stage["training_data_path"])))
            for index, stage in enumerate(branch["stages"])
            if stage["kind"] == "bridge"
        }
    development = None
    if branch["kind"] != "root" and not branch.get("skip_development_evaluation"):
        development = [
            dict(row)
            for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))
        ]
    grounded_reward = None
    if branch.get("grounded_reward_path"):
        grounded_dataset = load_from_disk(str(EXP_ROOT / str(branch["grounded_reward_path"])))
        grounded_reward = [
            dict(grounded_dataset[index])
            for index in range(
                min(
                    int(branch.get("grounded_reward_instances", len(grounded_dataset))),
                    len(grounded_dataset),
                )
            )
        ]
    scope_probe = None
    if branch.get("scope_probe_path"):
        scope_dataset = load_from_disk(str(EXP_ROOT / str(branch["scope_probe_path"])))
        scope_probe = [
            dict(scope_dataset[index])
            for index in range(
                min(int(branch.get("scope_probe_instances", 32)), len(scope_dataset))
            )
        ]

    client = VLLMRolloutClient(args.server_url, args.gpu)
    online_guard = branch.get("online_scope_guard")
    online_scope_reference = None
    last_safe_update = start_update
    science_updates = (
        {0, goal}
        if branch["kind"] == "root"
        else set(int(value) for value in config["evaluation"]["checkpoints"])
    )
    science_updates.update(int(value) for value in branch.get("evaluation_updates", []))
    science_updates.add(goal)
    if start_update == 0:
        if branch["kind"] == "root":
            client.use_base()
        else:
            _, sync_seconds = sync_adapter(model, client, args.branch_id, 0)
            cumulative["lora_sync_wall_seconds"] += sync_seconds
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.branch_id,
            0,
            {"cumulative": cumulative, "success_replay": success_replay},
        )
        prune_checkpoints(args.branch_id, science_updates)
        if development is not None and (
            branch["kind"] == "direct" or branch.get("evaluate_at_start")
        ):
            evaluate(
                client,
                branch,
                0,
                development,
                int(config["evaluation"]["rollouts_per_instance"]),
            )
            if scope_probe is not None:
                evaluate(
                    client,
                    branch,
                    0,
                    scope_probe,
                    int(branch.get("scope_probe_rollouts_per_instance", 8)),
                    split="scope_probe",
                )
        if grounded_reward is not None and branch.get("evaluate_grounded_at_start"):
            evaluate(
                client,
                branch,
                0,
                grounded_reward,
                int(branch.get("grounded_reward_rollouts_per_instance", 4)),
                split="grounded_reward",
            )
        if online_guard and scope_probe is not None:
            online_scope_reference = evaluate(
                client,
                branch,
                0,
                scope_probe,
                int(branch.get("scope_probe_rollouts_per_instance", 8)),
                split="online_scope_probe",
            )
    else:
        _, sync_seconds = sync_adapter(model, client, args.branch_id, start_update)
        cumulative["lora_sync_wall_seconds"] += sync_seconds
        if online_guard and scope_probe is not None:
            online_scope_reference = read_json(
                RAW_ROOT
                / branch["branch_id"]
                / "online_scope_probe"
                / "update_0000_summary.json"
            )

    prompts_per_update = int(
        branch.get("prompts_per_update", config["training"]["prompts_per_update"])
    )
    rollouts_per_prompt = int(
        branch.get("rollouts_per_prompt", config["training"]["rollouts_per_prompt"])
    )
    train_root = RAW_ROOT / args.branch_id / "train"
    stop_event = None
    completed_update = start_update
    try:
        for update in range(start_update + 1, goal + 1):
            stage_info = bridge_stage_at_update(branch, update)
            if stage_info is not None:
                stage_index, stage = stage_info
                previous_end = 0 if stage_index == 0 else int(branch["stages"][stage_index - 1]["through_update"])
                if (
                    stage["kind"] == "bridge"
                    and update == previous_end + 1
                    and branch.get("stage_learnability_gate")
                ):
                    decision = run_stage_learnability_gate(
                        client,
                        branch,
                        stage_index,
                        stage,
                        source[str(stage_index)],
                        update,
                    )
                    if not decision["accepted"]:
                        stop_event = {
                            "type": "stage_learnability_gate_rejected",
                            "last_safe_update": last_safe_update,
                            "decision": decision,
                        }
                        break
            rows, phase = rows_for_update(
                branch,
                update,
                target,
                source,
                prompts_per_update,
            )
            rollout_path = train_root / f"update_{update:04d}.jsonl"
            rollout_started = time.perf_counter()
            scored = client.score_rows(
                rows,
                rollouts_per_prompt,
                stable_int(
                    "recipe_train",
                    branch["seed"],
                    branch.get("training_rng_key", branch.get("candidate_id")),
                    update,
                ),
                rollout_path,
                verification_case_sets=rollout_verification_cases(
                    branch,
                    update,
                    rows,
                    rollouts_per_prompt,
                ),
                sampling_seed_key=str(
                    branch.get(
                        "training_rng_key",
                        branch.get("candidate_id") or branch["branch_id"],
                    )
                ),
            )
            rollout_wall = time.perf_counter() - rollout_started
            binary_rewards = [int(item["reward"]) for item in scored]
            replay_ids = {str(item["instance_id"]) for item in success_replay}
            new_replay_items = 0
            for item in scored:
                instance_id = str(item["instance_id"])
                if int(item["reward"]) != 1 or instance_id in replay_ids:
                    continue
                if branch.get("success_replay_target_only") and instance_id not in target_ids:
                    continue
                row = rows[int(item["instance_index"])]
                success_replay.append(
                    {
                        "instance_id": instance_id,
                        "messages": row["messages"],
                        "completion_token_ids": [
                            int(value) for value in item["completion_token_ids"]
                        ],
                        "source_branch_id": args.branch_id,
                        "source_update": update,
                    }
                )
                replay_ids.add(instance_id)
                new_replay_items += 1
            partial_weight = annealed_partial_weight(branch, update)
            dense_reward_active = (
                (
                    branch["kind"] == "candidate_dense"
                    and update <= int(branch["candidate_updates"])
                )
                or (
                    branch["kind"] == "target_dense_then_strict"
                    and update <= int(branch["dense_updates"])
                )
                or partial_weight > 0.0
            )
            if branch["kind"] in {
                "annealed_target_continuation",
                "bridge_chain",
                "grounded_meta_candidate",
                "grounded_meta_direct",
            }:
                rewards = [
                    float(binary) + partial_weight * (
                        float(item["passed_cases"]) / max(1, int(item["total_cases"]))
                    )
                    for binary, item in zip(binary_rewards, scored)
                ]
                phase = f"{phase}_lexicographic_partial_weight_{partial_weight:g}"
            elif dense_reward_active:
                rewards = [
                    float(item["passed_cases"]) / max(1, int(item["total_cases"]))
                    for item in scored
                ]
            else:
                rewards = [float(value) for value in binary_rewards]
            advantages = group_advantages(rewards, rollouts_per_prompt)
            active = sum(abs(value) > 1e-12 for value in advantages)
            if active == 0:
                cumulative["zero_advantage_updates"] += 1

            cumulative["training_rollouts"] += len(scored)
            cumulative["training_generated_tokens"] += sum(
                int(item["completion_tokens"]) for item in scored
            )
            cumulative["rollout_wall_seconds"] += rollout_wall

            learner_started = time.perf_counter()
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            kl_values: list[float] = []
            for item, advantage in zip(scored, advantages):
                if abs(advantage) <= 1e-12:
                    continue
                row = rows[int(item["instance_index"])]
                prompt_ids = tokenizer(
                    apply_chat_template(tokenizer, row["messages"]),
                    add_special_tokens=False,
                ).input_ids
                loss, kl = recipe_loss(
                    model,
                    prompt_ids,
                    [int(value) for value in item["completion_token_ids"]],
                    advantage,
                    kl_beta,
                    reference_model,
                )
                (loss / max(1, active)).backward()
                losses.append(float(loss.detach().cpu()))
                kl_values.append(kl)
            replay_examples: list[dict[str, Any]] = []
            replay_loss_values: list[float] = []
            replay_cap = int(branch.get("success_replay_examples_per_update", 0))
            replay_weight = float(branch.get("success_replay_loss_weight", 0.0))
            if success_replay and replay_cap > 0 and replay_weight > 0.0:
                replay_rng = random.Random(
                    stable_int(
                        "success_replay", branch["seed"],
                        branch.get("randomness_branch_id", args.branch_id), update
                    )
                )
                replay_examples = replay_rng.sample(
                    success_replay,
                    min(replay_cap, len(success_replay)),
                )
                for replay_item in replay_examples:
                    prompt_ids = tokenizer(
                        apply_chat_template(tokenizer, replay_item["messages"]),
                        add_special_tokens=False,
                    ).input_ids
                    replay_loss, _ = recipe_loss(
                        model,
                        prompt_ids,
                        [int(value) for value in replay_item["completion_token_ids"]],
                        1.0,
                        0.0,
                    )
                    (
                        replay_loss
                        * replay_weight
                        / max(1, len(replay_examples))
                    ).backward()
                    replay_loss_values.append(float(replay_loss.detach().cpu()))
                cumulative["success_replay_examples_used"] += len(replay_examples)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["training"]["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            learner_wall = time.perf_counter() - learner_started
            cumulative["learner_wall_seconds"] += learner_wall

            _, sync_wall = sync_adapter(model, client, args.branch_id, update)
            cumulative["lora_sync_wall_seconds"] += sync_wall
            completed_update = update
            dev_summary = None
            if development is not None and update in science_updates:
                dev_summary = evaluate(
                    client,
                    branch,
                    update,
                    development,
                    int(config["evaluation"]["rollouts_per_instance"]),
                )
            if scope_probe is not None and update in science_updates:
                evaluate(
                    client,
                    branch,
                    update,
                    scope_probe,
                    int(branch.get("scope_probe_rollouts_per_instance", 8)),
                    split="scope_probe",
                )
            if grounded_reward is not None and update in science_updates:
                evaluate(
                    client,
                    branch,
                    update,
                    grounded_reward,
                    int(branch.get("grounded_reward_rollouts_per_instance", 4)),
                    split="grounded_reward",
                )

            online_scope_summary = None
            if (
                online_guard
                and scope_probe is not None
                and update % int(online_guard["interval_updates"]) == 0
            ):
                online_scope_summary = evaluate(
                    client,
                    branch,
                    update,
                    scope_probe,
                    int(branch.get("scope_probe_rollouts_per_instance", 8)),
                    split="online_scope_probe",
                )
                strict_drop = float(online_scope_summary["full_pass_rate"]) - float(
                    online_scope_reference["full_pass_rate"]
                )
                soft_drop = float(online_scope_summary["mean_case_fraction"]) - float(
                    online_scope_reference["mean_case_fraction"]
                )
                safe = (
                    strict_drop >= -float(online_guard["maximum_strict_drop"])
                    and soft_drop >= -float(online_guard["maximum_case_fraction_drop"])
                )
                guard_decision = {
                    "update": update,
                    "strict_gain_vs_start": strict_drop,
                    "case_fraction_gain_vs_start": soft_drop,
                    "safe": safe,
                }
                atomic_json(
                    RAW_ROOT
                    / branch["branch_id"]
                    / "online_scope_guard"
                    / f"update_{update:04d}.json",
                    guard_decision,
                )
                if safe:
                    last_safe_update = update
                else:
                    stop_event = {
                        "type": "online_scope_guard_rejected",
                        "last_safe_update": last_safe_update,
                        "decision": guard_decision,
                    }

            metric = {
                "branch_id": args.branch_id,
                "kind": branch["kind"],
                "candidate_id": branch.get("candidate_id"),
                "seed": int(branch["seed"]),
                "update": update,
                "phase": phase,
                "rollouts": len(scored),
                "successes": sum(binary_rewards),
                "reward_mean": sum(binary_rewards) / len(binary_rewards),
                "reward_variance": float(
                    torch.tensor(binary_rewards, dtype=torch.float32).var(unbiased=False)
                ),
                "optimization_reward_mean": sum(rewards) / len(rewards),
                "optimization_reward_variance": float(
                    torch.tensor(rewards, dtype=torch.float32).var(unbiased=False)
                ),
                "dense_reward_active": dense_reward_active,
                "partial_reward_weight": partial_weight,
                "active_advantages": active,
                "success_replay_pool_size": len(success_replay),
                "new_success_replay_items": new_replay_items,
                "success_replay_examples_update": len(replay_examples),
                "success_replay_loss": (
                    sum(replay_loss_values) / len(replay_loss_values)
                    if replay_loss_values
                    else 0.0
                ),
                "sigma_hat_zero_groups": sum(
                    len(set(rewards[start : start + rollouts_per_prompt])) == 1
                    for start in range(0, len(rewards), rollouts_per_prompt)
                ),
                "loss": sum(losses) / len(losses) if losses else 0.0,
                "kl": sum(kl_values) / len(kl_values) if kl_values else 0.0,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "rollout_wall_seconds_update": rollout_wall,
                "learner_wall_seconds_update": learner_wall,
                "lora_sync_wall_seconds_update": sync_wall,
                "development": dev_summary,
                "online_scope": online_scope_summary,
                **cumulative,
            }
            atomic_json(train_root / f"update_{update:04d}_summary.json", metric)
            if (
                update % int(config["training"]["resume_interval"]) == 0
                or update in science_updates
                or update == goal
                or (
                    online_guard
                    and update % int(online_guard["interval_updates"]) == 0
                )
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    args.branch_id,
                    update,
                    {"cumulative": cumulative, "success_replay": success_replay},
                )
                prune_checkpoints(args.branch_id, science_updates)
            print(json.dumps(metric), flush=True)
            if stop_event is not None:
                break

        atomic_json(
            RAW_ROOT / args.branch_id / "status.json",
            {
                "branch_id": args.branch_id,
                "kind": branch["kind"],
                "candidate_id": branch.get("candidate_id"),
                "completed_updates": completed_update,
                "fixed_budget": completed_update == goal,
                "optimizer_inherited_from_parent": bool(
                    branch.get("inherit_optimizer_state")
                ),
                "optimizer_parent_loaded_at_fresh_start": optimizer_parent_loaded,
                "parent_checkpoint": branch.get("parent_checkpoint"),
                "status": "complete" if completed_update == goal else str(stop_event["type"]),
                "stop_event": stop_event,
                "last_safe_update": last_safe_update,
                "kl_beta": kl_beta,
                "learning_rate": float(
                    optimizer_config["training"]["learning_rate"]
                ),
                **cumulative,
            },
        )
    finally:
        client.use_base()


if __name__ == "__main__":
    main()
