"""Isolated all-candidate target-only follow-on. Shared runtime route is gated.

No Challenger, curriculum filtering, adaptive rollback or inherited optimizer.
Source weights stay untouched in their original branch checkpoint directory.
"""
import gc
import json
import os
import time
import torch

from common import EXP_ROOT, read_json, atomic_json, stable_int, group_advantages
from recipe_train_branch import build_recipe_model, build_reference_model, cycle_rows
from fast_train_branch import sync_adapter
from verge_round_core import ROOT, VERSION, config, profile
from verge_round_runtime import rows, optimizer_for, commit, latest, endpoint, VLLMServerPool, VLLMRolloutClient
from verge_repair_sampling import atomic_jsonl
from verge_repair_training import phase_boundary_masks, train_step
from verge_repair_protocol import prompt_role
from verge_followon_protocol import validate_prepared
from verge_followon_events import first_training_event, final_event


def main():
    frozen = read_json(ROOT / "round_frozen.json")
    cfg = config()
    validate_prepared(cfg, frozen)
    stream = cfg.get("random_stream_id", VERSION)
    index = 0  # One isolated continuation per original candidate.
    branch = next(b for b in frozen["branches"] if b["index"] == index)
    output = ROOT / "branches" / str(index)
    if (output / "complete.json").exists():
        completed = read_json(output / "complete.json")
        from verge_followon_verify import validate_segment
        validate_segment(frozen, completed, output)
        print(json.dumps(completed["training"]))
        return
    saved = latest(branch["id"])
    source = saved or EXP_ROOT / cfg["solver_start"]
    target = rows(cfg["target_train"])
    datasets = [rows(s["path"]) for s in branch["stages"]]
    port = 20000 + (int(os.environ.get("SLURM_JOB_ID", "0")) + index * 991) % 25000
    with VLLMServerPool(gpus=(0,), base_port=port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        tokenizer, model = build_recipe_model(42, source)
        reference = build_reference_model(42, EXP_ROOT / cfg["solver_start"])
        optimizer, scheduler = optimizer_for(model, cfg["solver_learning_rate"])
        state = {"update": 0, "used_by_stage": [0] * len(datasets), "group_by_stage": [0] * len(datasets),
            "train_tokens": 0, "nonzero_advantage_tokens": 0, "optimizer_steps": 0,
            "generated_tokens": 0, "rollouts": 0, "target_successes": 0, "first_target_success": None,
            "target_loss_tokens": 0, "masked_phase_boundary_groups": 0, "started_at": time.time()}
        if saved:
            stored = torch.load(saved / "state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(stored["optimizer"])
            scheduler.load_state_dict(stored["scheduler"])
            state = stored["runtime_state"]
            del stored
        else:
            saved = commit(model, optimizer, scheduler, branch["id"], 0, state)
        sync_adapter(model, client, branch["id"], state["update"])
        for phase, stage in enumerate(branch["stages"]):
            while state["used_by_stage"][phase] < stage["tokens"]:
                if state["update"] >= cfg["maximum_updates_per_branch"]:
                    raise RuntimeError("100-update ceiling reached; no automatic extension")
                update = state["update"] + 1
                unit = output / "train" / f"unit_{update:04d}"
                items, advantages, masks = [], [], []
                used, masked_groups, role_counts = 0, 0, {"target": 0, "curriculum": 0}
                remaining = stage["tokens"] - state["used_by_stage"][phase]
                # No optimizer step or adapter sync while these groups are sampled.
                # Keep all tokens at update boundaries; only a phase boundary masks.
                while used < min(cfg["token_update_threshold"], remaining):
                    group_index = state["group_by_stage"][phase] + len(items) // 8
                    role = prompt_role(group_index, stage["kind"])
                    dataset = target if role == "target" else datasets[phase]
                    row = cycle_rows(dataset, group_index + 1, 1, 42, f"{stream}_phase{phase}_{role}")[0]
                    batch = client.score_rows([row], 8, stable_int(stream, 42, phase, group_index),
                        unit / f"group_{group_index:06d}.jsonl",
                        sampling_seed_key=f"{stream}_phase{phase}_group{group_index}")
                    rewards = [int(r["reward"]) for r in batch]
                    if len(batch) != 8 or not set(rewards) <= {0, 1}:
                        raise RuntimeError("Invalid binary reward group")
                    control_advantages = group_advantages(rewards, 8)
                    allocation = phase_boundary_masks([len(r["completion_token_ids"]) for r in batch],
                        remaining - used, stable_int(stream, 42, phase, group_index, "boundary"))
                    masked_groups += int(sum(map(sum, allocation)) < sum(r["completion_tokens"] for r in batch))
                    for item in batch:
                        item["prompt_role"] = role
                        item["reward_group"] = group_index
                    items.extend(batch)
                    advantages.extend(control_advantages)
                    masks.extend(allocation)
                    role_counts[role] += 1
                    used += sum(map(sum, allocation))
                atomic_jsonl(output / "train" / f"update_{update:04d}.jsonl", items)
                allocation_path = output / "train" / f"update_{update:04d}_allocation.json"
                atomic_json(allocation_path, {"advantages": advantages, "loss_masks": masks,
                    "policy_checkpoint_update": state["update"], "stage": phase,
                    "fixed_completion_cap": 2048, "complete_reward_groups": len(items) // 8,
                    "advantage_source": "binary_complete_target_reward", "reward_mode": stage["reward_mode"]})
                metric = train_step(model, reference, optimizer, scheduler, items, advantages, masks, cfg,
                                    state["optimizer_steps"])
                target_successes = sum(r["reward"] for r in items if r["prompt_role"] == "target")
                state["first_target_success"] = first_training_event(state, items, masks, update)
                state["target_successes"] += target_successes
                state["used_by_stage"][phase] += used
                state["group_by_stage"][phase] += len(items) // 8
                state["train_tokens"] += used
                state["nonzero_advantage_tokens"] += metric["nonzero_advantage_tokens"]
                state["optimizer_steps"] += int(metric["optimizer_step"])
                state["generated_tokens"] += sum(r["completion_tokens"] for r in items)
                state["rollouts"] += len(items)
                state["target_loss_tokens"] += sum(sum(mask) for r, mask in zip(items, masks) if r["prompt_role"] == "target")
                state["masked_phase_boundary_groups"] += masked_groups
                state["update"] = update
                metric.update(update=update, stage=phase, stage_kind=stage["kind"],
                    successes=sum(r["reward"] for r in items), target_successes=target_successes,
                    mixed_groups=sum(len({r["reward"] for r in items[i:i+8]}) > 1 for i in range(0, len(items), 8)),
                    prompt_group_counts=role_counts, masked_phase_boundary_groups=masked_groups,
                    reward_mode=stage["reward_mode"],
                    optimization_reward_mean=sum(r["reward"] for r in items) / len(items),
                    optimization_mixed_groups=sum(len({r["reward"] for r in items[i:i+8]}) > 1
                                                  for i in range(0, len(items), 8)),
                    cumulative=dict(state))
                atomic_json(output / "train" / f"update_{update:04d}_summary.json", metric)
                saved = commit(model, optimizer, scheduler, branch["id"], update, state)
                # Include the metric in a recoverable checkpoint sidecar as well.
                atomic_json(saved / "verge_update_summary.json", metric)
                sync_adapter(model, client, branch["id"], update)
                print(json.dumps(metric), flush=True)
        if state["train_tokens"] != cfg["train_tokens_per_branch"]:
            raise RuntimeError("Unmatched branch token budgets")
        if state["masked_phase_boundary_groups"] > len(branch["stages"]):
            raise RuntimeError("Loss masking occurred outside a phase boundary")
        final_path = str(saved.relative_to(EXP_ROOT))
        del model, reference, optimizer, scheduler, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        result = endpoint(client, output, checkpoint=final_path,
            selection=rows(cfg["target_selection"])[:cfg["endpoint_instances"]],
            scope=rows(cfg["scope_dataset"])[:cfg["scope_instances"]])
        result.update(training=state, binary_rewards_only=all(s["reward_mode"] == "binary" for s in branch["stages"]),
            followon_only=True, source_candidate=cfg["followon_source"],
            followon_comparison_complete=False, fresh_optimizer=True, replay_loss=0.0,
            weight_decay=0.0, kl_reference=cfg["solver_start"], acceptance_only=cfg["acceptance_only"],
            native_token_accounting=True, fixed_completion_cap=2048)
        result["scope_safe_vs_start"] = all(result["scope"][k] >= frozen["initial"]["scope"][k]
            for k in ("full_pass_rate", "mean_case_fraction"))
        completion_rows = rows(cfg["target_completion"])[:cfg["completion_instances"]]
        scored = client.score_rows(completion_rows, cfg["completion_samples"],
            stable_int(stream,42,"completion"), output / "completion.jsonl",
            sampling_seed_key=f"{stream}_completion")
        result["completion"] = profile(scored, completion_rows)
        atomic_json(output / "completion_profile.json", result["completion"])
        result["followon_event"] = final_event(cfg["followon_source"], state, result["target"], result["completion"])
        from verge_followon_verify import validate_segment
        result.update(validate_segment(frozen, result, output))
        atomic_json(output / "complete.json", result)
        client.use_base()


if __name__ == "__main__":
    main()
