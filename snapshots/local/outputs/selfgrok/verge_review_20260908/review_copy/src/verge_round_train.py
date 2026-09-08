from __future__ import annotations

import gc
import json
import os
import random
import time

import torch
from common import EXP_ROOT, atomic_json, read_json, apply_chat_template, group_advantages, stable_int
from recipe_train_branch import build_recipe_model, build_reference_model, recipe_loss, cycle_rows
from fast_train_branch import sync_adapter
from verge_round_runtime import VLLMServerPool, VLLMRolloutClient
from verge_round_core import ROOT, VERSION, allocate_tokens
from verge_round_runtime import rows, optimizer_for, commit, latest, endpoint


def main():
    frozen = read_json(ROOT / "round_frozen.json")
    cfg = frozen["config"]
    if cfg.get("repair_integrated"):
        from verge_repair_train import main as repaired_main
        return repaired_main()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    branch = next(b for b in frozen["branches"] if b["index"] == index)
    output = ROOT / "branches" / str(index)
    if (output / "complete.json").exists():
        print(json.dumps(read_json(output / "complete.json")))
        return
    saved = latest(branch["id"])
    source_checkpoint = saved or EXP_ROOT / cfg["solver_start"]
    port = 20000 + (int(os.environ.get("SLURM_JOB_ID", "0")) + index * 991) % 25000
    target = rows(cfg["target_train"])
    target_ids = {r["id"] for r in target}
    datasets = [rows(stage["path"]) for stage in branch["stages"]]
    with VLLMServerPool(gpus=(0,), base_port=port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        tokenizer, model = build_recipe_model(42, source_checkpoint)
        reference = build_reference_model(42, EXP_ROOT / cfg["solver_start"])
        optimizer, scheduler = optimizer_for(model, cfg["solver_learning_rate"])
        state = {"update": 0, "used_by_stage": [0] * len(branch["stages"]),
                 "train_tokens": 0, "nonzero_advantage_tokens": 0, "optimizer_steps": 0,
                 "generated_tokens": 0, "rollouts": 0, "target_successes": 0,
                 "first_target_success": None, "started_at": time.time()}
        if saved:
            stored = torch.load(saved / "state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(stored["optimizer"])
            scheduler.load_state_dict(stored["scheduler"])
            state = stored["runtime_state"]
            del stored
        sync_adapter(model, client, branch["id"], state["update"])
        for phase, stage in enumerate(branch["stages"]):
            while state["used_by_stage"][phase] < stage["tokens"]:
                if state["update"] >= cfg["maximum_updates_per_branch"]:
                    raise RuntimeError("Prespecified update ceiling reached before token budget; no extension")
                update = state["update"] + 1
                count = 8 if stage["kind"] == "target" else 6
                batch = cycle_rows(datasets[phase], update, count, 42, f"{VERSION}_phase{phase}")
                if stage["kind"] == "curriculum":
                    batch += cycle_rows(target, update, 2, 42, VERSION + "_target_mix")
                    random.Random(stable_int(VERSION, 42, update, "shuffle")).shuffle(batch)
                scored = client.score_rows(batch, 8, stable_int(VERSION, 42, phase, update),
                    output / "train" / f"update_{update:04d}.jsonl",
                    sampling_seed_key=f"{VERSION}_phase{phase}_update{update}")
                rewards = [int(r["reward"]) for r in scored]
                assert len(scored) == 64 and set(rewards) <= {0, 1}
                advantages = group_advantages(rewards, 8)
                quota = allocate_tokens([len(r["completion_token_ids"]) for r in scored],
                                        stage["tokens"] - state["used_by_stage"][phase])
                used = sum(quota)
                if used == 0:
                    raise RuntimeError("Empty completion-token batch; no budget progress")
                optimizer.zero_grad(set_to_none=True)
                model.train()
                active_tokens = sum(n for n, a in zip(quota, advantages) if abs(a) > 1e-12)
                # Exact all-zero stationary direct branches need no redundant model passes.
                do_update = active_tokens > 0 or state["optimizer_steps"] > 0
                losses = []
                if do_update:
                    for item, n, advantage in zip(scored, quota, advantages):
                        if n == 0:
                            continue
                        prompt_ids = tokenizer(apply_chat_template(tokenizer, batch[item["instance_index"]]["messages"]),
                                               add_special_tokens=False).input_ids
                        loss, _ = recipe_loss(model, prompt_ids, item["completion_token_ids"][:n],
                                              advantage, cfg["kl_beta"], reference)
                        (loss * n / used).backward()
                        losses.append(float(loss.detach().cpu()))
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    scheduler.step()
                    state["optimizer_steps"] += 1
                target_successes = sum(r["reward"] for r in scored if r["instance_id"] in target_ids)
                if target_successes and state["first_target_success"] is None:
                    state["first_target_success"] = {"update": update, "phase": phase,
                                                     "generated_tokens_before": state["generated_tokens"]}
                state["target_successes"] += target_successes
                state["used_by_stage"][phase] += used
                state["train_tokens"] += used
                state["nonzero_advantage_tokens"] += active_tokens
                state["generated_tokens"] += sum(len(r["completion_token_ids"]) for r in scored)
                state["rollouts"] += len(scored)
                state["update"] = update
                metric = {"update": update, "stage": phase, "stage_kind": stage["kind"],
                          "successes": sum(rewards), "target_successes": target_successes,
                          "mixed_groups": sum(len(set(rewards[i:i+8])) > 1 for i in range(0, 64, 8)),
                          "tokens_used": used, "nonzero_advantage_tokens": active_tokens,
                          "loss": sum(losses) / max(1, len(losses)), "optimizer_step": do_update,
                          "cumulative": dict(state)}
                atomic_json(output / "train" / f"update_{update:04d}_summary.json", metric)
                saved = commit(model, optimizer, scheduler, branch["id"], update, state)
                sync_adapter(model, client, branch["id"], update)
                print(json.dumps(metric), flush=True)
        assert state["train_tokens"] == cfg["train_tokens_per_branch"]
        final_path = str(saved.relative_to(EXP_ROOT))
        del model, reference, optimizer, scheduler, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        result = endpoint(client, output, checkpoint=final_path, selection=rows(cfg["target_selection"]),
                          scope=rows(cfg["scope_dataset"])[:cfg["scope_instances"]])
        result["training"] = state
        result["binary_rewards_only"] = True
        result["fresh_optimizer"] = True
        result["replay_loss"] = 0.0
        result["weight_decay"] = 0.0
        result["kl_reference"] = cfg["solver_start"]
        result["scope_safe_vs_start"] = (
            result["scope"]["full_pass_rate"] >= frozen["initial"]["scope"]["full_pass_rate"] - cfg["scope_max_drop"]
            and result["scope"]["mean_case_fraction"] >= frozen["initial"]["scope"]["mean_case_fraction"] - cfg["scope_max_drop"])
        atomic_json(output / "complete.json", result)
        client.use_base()
        print(json.dumps({"index": index, "tokens": state["train_tokens"], "updates": state["update"],
                          "target_rates": result["target"]["rates"],
                          "scope_safe": result["scope_safe_vs_start"]}, indent=2))


if __name__ == "__main__":
    main()
