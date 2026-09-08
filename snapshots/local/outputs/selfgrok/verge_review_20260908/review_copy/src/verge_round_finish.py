from __future__ import annotations

import json
import os
import time

import torch
from common import EXP_ROOT, atomic_json, read_json, read_jsonl, apply_chat_template, stable_int
from recipe_train_branch import recipe_loss
from train_branch import build_model
from verge_round_runtime import VLLMServerPool, VLLMRolloutClient
from verge_round_core import ROOT, VERSION, profile
from verge_round_runtime import rows, optimizer_for, commit, free_models
from verge_round_decision import decide


def update_teacher(frozen, decision):
    if frozen["config"].get("repair_integrated"):
        from verge_repair_teacher import update_teacher as repaired_update
        return repaired_update(frozen, decision)
    output = ROOT / "challenger_update.json"
    if output.exists():
        return read_json(output)
    branch = VERSION + "_challenger"
    destination = EXP_ROOT / "checkpoints" / branch / "resume_u0001"
    advantages = decision["challenger_advantages"]
    updated = any(abs(a) > 1e-12 for a in advantages)
    losses = []
    if updated and not (destination / "verge_committed.json").exists():
        tokenizer, model = build_model(42, EXP_ROOT / frozen["teacher_checkpoint"])
        optimizer, scheduler = optimizer_for(model, frozen["config"]["challenger_learning_rate"])
        prompt_ids = tokenizer(apply_chat_template(tokenizer, frozen["generation"]["messages"]),
                               add_special_tokens=False).input_ids
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for candidate, advantage in zip(frozen["generation"]["candidates"], advantages):
            if abs(advantage) < 1e-12 or not candidate["completion_token_ids"]:
                continue
            loss, _ = recipe_loss(model, prompt_ids, candidate["completion_token_ids"], advantage, 0.0)
            (loss / len(advantages)).backward()
            losses.append(float(loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        commit(model, optimizer, scheduler, branch, 1, {"round": 1, "rewards": decision["challenger_rewards"],
            "advantages": advantages, "losses": losses, "whole_proposal_update": True})
        del tokenizer, model, optimizer, scheduler
        free_models()
    result = {"updated": updated, "optimizer_steps": int(updated),
              "checkpoint": str(destination.relative_to(EXP_ROOT)) if updated else frozen["teacher_checkpoint"],
              "whole_proposal_advantages": advantages, "rewards": decision["challenger_rewards"],
              "losses": losses, "algorithm": "one on-policy group-centered policy-gradient step; no second recentering",
              "zero_variance_convention": "zero advantages, retain Challenger", "teacher_kl_beta": 0.0}
    atomic_json(output, result)
    return result


def compute_ledger(frozen, branches):
    paths = [ROOT / "initial/target.jsonl", ROOT / "initial/scope.jsonl", ROOT / "completion.jsonl"]
    for p in frozen["probes"]:
        paths.append(ROOT / "probes" / f"c{p['candidate']}_s{p['stage']}.jsonl")
    for b in frozen["branches"]:
        root = ROOT / "branches" / str(b["index"])
        paths += [root / "target.jsonl", root / "scope.jsonl"]
        paths += [root / "train" / f"update_{i:04d}.jsonl"
                  for i in range(1, branches[b["index"]]["training"]["update"] + 1)]
    tokens, rollouts, verification_calls = 0, 0, 0
    for path in paths:
        data = read_jsonl(path)
        tokens += sum(r["completion_tokens"] for r in data)
        rollouts += len(data)
        verification_calls += sum(r["total_cases"] for r in data)
    proposal_tokens = sum(len(p["completion_token_ids"]) for p in frozen["generation"]["candidates"])
    training = sum(b["training"]["train_tokens"] for b in branches.values())
    return {"solver_training_tokens_including_zero_advantage": training,
            "solver_nonzero_advantage_tokens": sum(b["training"]["nonzero_advantage_tokens"] for b in branches.values()),
            "solver_generated_completion_tokens_all_phases": tokens, "solver_rollouts": rollouts,
            "challenger_generated_tokens": proposal_tokens,
            "non_training_generation_overhead_tokens": tokens - training + proposal_tokens,
            "binary_verifier_test_calls": verification_calls,
            "condition_reexecution_note": "Profiles reexecute target test suites separately; these are excluded from binary_verifier_test_calls",
            "hardware_hours": "Use Slurm accounting for completed prep, array elements and finish; not inferred from token count",
            "official_test_opened": False}


def report(frozen, branches, decision, teacher, completion, ledger):
    initial = frozen["initial"]["target"]
    rung = decision["reward_rung_zero_based"]
    lines = ["# VERGE: one exploratory Mistral outer round", "",
        "Human-curriculum warm start; one trajectory; no claim of autonomous discovery from base or generalization.", "",
        f"Target: {frozen['config']['target_descriptor']}",
        f"Starting target full pass: {initial['counts'][-1]}/{initial['rollouts']}.",
        f"Credit condition: {initial['condition_names'][rung]}; all credit uses fresh endpoint rollouts.", "",
        "| Branch | Updates | Training tokens | Nonzero-adv tokens | Endpoint full pass | Credit condition | Scope full pass |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    if frozen["config"].get("acceptance_only"):
        lines[0] = "# VERGE engineering acceptance — NOT a scientific experiment result"
        lines.insert(2, "Reduced training and evaluation budgets validate execution only. No claim of target transfer or learned bridge discovery.\n")
    if frozen["config"].get("book_suite"):
        cfg = frozen["config"]
        lines[0] = f"# VERGE experiment book: {cfg['book_arm']}, round {cfg['book_round'] + 1}/6"
        lines[2] = ("Single-seed protocol adaptation. Fresh backbone initialization at round zero; "
                    "subsequent rounds inherit their own arm's committed Solver and Challenger. "
                    "This round is not the completed experiment book.")
    for index, branch in branches.items():
        train, target = branch["training"], branch["target"]
        lines.append(f"| {'direct' if index == 0 else 'candidate_' + str(index)} | {train['update']} | {train['train_tokens']} | "
            f"{train['nonzero_advantage_tokens']} | {target['counts'][-1]}/{target['rollouts']} | "
            f"{target['rates'][rung]:.6f} | {branch['scope']['full_pass_rate']:.6f} |")
    lines += ["", f"Challenger rewards: {decision['challenger_rewards']}.",
        f"Challenger update performed: {teacher['updated']}. This is not yet evidence that its next proposals improve.",
        f"Retained Solver: {decision['selected']['name']} ({decision['selected']['checkpoint']}).",
        f"Fresh post-selection completion check: {completion['counts'][-1]}/{completion['rollouts']} full passes.",
        "The completion check was not used for selection or Challenger credit.", "",
        "## Limits", "",
        "One fixed target definition with multiple generator instances and draws, not 64 distinct task definitions. "
        "Condition gain is not full-pass gain. Paired intervals are exploratory and unadjusted. "
        "Scope preservation is an empirical tolerance check, not a guarantee. No official test was opened.", "",
        "## Generation and training accounting", "", "```json", json.dumps(ledger, indent=2), "```", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    manifest = EXP_ROOT / "manifests" / f"{VERSION}_complete.json"
    if manifest.exists():
        print(json.dumps(read_json(manifest)))
        return
    frozen = read_json(ROOT / "round_frozen.json")
    cfg = frozen["config"]
    branches = {b["index"]: read_json(ROOT / "branches" / str(b["index"]) / "complete.json")
                for b in frozen["branches"]}
    decision_path = ROOT / "decision.json"
    decision = read_json(decision_path) if decision_path.exists() else decide(frozen, branches)
    atomic_json(decision_path, decision)
    print(json.dumps(decision), flush=True)
    teacher = update_teacher(frozen, decision)
    completion_path = ROOT / "completion_profile.json"
    if completion_path.exists():
        completion = read_json(completion_path)
    else:
        port = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000
        with VLLMServerPool(gpus=(0,), base_port=port) as urls:
            client = VLLMRolloutClient(urls[0], 0)
            client.load_lora(VERSION + "_retained", EXP_ROOT / decision["selected"]["checkpoint"])
            completion_rows = rows(cfg["target_completion"])[:cfg.get("completion_instances", 64)]
            stream = cfg.get("random_stream_id", VERSION)
            scored = client.score_rows(completion_rows, 1, stable_int(stream, 42, "completion"),
                                      ROOT / "completion.jsonl", sampling_seed_key=stream + "_completion")
            completion = profile(scored, completion_rows)
            atomic_json(completion_path, completion)
            client.use_base()
    ledger = compute_ledger(frozen, branches)
    atomic_json(ROOT / "compute_ledger.json", ledger)
    atomic_json(ROOT / "archive.json", {"round": cfg.get("book_round", 0) + 1, "proposals": frozen["generation"], "probes": frozen["probes"],
        "initial": frozen["initial"], "branch_endpoints": branches, "decision": decision,
        "challenger_update": teacher, "completion": completion})
    report(frozen, branches, decision, teacher, completion, ledger)
    requirements = ["round_frozen.json", "decision.json", "challenger_update.json", "completion_profile.json",
                    "compute_ledger.json", "archive.json", "RESULTS.md"]
    assert all((ROOT / name).is_file() and (ROOT / name).stat().st_size > 0 for name in requirements)
    if cfg.get("repair_integrated"):
        from verify_verge_repair import validate
        validate(frozen, branches, decision, teacher, completion)
        from verge_repair_exports import export
        requirements += export(ROOT, frozen, branches, decision, teacher, completion)
        if cfg.get("acceptance_only"):
            from verge_repair_gradient_acceptance import check
            check(frozen)
    assert all((ROOT / name).is_file() and (ROOT / name).stat().st_size > 0 for name in requirements)
    complete_record = {"rounds_completed": 1, "completed_at": time.time(),
        "solver_checkpoint": decision["selected"]["checkpoint"], "challenger_checkpoint": teacher["checkpoint"],
        "scope_safe_continuation": True, "artifacts": [str((ROOT / name).relative_to(EXP_ROOT)) for name in requirements],
        "official_test_opened": False, "acceptance_only": cfg.get("acceptance_only", False),
        "claim_limit": "Engineering acceptance only, not a scientific result" if cfg.get("acceptance_only") else
                       "One warm-start outer round, not the completed paper experiment"}
    if cfg.get("book_suite"):
        complete_record.update(book_suite=cfg["book_suite"], book_arm=cfg["book_arm"], book_round=cfg["book_round"],
            root_provenance=cfg["root_provenance"], parent_versions=cfg["prior_book_versions"],
            claim_limit="One atomic round of a single-seed manuscript comparison; full matrix not yet complete")
    atomic_json(manifest, complete_record)
    print((ROOT / "RESULTS.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
