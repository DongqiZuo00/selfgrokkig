"""Advance the retained route checkpoint to the one-Painter bridge for four updates."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

from common import EXP_ROOT, atomic_json, read_json
from self_grok_decisive_v5_5_run import compact

VERSION = "self_grok_decisive_mistral_v5_6"
PREVIOUS = "self_grok_decisive_mistral_v5_5"
BRANCH = "decisive_mistral_v5_6__oracle"


def build_branch(prior, ladder, completed):
    assert completed["scope_safe"] and completed["candidate_eligible_for_continuation"]
    branch = copy.deepcopy(prior["branches"][0])
    parent = completed["retained_checkpoint"]
    assert parent == "checkpoints/decisive_mistral_v5_5__oracle/resume_u0004"
    stage = copy.deepcopy(ladder["branches"][1]["stages"][1])
    assert stage["name"] == "io_table_add_same_color_one_painter"
    assert stage["target_mix_fraction"] == 0.25 and stage["partial_weight"] == 0.0
    stage["through_update"] = 4
    branch.update(
        branch_id=BRANCH, candidate_id="v5_6_same_color_one_painter",
        parent_checkpoint=parent, parent_scientific_update=64,
        reference_checkpoint=parent, stages=[stage], goal_updates=4,
        grounded_reward_path=stage["training_data_path"],
        stage_learnability_gate={
            "instances": 8, "rollouts_per_instance": 8,
            "minimum_success_rate": 1 / 64,
            "minimum_mixed_groups": 1,
            "maximum_success_rate": 0.90, "mastered_success_rate": 0.90,
        },
        admission={
            "rule": "at least one mixed binary-reward group, or already mastered",
            "source": "fresh gate on the new one-Painter stage",
            "reuse_previous_stage_gate": False,
        },
    )
    branch["proposal"]["components"] = [copy.deepcopy(ladder["branches"][1]["proposal"]["components"][1])]
    branch["proposal"]["components"][0]["updates"] = 4
    assert branch["seed"] == 42 and branch["success_replay_loss_weight"] == 0.20
    assert not branch["inherit_optimizer_state"]
    return branch


def main():
    result_path = EXP_ROOT / f"raw_results/{VERSION}/decision.json"
    complete_path = EXP_ROOT / f"manifests/{VERSION}_complete.json"
    if complete_path.exists():
        print(json.dumps(read_json(result_path), indent=2))
        return
    branch = build_branch(
        read_json(EXP_ROOT / f"manifests/{PREVIOUS}_runtime_branches.json"),
        read_json(EXP_ROOT / "manifests/self_grok_decisive_mistral_v5_3_runtime_branches.json"),
        read_json(EXP_ROOT / f"manifests/{PREVIOUS}_complete.json"),
    )
    assert (EXP_ROOT / branch["parent_checkpoint"] / "adapter_config.json").exists()
    assert (EXP_ROOT / branch["grounded_reward_path"] / "dataset_info.json").exists()
    atomic_json(EXP_ROOT / f"manifests/{VERSION}_runtime_branches.json", {
        "protocol_version": VERSION, "branches": [branch], "branch_ids": [BRANCH],
        "planned_new_updates": 4, "retained_prior_updates": 28,
        "original_ladder_stage_index": 1,
        "exploratory": True, "matched_target_budget_claim": False,
        "official_test_opened": False,
    })
    root = EXP_ROOT / "raw_results/recipe_v1/branches" / BRANCH
    status_path = root / "status.json"
    # A scientific stop is a result, never an excuse to resample the gate.
    if not status_path.exists():
        subprocess.run([
            sys.executable, str(EXP_ROOT / "src/recipe_run_branch.py"),
            "--branch-id", BRANCH, "--port",
            str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000),
        ], check=True)
    status = read_json(status_path)
    updates = int(status["completed_updates"])
    assert 0 <= updates <= 4
    before = read_json(root / "grounded_reward/update_0000_summary.json")
    scope_before = read_json(root / "online_scope_probe/update_0000_summary.json")
    gate = read_json(root / "stage_gates/stage_00_before_update_0001.decision.json")
    training = []
    for update in range(1, updates + 1):
        metric = read_json(root / "train" / f"update_{update:04d}_summary.json")
        item = {key: metric[key] for key in (
            "update", "successes", "rollouts", "active_advantages",
            "sigma_hat_zero_groups", "success_replay_examples_update", "learning_rate",
        )}
        item["mixed_groups"] = metric["rollouts"] // 8 - metric["sigma_hat_zero_groups"]
        training.append(item)
    result = {
        "protocol_version": VERSION, "status": status, "gate": gate,
        "bridge_before": compact(before), "scope_before": compact(scope_before),
        "bridge_after": None, "scope_after": None, "scope_guard": None,
        "training": training, "new_updates": updates,
        "target_gamma_estimated": False, "official_test_opened": False,
    }
    safe = False
    if updates == 4:
        after = read_json(root / "grounded_reward/update_0004_summary.json")
        guard = read_json(root / "online_scope_guard/update_0004.json")
        result.update(
            bridge_after=compact(after),
            bridge_rate_change=after["full_pass_rate"] - before["full_pass_rate"],
            scope_after=compact(read_json(root / "online_scope_probe/update_0004_summary.json")),
            scope_guard=guard,
        )
        safe = bool(guard["safe"])
    retained = f"checkpoints/{BRANCH}/resume_u0004" if safe else branch["parent_checkpoint"]
    assert (EXP_ROOT / retained / "adapter_config.json").exists()
    result.update(candidate_eligible_for_continuation=safe,
                  retained_checkpoint=retained, retained_path_updates=32 if safe else 28)
    atomic_json(result_path, result)
    atomic_json(complete_path, {
        "status": "complete", "new_updates": updates, "gate_accepted": gate["accepted"],
        "candidate_eligible_for_continuation": safe,
        "scope_safe": safe if updates == 4 else None,
        "retained_checkpoint": retained, "official_test_opened": False,
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
