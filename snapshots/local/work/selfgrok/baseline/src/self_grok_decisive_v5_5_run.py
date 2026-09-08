"""A single four-update retry with stronger existing success replay."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

from common import EXP_ROOT, atomic_json, read_json

VERSION = "self_grok_decisive_mistral_v5_5"
BRANCH = "decisive_mistral_v5_5__oracle"
PREVIOUS = "self_grok_decisive_mistral_v5_4"


def build_branch(prior):
    source = prior["branches"][0]
    assert source["goal_updates"] == 4
    assert source["success_replay_loss_weight"] == 0.05
    assert source["success_replay_examples_per_update"] == 1
    assert source["parent_checkpoint"] == "checkpoints/decisive_mistral_v5_2__oracle/resume_u0004"
    branch = copy.deepcopy(source)
    branch.update(
        branch_id=BRANCH,
        candidate_id="v5_5_stronger_replay_four_update_probe",
        randomness_branch_id=source.get("randomness_branch_id", source["branch_id"]),
        success_replay_loss_weight=0.20,
    )
    # Branch names organize artifacts; reuse v5.4's shuffle and replay RNG stream.
    allowed = {"branch_id", "candidate_id", "randomness_branch_id", "success_replay_loss_weight"}
    assert all(branch[key] == value for key, value in source.items() if key not in allowed)
    assert branch["seed"] == 42 and not branch["inherit_optimizer_state"]
    return branch


def compact(summary):
    return {key: value for key, value in summary.items() if key != "instance_success_rates"}


def main():
    complete_path = EXP_ROOT / f"manifests/{VERSION}_complete.json"
    result_path = EXP_ROOT / f"raw_results/{VERSION}/decision.json"
    if complete_path.exists():
        print(json.dumps(read_json(result_path), indent=2))
        return
    prior = read_json(EXP_ROOT / f"manifests/{PREVIOUS}_runtime_branches.json")
    branch = build_branch(prior)
    if not (EXP_ROOT / branch["parent_checkpoint"] / "adapter_config.json").exists():
        raise RuntimeError("The prescribed safe starting adapter is missing")
    atomic_json(EXP_ROOT / f"manifests/{VERSION}_runtime_branches.json", {
        "protocol_version": VERSION, "branches": [branch], "branch_ids": [BRANCH],
        "planned_new_updates": 4, "retained_prior_updates": 24,
        "single_change": {"success_replay_loss_weight": {"before": 0.05, "after": 0.20}},
        "baseline": PREVIOUS, "training_and_evaluation_rng_aligned": True,
        "exploratory": True, "matched_target_budget_claim": False,
        "official_test_opened": False,
    })
    root = EXP_ROOT / "raw_results/recipe_v1/branches" / BRANCH
    status_path = root / "status.json"
    if not status_path.exists() or int(read_json(status_path)["completed_updates"]) < 4:
        subprocess.run([
            sys.executable, str(EXP_ROOT / "src/recipe_run_branch.py"),
            "--branch-id", BRANCH, "--port",
            str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000),
        ], check=True)
    status = read_json(status_path)
    if int(status["completed_updates"]) != 4:
        raise RuntimeError("Four-update probe did not finish; preserve logs and recover execution")
    before = read_json(root / "grounded_reward/update_0000_summary.json")
    after = read_json(root / "grounded_reward/update_0004_summary.json")
    scope_before = read_json(root / "online_scope_probe/update_0000_summary.json")
    scope_after = read_json(root / "online_scope_probe/update_0004_summary.json")
    guard = read_json(root / "online_scope_guard/update_0004.json")
    assert before["rollouts"] == after["rollouts"] == 64
    assert scope_before["rollouts"] == scope_after["rollouts"] == 128
    training = []
    for update in range(1, 5):
        metric = read_json(root / "train" / f"update_{update:04d}_summary.json")
        training.append({
            key: metric[key] for key in (
                "update", "successes", "rollouts", "active_advantages",
                "sigma_hat_zero_groups", "success_replay_examples_update",
                "success_replay_loss", "learning_rate",
            )
        })
        training[-1]["mixed_groups"] = (
            metric["rollouts"] // branch["rollouts_per_prompt"]
            - metric["sigma_hat_zero_groups"]
        )
    baseline = read_json(EXP_ROOT / f"raw_results/{PREVIOUS}/decision.json")
    safe = bool(guard["safe"])
    retained = f"checkpoints/{BRANCH}/resume_u0004" if safe else branch["parent_checkpoint"]
    if not (EXP_ROOT / retained / "adapter_config.json").exists():
        raise RuntimeError("Selected retained checkpoint is missing")
    result = {
        "protocol_version": VERSION, "status": status,
        "single_change": {"replay_weight_before": 0.05, "replay_weight_after": 0.20},
        "bridge_before": compact(before), "bridge_after": compact(after),
        "bridge_rate_change": after["full_pass_rate"] - before["full_pass_rate"],
        "scope_before": compact(scope_before), "scope_after": compact(scope_after),
        "scope_guard": guard, "training": training,
        "baseline_v5_4": {
            "bridge_before": baseline["bridge_before"]["full_pass_count"],
            "bridge_after": baseline["bridge_after"]["full_pass_count"],
            "scope_strict_change": baseline["scope_guard"]["strict_gain_vs_start"],
        },
        "starting_bridge_counts_reproduced": before["full_pass_count"] == baseline["bridge_before"]["full_pass_count"],
        "candidate_eligible_for_continuation": safe,
        "retained_checkpoint": retained,
        "new_updates": 4, "retained_path_updates": 28 if safe else 24,
        "target_gamma_estimated": False, "official_test_opened": False,
    }
    atomic_json(result_path, result)
    atomic_json(complete_path, {
        "status": "complete", "new_updates": 4, "scope_safe": safe,
        "candidate_eligible_for_continuation": safe,
        "retained_checkpoint": retained, "official_test_opened": False,
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
