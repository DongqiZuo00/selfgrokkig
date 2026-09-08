"""Four-update probe of a previously observed non-degenerate bridge group."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

from common import EXP_ROOT, atomic_json, read_json, read_jsonl

VERSION = "self_grok_decisive_mistral_v5_4"
BRANCH = "decisive_mistral_v5_4__oracle"


def main():
    prior = read_json(EXP_ROOT / "manifests/self_grok_decisive_mistral_v5_3_runtime_branches.json")
    branch = copy.deepcopy(prior["branches"][1])
    previous = read_json(EXP_ROOT / "raw_results/self_grok_decisive_mistral_v5_3/decision.json")
    evidence = previous["oracle_training"]["stop_event"]["decision"]
    assert previous["oracle_training"]["completed_updates"] == 0
    assert previous["single_start"]["checkpoint"] == branch["parent_checkpoint"]
    assert evidence["stage"]["training_data_path"] == branch["stages"][0]["training_data_path"]
    assert evidence["successes"] > 0 and evidence["mixed_groups"] >= 1
    branch.update(branch_id=BRANCH, candidate_id="v5_4_four_update_route_probe",
                  goal_updates=4, evaluation_updates=[4],
                  evaluate_grounded_at_start=True,
                  grounded_reward_path=branch["stages"][0]["training_data_path"],
                  grounded_reward_instances=8, grounded_reward_rollouts_per_instance=8,
                  evaluation_rng_key="v5_4_fixed_checkpoint_probe",
                  stage_learnability_gate=None)
    branch["stages"] = branch["stages"][:1]
    branch["stages"][0]["through_update"] = 4
    branch["admission"] = {
        "rule": "at least one observed mixed binary-reward group",
        "source": "raw_results/self_grok_decisive_mistral_v5_3/decision.json",
        "observed_successes": evidence["successes"],
        "observed_rollouts": evidence["rollouts"],
        "observed_mixed_groups": evidence["mixed_groups"],
        "accepted": True,
        "reuse_existing_gate_without_resampling": True,
    }
    atomic_json(EXP_ROOT / f"manifests/{VERSION}_runtime_branches.json", {
        "protocol_version": VERSION, "branches": [branch], "branch_ids": [BRANCH],
        "planned_new_updates": 4, "retained_prior_updates": 24,
        "scope": "exploratory bridge reinforcement, no matched target Gamma claim",
        "official_test_opened": False,
    })
    subprocess.run([sys.executable, str(EXP_ROOT / "src/recipe_run_branch.py"),
                    "--branch-id", BRANCH, "--port",
                    str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000)], check=True)
    root = EXP_ROOT / "raw_results/recipe_v1/branches" / BRANCH
    status = read_json(root / "status.json")
    before = read_json(root / "grounded_reward/update_0000_summary.json")
    after = read_json(root / "grounded_reward/update_0004_summary.json")
    guard = read_json(root / "online_scope_guard/update_0004.json")
    training = []
    for update in range(1, 5):
        rows = read_jsonl(root / "train" / f"update_{update:04d}.jsonl")
        groups = {}
        for row in rows:
            groups.setdefault(row["instance_index"], []).append(int(row["reward"]))
        training.append({"update": update, "successes": sum(int(r["reward"]) for r in rows),
                         "rollouts": len(rows),
                         "mixed_groups": sum(len(set(v)) > 1 for v in groups.values())})
    result = {"protocol_version": VERSION, "status": status,
              "admission": branch["admission"], "bridge_before": before,
              "bridge_after": after,
              "bridge_rate_change": after["full_pass_rate"] - before["full_pass_rate"],
              "scope_guard": guard,
              "scope_after": read_json(root / "online_scope_probe/update_0004_summary.json"),
              "training": training, "retained_prior_updates": 24,
              "new_updates": status["completed_updates"],
              "target_gamma_estimated": False, "official_test_opened": False}
    atomic_json(EXP_ROOT / f"raw_results/{VERSION}/decision.json", result)
    atomic_json(EXP_ROOT / f"manifests/{VERSION}_complete.json", {
        "status": "complete", "new_updates": status["completed_updates"],
        "scope_safe": guard["safe"], "official_test_opened": False})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
