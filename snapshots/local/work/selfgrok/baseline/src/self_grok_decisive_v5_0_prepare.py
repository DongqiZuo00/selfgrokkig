from __future__ import annotations

import copy
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_0"


def finite_stage(template, name, updates, inputs, operation="prepend"):
    stage = copy.deepcopy(template)
    stage["name"] = name
    stage["updates"] = int(updates)
    stage["test_cases"] = "all"
    stage["prepend_curriculum_inputs"] = [str(value) for value in inputs]
    stage["prepend_curriculum_operation"] = str(operation)
    stage.pop("verification_case_counts", None)
    stage.pop("verification_subset_mode", None)
    stage.pop("verification_required_case_index", None)
    stage.pop("gate_seed_branch_id", None)
    stage.pop("gate_seed_stage_index", None)
    return stage


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    overlay = read_json(EXP_ROOT / "manifests" / f"{VERSION}_overlay.json")
    config = read_json(EXP_ROOT / str(overlay["base_manifest"]))
    config["protocol_version"] = VERSION
    config["purpose"] = (
        "Single-start, single-trajectory strict binary-reward test of a compositional "
        "pull/paint/replay skill ladder before the sparse official PREPEND target."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    updates = int(overlay["stage_updates"])
    template = config["oracle_ladder"][2]
    config["oracle_ladder"] = [
        finite_stage(
            template,
            "consume_opposite_single",
            updates,
            ["$opposite"],
            "consume",
        ),
        finite_stage(
            template,
            "replace_opposite_with_prefix",
            updates,
            ["$opposite"],
            "replace",
        ),
        finite_stage(
            template,
            "prepend_opposite_single",
            updates,
            ["$opposite"],
        ),
        finite_stage(
            template,
            "prepend_all_single",
            updates,
            ["R", "B"],
        ),
        finite_stage(
            template,
            "prepend_empty_and_all_single",
            updates,
            ["", "R", "B"],
        ),
        finite_stage(
            template,
            "prepend_all_inputs_upto_two",
            updates,
            ["", "R", "B", "RR", "RB", "BR", "BB"],
        ),
    ]
    config["solver"]["target_tail_updates"] = int(overlay["target_tail_updates"])
    config["solver"]["matched_updates"] = int(overlay["matched_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v5.0 matched budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    config["skip_root_stage_probes"] = bool(overlay["skip_root_stage_probes"])
    config["reused_direct_protocol"] = str(overlay["reused_direct_protocol"])
    config["reused_direct_branch_id"] = str(overlay["reused_direct_branch_id"])
    config["student_array_spec"] = "1"
    config["evaluation_rng_key"] = str(overlay["evaluation_rng_key"])
    config["primitive_ladder"] = {
        "explicit_finite_prompts": True,
        "hidden_verifier_slicing": False,
        "strict_binary_reward_per_task": True,
        "partial_condition_reward": False,
        "first_stage_minimal_program_fragment": "PULLER_RB -> END",
        "second_stage_adds": "one prefix PAINTER",
        "third_stage_adds": "one replay PAINTER",
        "target_rows_always_use_official_full_verifier": True,
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
