from __future__ import annotations

import copy
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_1"


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
        "Single-start, single-trajectory strict binary-reward continuation that "
        "isolates the remaining second-Painter/replay discontinuity."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    template = config["oracle_ladder"][2]
    config["oracle_ladder"] = [
        finite_stage(
            template,
            "replace_opposite_with_double_prefix",
            6,
            ["$opposite"],
            "replace_repeat",
        ),
        finite_stage(
            template,
            "prepend_opposite_single",
            6,
            ["$opposite"],
        ),
        finite_stage(
            template,
            "prepend_all_single",
            6,
            ["R", "B"],
        ),
        finite_stage(
            template,
            "prepend_empty_and_all_single",
            6,
            ["", "R", "B"],
        ),
        finite_stage(
            template,
            "prepend_all_inputs_upto_two",
            8,
            ["", "R", "B", "RR", "RB", "BR", "BB"],
        ),
    ]
    config["solver"]["target_tail_updates"] = 6
    config["solver"]["matched_updates"] = int(overlay["additional_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v5.1 additional budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    if int(overlay["prior_bridge_updates"]) + expected != int(
        overlay["direct_completed_updates"]
    ):
        raise RuntimeError("v5.1 cumulative oracle budget does not match direct")
    config["skip_root_stage_probes"] = bool(overlay["skip_root_stage_probes"])
    config["reused_direct_protocol"] = str(overlay["reused_direct_protocol"])
    config["reused_direct_branch_id"] = str(overlay["reused_direct_branch_id"])
    config["direct_completed_updates"] = int(overlay["direct_completed_updates"])
    config["student_array_spec"] = "1"
    config["evaluation_rng_key"] = str(overlay["evaluation_rng_key"])
    config["cumulative_budget"] = {
        "prior_bridge_updates": int(overlay["prior_bridge_updates"]),
        "new_oracle_updates": expected,
        "oracle_total_from_common_start": int(overlay["prior_bridge_updates"])
        + expected,
        "direct_total_from_common_start": int(overlay["direct_completed_updates"]),
        "matched": True,
    }
    config["second_painter_ladder"] = {
        "explicit_finite_prompts": True,
        "hidden_verifier_slicing": False,
        "strict_binary_reward_per_task": True,
        "partial_condition_reward": False,
        "first_new_stage_adds": "same-color second PAINTER",
        "second_new_stage_changes": "only final PAINTER to consumed input color",
        "target_rows_always_use_official_full_verifier": True,
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
        "fresh_optimizer": True,
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
