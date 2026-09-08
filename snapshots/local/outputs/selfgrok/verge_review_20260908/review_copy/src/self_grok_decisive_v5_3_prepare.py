from __future__ import annotations

import copy
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_3"


def table_stage(
    template,
    name,
    updates,
    inputs,
    *,
    operation="prepend",
    semantic_label=False,
):
    stage = copy.deepcopy(template)
    stage["name"] = name
    stage["updates"] = int(updates)
    stage["test_cases"] = "all"
    stage["prepend_curriculum_inputs"] = [str(value) for value in inputs]
    stage["prepend_curriculum_operation"] = str(operation)
    stage["finite_curriculum_prompt_mode"] = "io_table"
    stage["prepend_curriculum_semantic_label"] = bool(semantic_label)
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
        "Single-start, single-trajectory strict binary-reward experiment that "
        "constructs the missing second Puller route one executable fragment at a time."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    template = config["oracle_ladder"][2]
    opposite_and_same = ["$opposite", "$prefix"]
    all_upto_two = ["", "R", "B", "RR", "RB", "BR", "BB"]
    config["oracle_ladder"] = [
        table_stage(
            template,
            "io_table_add_same_color_consume_route",
            4,
            opposite_and_same,
            operation="branch_consume_same",
        ),
        table_stage(
            template,
            "io_table_add_same_color_one_painter",
            4,
            opposite_and_same,
            operation="branch_replace_same",
        ),
        table_stage(
            template,
            "io_table_complete_both_single_routes",
            4,
            opposite_and_same,
        ),
        table_stage(
            template,
            "io_table_add_empty_route",
            4,
            ["", "R", "B"],
        ),
        table_stage(
            template,
            "io_table_all_inputs_upto_two",
            6,
            all_upto_two,
        ),
        table_stage(
            template,
            "same_table_with_prepend_semantic_label",
            4,
            all_upto_two,
            semantic_label=True,
        ),
    ]
    config["solver"]["target_tail_updates"] = 4
    config["solver"]["matched_updates"] = int(overlay["additional_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v5.3 additional budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    if int(overlay["prior_bridge_updates"]) + expected != int(
        overlay["direct_completed_updates"]
    ):
        raise RuntimeError("v5.3 cumulative oracle budget does not match direct")
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
    config["conditional_route_bridge"] = {
        "explicit_finite_io_table": True,
        "lexical_operation_words_withheld_until_semantic_stage": True,
        "strict_binary_reward_per_task": True,
        "partial_condition_reward": False,
        "stage_0_adds_only": "same-color Puller route ending on empty tape",
        "stage_1_adds_only": "one same-color Painter after that route",
        "stage_2_adds_only": "second same-color Painter for full PREPEND",
        "target_rows_always_use_official_full_verifier": True,
        "target_mix_fraction": float(config["solver"]["target_mix_fraction"]),
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
        "fresh_optimizer": True,
        "single_seed": 42,
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
