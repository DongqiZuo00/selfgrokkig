from __future__ import annotations

import copy
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_2"


def table_stage(template, name, updates, inputs, *, semantic_label=False):
    stage = copy.deepcopy(template)
    stage["name"] = name
    stage["updates"] = int(updates)
    stage["test_cases"] = "all"
    stage["prepend_curriculum_inputs"] = [str(value) for value in inputs]
    stage["prepend_curriculum_operation"] = "prepend"
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
        "Single-start, single-trajectory strict binary-reward continuation that "
        "separates executable I/O mappings from the misleading PREPEND lexical prior."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    template = config["oracle_ladder"][2]
    config["oracle_ladder"] = [
        table_stage(
            template,
            "io_table_opposite_single",
            6,
            ["$opposite"],
        ),
        table_stage(
            template,
            "io_table_all_single",
            6,
            ["R", "B"],
        ),
        table_stage(
            template,
            "io_table_empty_and_all_single",
            6,
            ["", "R", "B"],
        ),
        table_stage(
            template,
            "io_table_all_inputs_upto_two",
            8,
            ["", "R", "B", "RR", "RB", "BR", "BB"],
        ),
        table_stage(
            template,
            "same_table_with_prepend_semantic_label",
            4,
            ["", "R", "B", "RR", "RB", "BR", "BB"],
            semantic_label=True,
        ),
    ]
    config["solver"]["target_tail_updates"] = 4
    config["solver"]["matched_updates"] = int(overlay["additional_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v5.2 additional budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    if int(overlay["prior_bridge_updates"]) + expected != int(
        overlay["direct_completed_updates"]
    ):
        raise RuntimeError("v5.2 cumulative oracle budget does not match direct")
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
    config["lexical_bridge"] = {
        "mapping_stages_share_identical_prompt_schema": True,
        "mapping_stages_forbid_replace_and_prepend_words": True,
        "semantic_label_isolated_to_final_bridge_stage": True,
        "explicit_finite_io_table": True,
        "strict_binary_reward_per_task": True,
        "partial_condition_reward": False,
        "target_rows_always_use_official_full_verifier": True,
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
        "fresh_optimizer": True,
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
