from __future__ import annotations

import copy
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v4_8"


def anchored_stage(template, name, updates, counts, required, mode):
    stage = copy.deepcopy(template)
    stage["name"] = name
    stage["updates"] = int(updates)
    stage["test_cases"] = 3
    stage["verification_case_counts"] = [int(value) for value in counts]
    stage["verification_subset_mode"] = str(mode)
    stage["verification_required_case_index"] = int(required)
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
        "Single-start, single-trajectory strict binary-reward test of whether a "
        "previously unreachable third verifier case can be taught by forcing every "
        "partial-reward subset to contain that case before full three-case credit."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    required = int(overlay["required_case_index"])
    mode = str(overlay["verification_subset_mode"])
    updates = int(overlay["stage_updates"])
    template = config["oracle_ladder"][2]
    strict = copy.deepcopy(config["oracle_ladder"][5])
    strict["name"] = "prepend_nomutation_len1_three_cases_strict_after_case3_anchor"
    strict["updates"] = int(overlay["strict_stage_updates"])
    strict.pop("verification_case_counts", None)
    strict.pop("verification_subset_mode", None)
    strict.pop("verification_required_case_index", None)
    config["oracle_ladder"] = [
        anchored_stage(
            template,
            "prepend_nomutation_len1_case3_only",
            updates,
            [1] * 8,
            required,
            mode,
        ),
        anchored_stage(
            template,
            "prepend_nomutation_len1_case3_plus_one_old",
            updates,
            [2] * 8,
            required,
            mode,
        ),
        anchored_stage(
            template,
            "prepend_nomutation_len1_case3_anchor_25pct_full",
            updates,
            [2, 2, 2, 2, 2, 2, 3, 3],
            required,
            mode,
        ),
        anchored_stage(
            template,
            "prepend_nomutation_len1_case3_anchor_50pct_full",
            updates,
            [2, 2, 2, 2, 3, 3, 3, 3],
            required,
            mode,
        ),
        anchored_stage(
            template,
            "prepend_nomutation_len1_case3_anchor_75pct_full",
            updates,
            [2, 2, 3, 3, 3, 3, 3, 3],
            required,
            mode,
        ),
        strict,
    ]
    config["solver"]["target_tail_updates"] = int(overlay["target_tail_updates"])
    config["solver"]["matched_updates"] = int(overlay["matched_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v4.8 matched budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    config["skip_root_stage_probes"] = bool(overlay["skip_root_stage_probes"])
    config["case3_anchor_curriculum"] = {
        "required_case_index": required,
        "subset_mode": mode,
        "all_partial_rewards_include_required_case": True,
        "strict_binary_reward": True,
        "partial_condition_reward": False,
        "target_rows_always_use_full_verifier": True,
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
