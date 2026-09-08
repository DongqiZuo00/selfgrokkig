from __future__ import annotations

import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v4_7"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    overlay = read_json(EXP_ROOT / "manifests" / f"{VERSION}_overlay.json")
    config = read_json(EXP_ROOT / str(overlay["base_manifest"]))
    config["protocol_version"] = VERSION
    config["purpose"] = (
        "Single-start continuation from the durable v4.5 update-36 checkpoint, "
        "isolating whether rotating strict verifier subsets can teach every new "
        "constraint and ignite the full three- and four-case verifiers."
    )
    config["single_start"]["checkpoint"] = str(overlay["start_checkpoint"])
    config["single_start"]["scientific_update"] = int(
        overlay["start_scientific_update"]
    )
    config["oracle_ladder"] = config["oracle_ladder"][
        int(overlay["drop_completed_prefix_stages"]):
    ]
    for stage in config["oracle_ladder"]:
        if "verification_case_counts" in stage:
            stage["verification_subset_mode"] = str(
                overlay["verification_subset_mode"]
            )
    config["solver"]["matched_updates"] = int(overlay["matched_updates"])
    expected = sum(int(stage["updates"]) for stage in config["oracle_ladder"])
    expected += int(config["solver"]["target_tail_updates"])
    if expected != int(config["solver"]["matched_updates"]):
        raise RuntimeError(
            f"v4.7 matched budget mismatch: computed {expected}, "
            f"configured {config['solver']['matched_updates']}"
        )
    config["coverage_annealing"] = {
        "subset_mode": str(overlay["verification_subset_mode"]),
        "strict_binary_reward": True,
        "partial_reward": False,
        "target_rows_always_use_full_verifier": True,
        "continuation_checkpoint": str(overlay["start_checkpoint"]),
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
