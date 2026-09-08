from __future__ import annotations

import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v4_6"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    overlay = read_json(
        EXP_ROOT / "manifests" / f"{VERSION}_overlay.json"
    )
    config = read_json(EXP_ROOT / str(overlay["base_manifest"]))
    config["protocol_version"] = VERSION
    config["purpose"] = (
        "Single-start, single-trajectory strict binary-reward experiment that "
        "rotates every easier verifier subset so each new constraint appears "
        "inside reward-bearing subproblems before the full verifier is enabled."
    )
    for stage in config["oracle_ladder"]:
        if "verification_case_counts" in stage:
            stage["verification_subset_mode"] = str(
                overlay["verification_subset_mode"]
            )
    config["coverage_annealing"] = {
        "subset_mode": str(overlay["verification_subset_mode"]),
        "strict_binary_reward": True,
        "partial_reward": False,
        "target_rows_always_use_full_verifier": True,
    }
    atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", config)

    from self_grok_decisive_v4_3_prepare import main as run_protocol

    run_protocol()


if __name__ == "__main__":
    main()
