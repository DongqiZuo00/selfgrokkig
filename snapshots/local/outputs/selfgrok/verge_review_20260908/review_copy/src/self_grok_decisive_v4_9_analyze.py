from __future__ import annotations

import json
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v4_9"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    from self_grok_decisive_v4_3_analyze import main as run_base_analysis

    run_base_analysis()
    root = EXP_ROOT / "raw_results" / VERSION
    recipe = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
    oracle_id = "decisive_mistral_v4_9__oracle"
    gate_dir = recipe / oracle_id / "stage_gates"
    gate_decisions = []
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.decision.json")):
            gate_decisions.append(read_json(path))
    final_gate = next(
        (item for item in gate_decisions if int(item["stage_index"]) == 5), None
    )
    decision_path = root / "decision.json"
    decision = read_json(decision_path)
    decision["finite_prepend_curriculum_result"] = {
        "explicit_finite_prompts": True,
        "painter_shortcut_excluded_at_first_stage": True,
        "gate_decisions": gate_decisions,
        "deepest_gate_reached": max(
            (int(item["stage_index"]) for item in gate_decisions), default=-1
        ),
        "all_inputs_upto_two_gate_reached": final_gate is not None,
        "all_inputs_upto_two_gate_accepted": bool(
            final_gate is not None and final_gate["accepted"]
        ),
    }
    atomic_json(decision_path, decision)
    complete_path = EXP_ROOT / "manifests" / f"{VERSION}_complete.json"
    complete = read_json(complete_path)
    complete["all_inputs_upto_two_gate_accepted"] = bool(
        final_gate is not None and final_gate["accepted"]
    )
    atomic_json(complete_path, complete)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
