from __future__ import annotations

import json
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_2"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    from self_grok_decisive_v4_3_analyze import main as run_base_analysis

    run_base_analysis()
    root = EXP_ROOT / "raw_results" / VERSION
    recipe = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
    oracle_id = "decisive_mistral_v5_2__oracle"
    gate_dir = recipe / oracle_id / "stage_gates"
    gate_decisions = []
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.decision.json")):
            gate_decisions.append(read_json(path))
    length_two_gate = next(
        (item for item in gate_decisions if int(item["stage_index"]) == 3), None
    )
    semantic_gate = next(
        (item for item in gate_decisions if int(item["stage_index"]) == 4), None
    )
    decision_path = root / "decision.json"
    decision = read_json(decision_path)
    new_updates = int(decision["oracle_training"]["completed_updates"])
    decision["cumulative_budget"] = {
        "v5_0_and_v5_1_prior_updates": 20,
        "v5_2_new_updates": new_updates,
        "actual_oracle_total": 20 + new_updates,
        "planned_oracle_total": 54,
        "direct_control_total": 54,
    }
    decision["lexical_bridge_result"] = {
        "explicit_io_table_prompts": True,
        "strict_full_verifier_each_stage": True,
        "direct_control_reused_from": "self_grok_decisive_mistral_v4_9",
        "gate_decisions": gate_decisions,
        "deepest_gate_reached": max(
            (int(item["stage_index"]) for item in gate_decisions), default=-1
        ),
        "true_one_symbol_table_gate_accepted": bool(
            gate_decisions and int(gate_decisions[0]["stage_index"]) == 0
            and gate_decisions[0]["accepted"]
        ),
        "length_two_table_gate_accepted": bool(
            length_two_gate is not None and length_two_gate["accepted"]
        ),
        "prepend_semantic_gate_reached": semantic_gate is not None,
        "prepend_semantic_gate_accepted": bool(
            semantic_gate is not None and semantic_gate["accepted"]
        ),
    }
    atomic_json(decision_path, decision)
    complete_path = EXP_ROOT / "manifests" / f"{VERSION}_complete.json"
    complete = read_json(complete_path)
    complete["actual_oracle_total_updates"] = 20 + new_updates
    complete["true_one_symbol_table_gate_accepted"] = bool(
        gate_decisions and gate_decisions[0]["accepted"]
    )
    complete["prepend_semantic_gate_accepted"] = bool(
        semantic_gate is not None and semantic_gate["accepted"]
    )
    complete["direct_control_reused_from"] = "self_grok_decisive_mistral_v4_9"
    atomic_json(complete_path, complete)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
