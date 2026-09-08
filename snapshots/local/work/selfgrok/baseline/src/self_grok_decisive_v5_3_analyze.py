from __future__ import annotations

import json
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_3"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    from self_grok_decisive_v4_3_analyze import main as run_base_analysis

    run_base_analysis()
    root = EXP_ROOT / "raw_results" / VERSION
    recipe = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
    oracle_id = "decisive_mistral_v5_3__oracle"
    gate_dir = recipe / oracle_id / "stage_gates"
    gate_decisions = []
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.decision.json")):
            gate_decisions.append(read_json(path))
    by_stage = {int(item["stage_index"]): item for item in gate_decisions}
    decision_path = root / "decision.json"
    decision = read_json(decision_path)
    new_updates = int(decision["oracle_training"]["completed_updates"])
    decision["cumulative_budget"] = {
        "v5_0_through_v5_2_safe_prior_updates": 24,
        "v5_3_new_updates": new_updates,
        "actual_oracle_total": 24 + new_updates,
        "planned_oracle_total": 54,
        "direct_control_total": 54,
    }
    decision["conditional_route_bridge_result"] = {
        "explicit_io_table_prompts": True,
        "strict_full_verifier_each_stage": True,
        "single_seed": 42,
        "direct_control_reused_from": "self_grok_decisive_mistral_v4_9",
        "gate_decisions": gate_decisions,
        "deepest_gate_reached": max(by_stage, default=-1),
        "same_color_consume_route_accepted": bool(
            by_stage.get(0, {}).get("accepted", False)
        ),
        "same_color_one_painter_accepted": bool(
            by_stage.get(1, {}).get("accepted", False)
        ),
        "both_single_routes_accepted": bool(
            by_stage.get(2, {}).get("accepted", False)
        ),
        "empty_route_accepted": bool(by_stage.get(3, {}).get("accepted", False)),
        "length_two_table_accepted": bool(
            by_stage.get(4, {}).get("accepted", False)
        ),
        "prepend_semantic_gate_accepted": bool(
            by_stage.get(5, {}).get("accepted", False)
        ),
    }
    atomic_json(decision_path, decision)
    complete_path = EXP_ROOT / "manifests" / f"{VERSION}_complete.json"
    complete = read_json(complete_path)
    complete["actual_oracle_total_updates"] = 24 + new_updates
    complete["same_color_consume_route_accepted"] = bool(
        by_stage.get(0, {}).get("accepted", False)
    )
    complete["both_single_routes_accepted"] = bool(
        by_stage.get(2, {}).get("accepted", False)
    )
    complete["prepend_semantic_gate_accepted"] = bool(
        by_stage.get(5, {}).get("accepted", False)
    )
    complete["direct_control_reused_from"] = "self_grok_decisive_mistral_v4_9"
    atomic_json(complete_path, complete)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
