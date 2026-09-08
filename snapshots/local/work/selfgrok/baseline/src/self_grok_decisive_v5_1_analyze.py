from __future__ import annotations

import json
import os

from common import EXP_ROOT, atomic_json, read_json


VERSION = "self_grok_decisive_mistral_v5_1"


def main() -> None:
    if os.environ.get("SELF_GROK_DECISIVE_VERSION") != VERSION:
        raise RuntimeError(f"SELF_GROK_DECISIVE_VERSION must be {VERSION}")
    from self_grok_decisive_v4_3_analyze import main as run_base_analysis

    run_base_analysis()
    root = EXP_ROOT / "raw_results" / VERSION
    recipe = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
    oracle_id = "decisive_mistral_v5_1__oracle"
    gate_dir = recipe / oracle_id / "stage_gates"
    gate_decisions = []
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.decision.json")):
            gate_decisions.append(read_json(path))
    true_prepend_gate = next(
        (item for item in gate_decisions if int(item["stage_index"]) == 1), None
    )
    final_gate = next(
        (item for item in gate_decisions if int(item["stage_index"]) == 4), None
    )
    decision_path = root / "decision.json"
    decision = read_json(decision_path)
    decision["cumulative_budget"] = {
        "v5_0_prior_updates": 16,
        "v5_1_new_updates": int(decision["oracle_training"]["completed_updates"]),
        "planned_oracle_total": 54,
        "direct_control_total": 54,
    }
    decision["second_painter_ladder_result"] = {
        "explicit_finite_prompts": True,
        "strict_full_verifier_each_stage": True,
        "direct_control_reused_from": "self_grok_decisive_mistral_v4_9",
        "gate_decisions": gate_decisions,
        "deepest_gate_reached": max(
            (int(item["stage_index"]) for item in gate_decisions), default=-1
        ),
        "same_color_second_painter_gate_accepted": bool(
            gate_decisions and int(gate_decisions[0]["stage_index"]) == 0
            and gate_decisions[0]["accepted"]
        ),
        "true_one_symbol_prepend_gate_accepted": bool(
            true_prepend_gate is not None and true_prepend_gate["accepted"]
        ),
        "full_prepend_gate_reached": final_gate is not None,
        "full_prepend_gate_accepted": bool(
            final_gate is not None and final_gate["accepted"]
        ),
    }
    atomic_json(decision_path, decision)
    complete_path = EXP_ROOT / "manifests" / f"{VERSION}_complete.json"
    complete = read_json(complete_path)
    complete["cumulative_budget_matched_if_full_run"] = True
    complete["true_one_symbol_prepend_gate_accepted"] = bool(
        true_prepend_gate is not None and true_prepend_gate["accepted"]
    )
    complete["full_prepend_gate_accepted"] = bool(
        final_gate is not None and final_gate["accepted"]
    )
    complete["direct_control_reused_from"] = "self_grok_decisive_mistral_v4_9"
    atomic_json(complete_path, complete)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
