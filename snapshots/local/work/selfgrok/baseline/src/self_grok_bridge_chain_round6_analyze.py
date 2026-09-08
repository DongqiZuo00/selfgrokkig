from __future__ import annotations

from common import EXP_ROOT, atomic_json, read_json
from self_grok_consolidation_round5_analyze import (
    endpoint,
    paired_effect,
    strict_effect,
    training_signal,
)


RAW_ROOT = EXP_ROOT / "raw_results" / "recipe_v1" / "branches"
DIRECT = "sg_bridge_r6__direct_target"
CHALLENGER = "sg_bridge_r6__challenger_chain"
UPDATES = (0, 25, 50, 75, 100)


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "self_grok_bridge_chain_round6.json")
    for branch_id in (DIRECT, CHALLENGER):
        status = read_json(RAW_ROOT / branch_id / "status.json")
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != 100:
            raise RuntimeError(f"incomplete branch: {branch_id}")

    endpoints = {}
    target_rates = {}
    scope_rates = {}
    for branch_id in (DIRECT, CHALLENGER):
        endpoints[branch_id] = {"target": {}, "scope_probe": {}}
        target_rates[branch_id] = {}
        scope_rates[branch_id] = {}
        for update in UPDATES:
            target, target_rate = endpoint(branch_id, "development", update, 50, 16)
            scope, scope_rate = endpoint(branch_id, "scope_probe", update, 32, 8)
            endpoints[branch_id]["target"][str(update)] = target
            endpoints[branch_id]["scope_probe"][str(update)] = scope
            target_rates[branch_id][update] = target_rate
            scope_rates[branch_id][update] = scope_rate

    margin = float(protocol["matched_contract"]["scope_margin"])
    comparisons = {}
    safe_updates = []
    for update in UPDATES:
        scope = paired_effect(scope_rates[CHALLENGER][update], scope_rates[DIRECT][update])
        target = paired_effect(target_rates[CHALLENGER][update], target_rates[DIRECT][update])
        strict = strict_effect(
            endpoints[CHALLENGER]["target"][str(update)],
            endpoints[DIRECT]["target"][str(update)],
        )
        scope_safe = scope["ci95_student_t"][0] > margin
        comparisons[str(update)] = {
            "target_soft_phi": target,
            "target_strict": strict,
            "scope_soft_phi": scope,
            "scope_safe": scope_safe,
        }
        if scope_safe:
            safe_updates.append(update)

    selected_update = max(safe_updates) if safe_updates else 0
    direct = endpoints[DIRECT]["target"][str(selected_update)]
    challenger = endpoints[CHALLENGER]["target"][str(selected_update)]
    decisive = challenger["strict_full_pass_rate"] > direct["strict_full_pass_rate"]
    decision = {
        "selected_scope_safe_update": selected_update,
        "challenger_strict_gain_over_direct": decisive,
        "decisive_single_seed_self_grok_phenomenon": decisive,
        "confirmation_opened": False,
        "official_test_opened": False,
        "no_further_automatic_experiment": True,
    }
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "complete",
        "endpoints": endpoints,
        "comparisons": comparisons,
        "training_signal": {branch: training_signal(branch) for branch in (DIRECT, CHALLENGER)},
        "decision": decision,
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "self_grok_bridge_chain_round6.json", result)
    atomic_json(
        EXP_ROOT / "manifests" / "self_grok_bridge_chain_round6_complete.json",
        {"status": "complete", **decision, "summary": "aggregated_results/self_grok_bridge_chain_round6.json"},
    )
    strict = comparisons[str(selected_update)]["target_strict"]
    scope = comparisons[str(selected_update)]["scope_soft_phi"]
    lines = [
        "# SELF-GROK bridge-chain Round 6",
        "",
        f"- Selected scope-safe update: {selected_update}",
        f"- Direct: strict={direct['strict_full_pass_count']}/{direct['rollouts']}, soft Phi={direct['soft_phi']:.6f}",
        f"- Challenger chain: strict={challenger['strict_full_pass_count']}/{challenger['rollouts']}, soft Phi={challenger['soft_phi']:.6f}",
        f"- Strict rate difference={strict['rate_difference']:.6f}, Fisher p={strict['fisher_one_sided_p']:.6g}",
        f"- Scope difference={scope['estimate']:.6f}, CI={scope['ci95_student_t']}",
        f"- Decisive SELF-GROK phenomenon: {decisive}",
        "- Confirmation and official test remain sealed. No further experiment was submitted automatically.",
        "",
    ]
    (EXP_ROOT / "RESULTS_SELF_GROK_BRIDGE_CHAIN_ROUND6.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
