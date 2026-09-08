"""Explicit new protocol; the completed round-1 manifest stays immutable."""
import json


def repair_config(version, exp):
    cfg = json.loads((exp / "manifests/verge_mistral_round1.json").read_text())
    acceptance = version == "verge_mistral_repair_acceptance"
    cfg.update(protocol_version=version, repair_integrated=True,
        scope="engineering acceptance only" if acceptance else "one repaired warm-start exploratory outer round",
        train_tokens_per_branch=32768 if acceptance else 524288,
        token_update_threshold=8192 if acceptance else 16384,
        proposal_max_tokens=2048, completion_tokens=2048,
        scope_max_drop=0.0, solver_top_p=1.0, solver_top_k=-1, solver_temperature=1.0,
        challenger_sampling="schema_constrained", challenger_loss="same_schema_masked_log_probability",
        scope_rule="observed non-decrease in both scope metrics; no population guarantee",
        output_interface="verge_code_prefix_v2", invalid_credit_policy="skip_entire_challenger_group",
        policy_identity_rule="zero-update fresh branches have exact zero gain against unchanged direct; retain raw estimates separately",
        last_batch_rule="fixed 2048 generation cap; full reward groups; uniform loss-token subset only at phase boundary",
        budget_note="524288 is 4x round 1; matched actual loss tokens, not matched optimizer-step count",
        update_note="step after >=threshold tokens at complete-group boundary; fewer steps than nominal token units",
        acceptance_only=acceptance, auto_submit_branches=False,
        target_train="data/verge_mistral_round1/target_train",
        target_selection="data/verge_mistral_round1/target_selection",
        target_completion="data/verge_mistral_round1/target_completion")
    if acceptance:
        cfg.update(selection_samples_per_instance=2, scope_instances=2, scope_samples=2,
                   stage_train_instances=8, endpoint_instances=2, completion_instances=2,
                   stage_probe_instances=0)
    else:
        cfg.update(endpoint_instances=64, completion_instances=64, stage_probe_instances=0)
    cfg["explicit_pilot_differences"] += [
        "v2 adds generic finite_map intermediate tasks; original final target data reused unchanged",
        "No separate stage probes or candidate rejection by estimated success",
        "Full-support temperature-1 sampling for both learners; Challenger additionally uses xgrammar mask",
        "Zero observed scope-drop tolerance replaces old five-percentage-point tolerance",
        "Fresh endpoint on shared generic code prefix; historical rates are not new baselines",
        "Engineering acceptance uses reduced training/evaluation budgets and cannot count as scientific result"]
    return cfg
