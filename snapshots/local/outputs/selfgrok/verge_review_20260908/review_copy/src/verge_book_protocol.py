"""Single-seed manuscript execution contract. No historical warm-start outcomes."""
import copy
import json
import os
from pathlib import Path

SUITE = os.environ.get("VERGE_BOOK_SUITE", "verge_book_v1")
if SUITE not in ("verge_book_v1", "verge_book_v2"):
    raise ValueError("Unknown book suite")
PRIMARY_ARMS = ("verge", "frozen", "uncertainty", "outcome")
MAX_ROUNDS = 6
BRANCH_TOKENS = 524288


def version(arm, round_index):
    if arm not in PRIMARY_ARMS or type(round_index) is not int or not 0 <= round_index < MAX_ROUNDS:
        raise ValueError("Arm or round outside the authorized bounded primary block")
    return f"{SUITE}_{arm}_r{round_index:02d}"


def compact_round(frozen, decision, branches, teacher):
    result = {"round": frozen["config"]["book_round"], "version": frozen["config"]["protocol_version"],
        "reward_condition": decision["reward_rung_zero_based"], "selected": decision["selected"]["name"],
        "teacher_rewards": decision["challenger_rewards"], "teacher_updated": teacher["updated"], "curricula": []}
    for proposal in frozen["generation"]["candidates"]:
        endpoint = branches[proposal["index"]]
        result["curricula"].append({"index": proposal["index"], "spec": proposal["spec"],
            "target_counts": endpoint["target"]["counts"], "target_draws": endpoint["target"]["rollouts"],
            "scope_full_rate": endpoint["scope"]["full_pass_rate"],
            "scope_mean_case_fraction": endpoint["scope"]["mean_case_fraction"],
            "optimizer_steps": endpoint["training"]["optimizer_steps"],
            "active_tokens": endpoint["training"]["nonzero_advantage_tokens"]})
    return result


def make_config(exp, arm, round_index, previous=None):
    from verge_repair_config import repair_config
    cfg = repair_config("verge_mistral_repair_v2", Path(exp))
    name = version(arm, round_index)
    cfg.update(protocol_version=name, book_suite=SUITE, book_arm=arm, book_round=round_index,
        scope="single-seed manuscript primary comparison; one of six atomic outer rounds",
        outer_rounds=MAX_ROUNDS, solver_rank=16, solver_learning_rate=1e-5,
        solver_start=f"checkpoints/{SUITE}_initial_solver/resume_u0000",
        challenger_start_checkpoint=f"checkpoints/{SUITE}_initial_challenger/resume_u0000",
        teacher_optimizer_inherit=True, prior_book_versions=[],
        teacher_update_enabled=arm != "frozen", teacher_reward=arm,
        train_tokens_per_branch=BRANCH_TOKENS, stage_probe_instances=0,
        uncertainty_source="stage curriculum-task training rollouts; no extra diagnostic sampling",
        root_provenance="Fresh rank-16 LoRA on frozen Mistral backbone, no human-curriculum training",
        random_stream_id=f"{SUITE}_paired_r{round_index:02d}",
        auto_submit_branches=False, book_atomic_round=True,
        final_evaluation_allowed=False, stop_after_first_success=False)
    if SUITE == "verge_book_v2":
        # Keep the existing paired base streams and training seed. Only the
        # within-batch per-prompt RNG allocation changes in this recovery.
        cfg.update(independent_prompt_streams=True,
            sampling_protocol="per_prompt_disjoint_seed_ranges_v1",
            random_stream_id=f"verge_book_v1_paired_r{round_index:02d}",
            restart_authorization="manifests/verge_book_v2_restart_authorization.json",
            initial_root_copy_from_suite="verge_book_v1")
    cfg["explicit_pilot_differences"] = [
        "User overrides manuscript Qwen and 6/3 seeds: Mistral only and one training seed",
        "Both roles start at fresh rank16 alpha32 dropout0 adapters; no warm start and no old curriculum history",
        "Manuscript Solver lr1e-5 and Challenger lr1e-6; each Solver branch has fresh optimizer and round-start KL",
        "2048 fixed completion cap and full-support sampler retained from repaired on-policy interface",
        "Typed JSON grammar masks are included in Challenger learner probabilities",
        "Registered task families plus previously implemented generic finite_map extension shared across arms",
        "No screening or extra stage-probe audits; uncertainty measured on logged curriculum training rollouts",
        "Identical loss-token allocations across primary arms; actual end-to-end costs reported, not presumed equal",
        "Single-seed estimates cannot replace manuscript six/three-seed uncertainty or generalization claims",
        "No early success stop; execute six rounds then freeze primary trajectory; official test remains sealed",
        "Invalid constrained action groups fail closed without resampling; later invalid-policy control remains a separate matrix item"]
    if round_index:
        if (previous is None or previous["config"]["book_round"] != round_index - 1
                or previous["config"]["book_arm"] != arm or previous["config"].get("book_suite") != SUITE):
            raise ValueError("Next round requires the immediately preceding same-arm atomic checkpoint pair")
        cfg["solver_start"] = previous["selected_solver"]
        cfg["challenger_start_checkpoint"] = previous["selected_challenger"]
        cfg["prior_book_versions"] = previous["config"]["prior_book_versions"] + [previous["config"]["protocol_version"]]
    return cfg


def proposal_messages(cfg, initial, history):
    profile = {key: initial["target"][key] for key in ("counts", "rates", "rollouts", "condition_names", "bottleneck_index")}
    context = {"target": cfg["target_descriptor"], "current_solver_profile": profile,
        "round_zero_based": cfg["book_round"], "archive": history,
        "training_contract": {"loss_tokens_per_branch": cfg["train_tokens_per_branch"],
            "target_prompt_fraction": .25, "target_only_tail_fraction": .25,
            "maximum_stages": 4, "solver_reward": "binary complete task pass", "completion_cap": 2048},
        "initialization": cfg["root_provenance"], "legal_families": cfg["families"]}
    return [{"role": "system", "content":
        "You are the curriculum-generating Challenger. Choose every task configuration and stage order yourself. "
        "Use the target, current Solver profile and past observations; no human curriculum is prescribed. "
        "Propose 1 to 4 executable intermediate stages as the requested JSON. Do not output solution programs."},
        {"role": "user", "content": json.dumps(context) +
         "\nFor standard families choose colors 2/4, length 1..6 and legal mutation none/simple/pattern. "
         "finite_map may instead specify colors, min_input_length and max_input_length in 0..2 and outputs. "
         "The outputs array follows increasing input length then Cartesian-product order over RB or RBYG. "
         "One output per allowed input; each output uses the same alphabet and has length 0..6. "
         "Every intermediate task uses its complete verifier. The fixed target is never weakened."}]


def arm_decision(frozen, branches, base_decision):
    """Change only teacher return/update; preserve common target-conditioned selector."""
    from verge_round_core import centered_advantages
    cfg = frozen["config"]
    decision = copy.deepcopy(base_decision)
    mode = cfg["teacher_reward"]
    if mode == "uncertainty":
        values = []
        for i in (1, 2, 3):
            stats = branches[i]["training"]["curriculum_stage_reward_counts"]
            rates = [x["successes"] / x["draws"] for x in stats if x["draws"]]
            if len(rates) != len(frozen["generation"]["candidates"][i-1]["spec"]["stages"]):
                raise ValueError("Missing curriculum-stage observations")
            values.append(sum(1 - 2 * abs(p - .5) for p in rates) / len(rates))
    elif mode == "outcome":
        initial_rate = frozen["initial"]["target"]["rates"][-1]
        values = [branches[i]["target"]["rates"][-1] - initial_rate
                  if branches[i]["training"]["optimizer_steps"] else 0. for i in (1, 2, 3)]
    elif mode in ("verge", "frozen"):
        values = decision["challenger_rewards"]
    else:
        raise ValueError("Unsupported primary return")
    decision.update(target_condition_gains=base_decision["challenger_rewards"], teacher_reward_mode=mode,
        challenger_rewards=values, challenger_advantages=centered_advantages(values),
        challenger_update_allowed=cfg["teacher_update_enabled"] and len(set(values)) > 1,
        challenger_credit_note="Frozen common selector; teacher-return ablation with logged source")
    return decision
