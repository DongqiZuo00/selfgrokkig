"""Bounded direct-control segment configuration; no budget or warm-up inference."""
import copy
import re
from pathlib import Path
from verge_repair_config import repair_config

CONTROL_PATTERN = r"verge_control_v1_(binary|dense|dense_then_binary)_r0[0-5]"


def validate_control_config(cfg, namespace):
    match = re.fullmatch(CONTROL_PATTERN, namespace)
    if (match is None or cfg.get("protocol_version") != namespace
            or cfg.get("control_only") is not True or cfg.get("control_reward_protocol") != "appendix_d_v1"
            or cfg.get("control_label") != match.group(1)
            or cfg.get("book_suite") or cfg.get("book_atomic_round")):
        raise ValueError("Not an explicitly isolated direct-control segment")
    if (cfg.get("independent_prompt_streams") is not True
            or cfg.get("sampling_protocol") != "per_prompt_disjoint_seed_ranges_v1"
            or cfg.get("maximum_updates_per_branch") != 100
            or cfg.get("completion_tokens") != 2048 or cfg.get("solver_rank") != 16
            or cfg.get("solver_learning_rate") != 1e-5 or cfg.get("final_evaluation_allowed") is not False):
        raise ValueError("Control segment changed the fixed sampler/optimizer/safety contract")
    if (cfg.get("seed") != 42 or cfg.get("weight_decay") != 0.0
            or cfg.get("kl_beta") != 0.02 or cfg.get("solver_temperature") != 1.0
            or cfg.get("solver_top_p") != 1.0 or cfg.get("solver_top_k") != -1
            or cfg.get("selection_samples_per_instance") != 8
            or cfg.get("endpoint_instances") != 64 or cfg.get("scope_instances") != 32
            or cfg.get("scope_samples") != 4 or cfg.get("token_update_threshold") != 16384
            or cfg.get("acceptance_only") is not False):
        raise ValueError("Control segment changed the fixed training or endpoint contract")
    round_index = cfg.get("control_round")
    if type(round_index) is not int or namespace != f"verge_control_v1_{cfg['control_label']}_r{round_index:02d}":
        raise ValueError("Control round does not match its namespace")
    if not re.fullmatch(r"manifests/verge_control_v1_[a-z_]+_protocol\.json", cfg.get("control_block_protocol", "")):
        raise ValueError("A separate control-block protocol reference is required")
    budget = cfg["train_tokens_per_branch"]
    phases = cfg["control_phases"]
    if (type(budget) is not int or not 0 < budget <= 524288 or not 1 <= len(phases) <= 2
            or any(p.get("kind") != "target" or p.get("reward_mode") not in ("binary", "dense")
                   or type(p.get("tokens")) is not int or p["tokens"] <= 0 for p in phases)
            or sum(p["tokens"] for p in phases) != budget):
        raise ValueError("Control must declare exact, bounded target-only loss-token phases")
    modes = [p["reward_mode"] for p in phases]
    if cfg["control_label"] in ("binary", "dense") and modes != [cfg["control_label"]]:
        raise ValueError("Single-mode control has an inconsistent phase plan")
    if cfg["control_label"] == "dense_then_binary" and modes not in (["dense"], ["binary"], ["dense", "binary"]):
        raise ValueError("A fixed global warm-up may never switch back from binary to dense")
    source = cfg["solver_start"]
    if round_index == 0:
        source_ok = source == "checkpoints/verge_book_v2_initial_solver/resume_u0000"
    else:
        expected = f"checkpoints/verge_control_v1_{cfg['control_label']}_r{round_index-1:02d}_direct/resume_u"
        source_ok = isinstance(source, str) and re.fullmatch(re.escape(expected) + r"\d{4}", source) is not None
    if not source_ok:
        raise ValueError("Control must start at the untrained root or its immediately preceding same-arm segment")
    return cfg


def segment_config(exp, label, round_index, *, phases, starting_checkpoint, block_protocol):
    """Caller must supply the pre-frozen global comparison and this segment's phases.

The function does not pick a warm-up length or authorize additional segments.
An outer dispatcher must validate the predecessor and total comparison budget.
"""
    if label not in ("binary", "dense", "dense_then_binary") or type(round_index) is not int or not 0 <= round_index < 6:
        raise ValueError("Unknown or unbounded control segment")
    if not isinstance(block_protocol, str) or not re.fullmatch(r"manifests/verge_control_v1_[a-z_]+_protocol\.json", block_protocol):
        raise ValueError("A separate control-block protocol reference is required")
    cfg = repair_config("verge_mistral_repair_v2", Path(exp))
    cfg.update(protocol_version=f"verge_control_v1_{label}_r{round_index:02d}",
        control_only=True, control_label=label, control_round=round_index,
        control_reward_protocol="appendix_d_v1", control_block_protocol=block_protocol,
        control_phases=copy.deepcopy(phases), train_tokens_per_branch=sum(p["tokens"] for p in phases),
        solver_start=starting_checkpoint, solver_rank=16, solver_learning_rate=1e-5,
        maximum_updates_per_branch=100, independent_prompt_streams=True,
        sampling_protocol="per_prompt_disjoint_seed_ranges_v1",
        random_stream_id=f"verge_control_v1_paired_r{round_index:02d}",
        challenger_start=None, proposal_count=0, stage_probe_instances=0,
        budget_note="One separately frozen direct-control segment; no total-compute equivalence is implied",
        explicit_pilot_differences=[
            "User-selected Mistral primary backbone and one training seed42",
            "Untrained rank16 initial Solver; subsequent segments inherit only this same control arm",
            "Fixed PREPEND target and unchanged full verifier; partial optimization rewards are not full successes",
            "Loss-token budgets only; total-compute comparison and global warm-up require a separate frozen block",
            "No new audit, stage probe, training-seed scan or official-test access"],
        scope="One bounded direct-control segment; not the completed comparison or experiment book",
        final_evaluation_allowed=False, stop_after_first_success=False)
    return validate_control_config(cfg, cfg["protocol_version"])
