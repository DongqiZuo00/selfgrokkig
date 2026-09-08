"""Opt-in training adapter for separate control runners, never the primary loop.

Keep the raw binary verifier outcome for full-success events and give the
optimizer its separately recorded control reward. No model sampling happens here.
"""
import math
from verge_control_rewards import solver_rewards


def prepare_group(records, row, stage, cfg, *, role, condition_evaluator=None):
    if cfg.get("control_only") is not True or cfg.get("control_reward_protocol") != "appendix_d_v1":
        raise ValueError("Control rewards require an explicitly isolated control configuration")
    if cfg.get("book_atomic_round") or cfg.get("book_suite") in ("verge_book_v1", "verge_book_v2"):
        raise ValueError("Refusing control reward injection into a primary book round")
    if len(records) != 8 or any(r["instance_id"] != row["id"] for r in records):
        raise ValueError("Expected one complete eight-action group for the declared prompt")
    if [r["rollout_index"] for r in records] != list(range(8)):
        raise ValueError("Incomplete or misordered action group")
    mode = stage["reward_mode"]
    vectors = None
    if mode == "target_condition_shaped" and role == "target":
        if condition_evaluator is None:
            from verge_round_core import condition_vector
            condition_evaluator = condition_vector
        vectors = [condition_evaluator(r["completion"], row) for r in records]
    values = solver_rewards(mode, records, role=role, condition_vectors=vectors,
        frozen_rung=cfg.get("frozen_reward_rung"), coefficient=cfg.get("shaping_coefficient"))
    if len(values) != 8 or any(not math.isfinite(v) for v in values):
        raise ValueError("Invalid optimization rewards")
    from common import group_advantages
    advantages = group_advantages(values, 8)
    prepared = []
    for record, reward in zip(records, values):
        prepared.append(dict(record, optimization_reward=reward, optimization_reward_mode=mode,
                             prompt_role=role, control_reward_protocol="appendix_d_v1"))
    return prepared, advantages, {
        "reward_mode": mode, "prompt_role": role,
        "binary_full_successes": sum(r["reward"] for r in records),
        "mean_optimization_reward": sum(values) / len(values),
        "optimization_reward_varies": len(set(values)) > 1,
        "raw_verifier_rewards_overwritten": False}
