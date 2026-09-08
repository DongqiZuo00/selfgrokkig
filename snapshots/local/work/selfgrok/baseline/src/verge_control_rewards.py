"""Isolated Appendix D reward definitions; not imported by the primary runner.

These helpers are preparation for separate, not-yet-launched control blocks.
All inputs must come from the complete verifier, including every test case.
"""
import math

MODES = ("binary", "dense", "target_condition_shaped")


def checked_verdict(record):
    full, passed, total = (record[k] for k in ("reward", "passed_cases", "total_cases"))
    if full not in (0, 1) or type(passed) is not int or type(total) is not int:
        raise ValueError("Expected binary full verifier and integer all-test counts")
    if total < 1 or not 0 <= passed <= total or bool(full) != (passed == total):
        raise ValueError("Incomplete or inconsistent all-test verification")
    return float(full), passed / total


def solver_rewards(mode, records, *, role, condition_vectors=None, frozen_rung=None, coefficient=None):
    if mode not in MODES or role not in ("target", "curriculum") or not records:
        raise ValueError("Unknown reward mode, prompt role or empty action group")
    verdicts = [checked_verdict(r) for r in records]
    if role == "curriculum":
        if condition_vectors is not None:
            raise ValueError("Never feed target-condition scores to intermediate-task rewards")
        return [full for full, _ in verdicts]
    if mode == "binary":
        return [full for full, _ in verdicts]
    if mode == "dense":
        return [fraction for _, fraction in verdicts]
    if (type(frozen_rung) is not int or frozen_rung < 0 or coefficient is None
            or not math.isfinite(coefficient) or coefficient < 0
            or condition_vectors is None or len(condition_vectors) != len(records)):
        raise ValueError("Shaping requires a pre-frozen rung and finite nonnegative coefficient")
    rewards = []
    width = len(condition_vectors[0])
    for (full, _), vector in zip(verdicts, condition_vectors):
        if (len(vector) != width or frozen_rung >= len(vector) or any(x not in (0, 1) for x in vector)
                or any(b > a for a, b in zip(vector, vector[1:])) or vector[-1] != full):
            raise ValueError("Invalid cumulative necessary-condition vector")
        rewards.append(full + coefficient * vector[frozen_rung])
    return rewards


def direct_reward_phases(mode, total_loss_tokens, *, warmup_loss_tokens=None):
    """No inferred warm-up duration. The caller must freeze its token boundary."""
    if type(total_loss_tokens) is not int or total_loss_tokens <= 0:
        raise ValueError("Training budget must be a positive number of loss tokens")
    if mode in ("binary", "dense"):
        if warmup_loss_tokens is not None:
            raise ValueError("Unexpected warm-up setting for a single-phase control")
        return [{"reward_mode": mode, "kind": "target", "tokens": total_loss_tokens}]
    if mode != "dense_then_binary" or type(warmup_loss_tokens) is not int or not 0 < warmup_loss_tokens < total_loss_tokens:
        raise ValueError("Dense-to-binary control needs an explicit interior token boundary")
    # The existing phase-boundary masking rule can enforce this exact quota
    # without truncating generated programs or splitting a reward group.
    return [{"reward_mode": "dense", "kind": "target", "tokens": warmup_loss_tokens},
            {"reward_mode": "binary", "kind": "target", "tokens": total_loss_tokens - warmup_loss_tokens}]


def scalar_partial_credit(candidate, direct, *, unchanged_from_common_start=False):
    """Endpoint-only per-test fraction gain; never accept training trajectories.

Callers supply validated fixed-checkpoint batch containers. Retain the observed
estimate separately when exact policy identity follows from zero-update lineage.
"""
    from verge_independent_sampling import PROTOCOL
    for batch in (candidate, direct):
        if batch.get("data_role") != "fixed_checkpoint_endpoint" or not batch.get("checkpoint"):
            raise ValueError("Scalar Challenger credit requires fixed-checkpoint endpoint batches")
        rows = batch["records"]
        if not rows or any(r.get("sampling_protocol") != PROTOCOL for r in rows):
            raise ValueError("Missing independent-stream endpoint records")
        if len({r["sampling_child_seed"] for r in rows}) != len(rows):
            raise ValueError("Endpoint batch reused configured child random streams")
    def keyed(batch):
        result = {}
        for row in batch["records"]:
            key = (row["instance_id"], row["rollout_index"])
            if key in result:
                raise ValueError("Duplicate endpoint draw")
            result[key] = (checked_verdict(row)[1], row["sampling_child_seed"])
        return result
    a, b = keyed(candidate), keyed(direct)
    if a.keys() != b.keys() or any(a[k][1] != b[k][1] for k in a):
        raise ValueError("Endpoint instances and configured random streams must remain paired")
    observed = sum(a[k][0] - b[k][0] for k in a) / len(a)
    return {"observed_gain": observed, "reward": 0. if unchanged_from_common_start else observed,
            "identity_override": unchanged_from_common_start, "draws": len(a)}
