"""Endpoint-only paired selection. No trajectory rollouts enter credit estimates."""
import math
import numpy as np
from verge_round_core import centered_advantages


def paired_interval(candidate, direct, rung, replicates):
    left, right = candidate["keyed_vectors"], direct["keyed_vectors"]
    assert set(left) == set(right)
    instance_ids = sorted({k.rsplit("::", 1)[0] for k in left})
    diff = np.array([[left[k][rung] - right[k][rung] for k in sorted(left)
                      if k.rsplit("::", 1)[0] == instance] for instance in instance_ids])
    rng = np.random.default_rng(42)
    means = []
    n, m = diff.shape
    for _ in range(replicates):
        instances = rng.integers(0, n, size=n)
        samples = rng.integers(0, m, size=(n, m))
        means.append(float(diff[instances[:, None], samples].mean()))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def scope_safe(endpoint, reference, tolerance):
    return all(endpoint["scope"][key] >= reference["scope"][key] - tolerance
               for key in ("full_pass_rate", "mean_case_fraction"))


def decide(frozen, branches):
    cfg, start, direct = frozen["config"], frozen["initial"], branches[0]
    assert all(b["training"]["train_tokens"] == cfg["train_tokens_per_branch"] for b in branches.values())
    enough = sum(b["target"]["counts"][-1] >= cfg["reward_switch_q"] for b in branches.values())
    reward_rung = 10 if enough >= 2 else frozen["reward_condition"]
    selection_rung = 10 if enough >= 1 else frozen["reward_condition"]
    evaluated = []
    rewards = []
    eligible = []
    for proposal in frozen["generation"]["candidates"]:
        index = proposal["index"]
        if not proposal["valid"]:
            rewards.append(-2.0)
            evaluated.append({"index": index, "valid": False, "reward": -2.0})
            continue
        b = branches[index]
        reward = b["target"]["rates"][reward_rung] - direct["target"]["rates"][reward_rung]
        gain = b["target"]["rates"][selection_rung] - direct["target"]["rates"][selection_rung]
        interval = paired_interval(b["target"], direct["target"], selection_rung, cfg["bootstrap_replicates"])
        observed_reward, observed_gain = reward, gain
        identical = (cfg.get("repair_integrated", False) and
                     b["training"]["optimizer_steps"] == direct["training"]["optimizer_steps"] == 0)
        if identical:
            # Both fresh branches retain exactly the same start adapter. This is
            # a policy identity, not an estimated transfer from sampling noise.
            reward, gain, interval = 0.0, 0.0, [0.0, 0.0]
        safe = scope_safe(b, start, cfg["scope_max_drop"]) and scope_safe(b, direct, cfg["scope_max_drop"])
        item = {"index": index, "valid": True, "reward": reward, "selection_gain": gain,
                "paired_95_interval": interval, "scope_safe": safe,
                "eligible": interval[0] > 0 and safe}
        if cfg.get("repair_integrated"):
            item.update(observed_endpoint_reward=observed_reward, observed_endpoint_gain=observed_gain,
                        identical_policy_by_zero_update_provenance=identical)
        rewards.append(reward)
        evaluated.append(item)
        if item["eligible"]:
            eligible.append({"name": f"candidate_{index}", "checkpoint": b["checkpoint"], "gain": gain})
    # Zero optimizer steps and zero weight decay imply the direct adapter is unchanged.
    # Deduplication uses training provenance, never a scan of checkpoint hashes.
    same_direct = direct["training"]["optimizer_steps"] == 0
    start_reference = {"name": "start", "checkpoint": start["checkpoint"],
                       "gain": start["target"]["rates"][selection_rung] - direct["target"]["rates"][selection_rung]}
    if cfg.get("book_suite") and same_direct:
        start_reference["gain"] = 0.0
    references = [start_reference]
    if not same_direct and scope_safe(direct, start, cfg["scope_max_drop"]):
        references.append({"name": "direct", "checkpoint": direct["checkpoint"], "gain": 0.0})
    probabilities = None
    if not eligible:
        selected = max(references, key=lambda r: r["gain"])
        rule = "no admissible positive-gain curriculum: retain better safe reference, ties retain start"
    else:
        candidates = eligible + references
        values = np.array([r["gain"] for r in candidates])
        z = (values - values.mean()) / max(1e-6, float(values.std()))
        weights = np.exp(cfg["selector_beta"] * z - max(cfg["selector_beta"] * z))
        probabilities = (weights / weights.sum()).tolist()
        threshold, selected = frozen["selection_draw"], candidates[-1]
        for candidate, probability in zip(candidates, probabilities):
            threshold -= probability
            if threshold <= 0:
                selected = candidate
                break
        rule = "sample positive paired-CI scope-safe curricula plus distinct safe references"
        probabilities = dict(zip([r["name"] for r in candidates], probabilities))
    result = {"reward_rung_zero_based": reward_rung, "selection_rung_zero_based": selection_rung,
            "full_pass_informative_branches": enough, "evaluated_candidates": evaluated,
            "challenger_rewards": rewards, "challenger_advantages": centered_advantages(rewards),
            "selected": selected, "selection_rule": rule, "selection_probabilities": probabilities,
            "direct_start_deduplicated": same_direct, "endpoint_only_credit": True,
            "confidence_note": "Exploratory paired percentile intervals, no multiplicity correction; not confirmatory evidence"}
    if cfg.get("repair_integrated"):
        from verge_repair_protocol import challenger_credit
        valid = [p["valid"] for p in frozen["generation"]["candidates"]]
        credit = challenger_credit(rewards, valid)
        result["challenger_advantages"] = credit["advantages"]
        result["challenger_update_allowed"] = credit["update_allowed"]
        result["challenger_credit_note"] = credit.get("reason", "Only legal candidates' signed target gains are centered")
        result["scope_rule"] = cfg["scope_rule"]
    if cfg.get("book_suite"):
        from verge_book_protocol import arm_decision
        return arm_decision(frozen, branches, result)
    return result
