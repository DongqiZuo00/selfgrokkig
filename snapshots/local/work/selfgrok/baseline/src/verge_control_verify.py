"""Check control-segment artifacts without counting partial rewards as successes."""
from common import read_json, read_jsonl, group_advantages
from verge_control_protocol import validate_control_config
from verge_control_rewards import solver_rewards
from verge_independent_sampling import validate_response, validate_records, PROTOCOL


def validate_segment(frozen, result, directory):
    cfg = frozen["config"]
    validate_control_config(cfg, cfg["protocol_version"])
    state = result["training"]
    assert len(frozen["branches"]) == 1 and frozen["branches"][0]["index"] == 0
    stages = frozen["branches"][0]["stages"]
    assert [{k:s[k] for k in ("tokens", "kind", "reward_mode")} for s in stages] == cfg["control_phases"]
    assert 0 < state["update"] <= 100
    totals = {"tokens":0, "generated":0, "active":0, "steps":0, "full_successes":0}
    phase_tokens = [0]*len(stages)
    for update in range(1, state["update"]+1):
        rows = read_jsonl(directory / "train" / f"update_{update:04d}.jsonl")
        allocation = read_json(directory / "train" / f"update_{update:04d}_allocation.json")
        summary = read_json(directory / "train" / f"update_{update:04d}_summary.json")
        phase = allocation["stage"]
        mode = stages[phase]["reward_mode"]
        assert len(rows) % 8 == 0 and allocation["advantage_source"] == "optimization_reward"
        rewards = solver_rewards(mode, rows, role="target")
        assert rewards == [r["optimization_reward"] for r in rows]
        assert allocation["advantages"] == group_advantages(rewards, 8)
        for start in range(0, len(rows), 8):
            group = rows[start:start+8]
            request = {"prompt":[group[0]["prompt_token_ids"]], "n":8,
                       "seed":group[0]["sampling_parent_seed"]}
            validate_records(request, group)
            assert len({r["instance_id"] for r in group}) == 1
        masks = allocation["loss_masks"]
        assert len(masks) == len(rows)
        for row, mask, adv in zip(rows, masks, allocation["advantages"]):
            assert row["sampling_protocol"] == PROTOCOL and row["prompt_role"] == "target"
            assert row["optimization_reward_mode"] == mode and row["max_tokens"] == 2048
            assert 0 < len(mask) == row["completion_tokens"] == len(row["completion_token_ids"]) <= 2048
            assert set(mask) <= {0,1}
            n = sum(mask)
            totals["tokens"] += n
            phase_tokens[phase] += n
            totals["active"] += n * int(abs(adv) > 1e-12)
            totals["generated"] += row["completion_tokens"]
            totals["full_successes"] += row["reward"]
        totals["steps"] += int(summary["optimizer_step"])
    assert phase_tokens == state["used_by_stage"] == [p["tokens"] for p in cfg["control_phases"]]
    assert totals["tokens"] == state["train_tokens"] == cfg["train_tokens_per_branch"]
    assert totals["generated"] == state["generated_tokens"]
    assert totals["active"] == state["nonzero_advantage_tokens"]
    assert totals["steps"] == state["optimizer_steps"]
    assert totals["full_successes"] == state["target_successes"]
    for kind, count in (("target",cfg["endpoint_instances"]*cfg["selection_samples_per_instance"]),
                        ("scope",cfg["scope_instances"]*cfg["scope_samples"])):
        path = directory / f"{kind}.jsonl"
        saved = read_json(path.with_suffix(".response.json"))
        records = read_jsonl(path)
        assert len(records) == count and saved["backend_protocol"] == PROTOCOL
        validate_response(saved["request"], saved["response"])
        validate_records(saved["request"], records)
    return {"control_segment_verified": True, "comparison_complete": False}
