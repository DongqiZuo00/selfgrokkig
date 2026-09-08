"""Validate produced accounting and endpoint artifacts; no sampling, hashes or audits."""
from common import EXP_ROOT, read_json, read_jsonl, group_advantages
from verge_round_core import ROOT, VERSION
from verge_round_decision import decide


def validate_independent_batch(path):
    from verge_independent_sampling import PROTOCOL, validate_response, validate_records
    saved = read_json(path.with_suffix(".response.json"))
    assert saved["backend_protocol"] == PROTOCOL
    validate_response(saved["request"], saved["response"])
    validate_records(saved["request"], read_jsonl(path))


def validate(frozen, branches, decision, teacher, completion):
    cfg = frozen["config"]
    assert cfg["repair_integrated"] and cfg["scope_max_drop"] == 0
    assert len(branches) == 4 and all(c["valid"] for c in frozen["generation"]["candidates"])
    assert frozen["initial"]["interface_version"] == cfg["output_interface"]
    assert frozen["initial"]["target"]["rollouts"] == cfg["endpoint_instances"] * cfg["selection_samples_per_instance"]
    if cfg.get("independent_prompt_streams"):
        for kind in ("target", "scope"):
            validate_independent_batch(ROOT / "initial" / f"{kind}.jsonl")
        validate_independent_batch(ROOT / "completion.jsonl")
    for index, branch in branches.items():
        state = branch["training"]
        assert state["train_tokens"] == cfg["train_tokens_per_branch"]
        assert state["update"] <= 100 and branch["native_token_accounting"]
        directory = ROOT / "branches" / str(index)
        tokens = generated = active = steps = masked = 0
        for update in range(1, state["update"] + 1):
            data = read_jsonl(directory / "train" / f"update_{update:04d}.jsonl")
            if cfg.get("independent_prompt_streams"):
                from verge_independent_sampling import PROTOCOL
                assert all(r.get("sampling_protocol") == PROTOCOL for r in data)
                # Training units concatenate complete one-prompt/eight-draw groups.
                for start in range(0, len(data), 8):
                    group = data[start:start+8]
                    parent = group[0]["sampling_parent_seed"]
                    assert all(r["sampling_parent_seed"] == parent for r in group)
                    assert [r["sampling_child_seed"] for r in group] == list(range(parent, parent+8))
            allocation = read_json(directory / "train" / f"update_{update:04d}_allocation.json")
            summary = read_json(directory / "train" / f"update_{update:04d}_summary.json")
            assert len(data) % 8 == 0
            assert len(data) == len(allocation["advantages"]) == len(allocation["loss_masks"])
            assert allocation["advantages"] == group_advantages([r["reward"] for r in data], 8)
            for record, mask in zip(data, allocation["loss_masks"]):
                assert record["max_tokens"] == 2048
                assert 0 < len(record["completion_token_ids"]) == len(mask) == record["completion_tokens"] <= 2048
                assert set(mask) <= {0, 1} and record["prompt_token_ids"]
                generated += record["completion_tokens"]
                tokens += sum(mask)
            active += summary["nonzero_advantage_tokens"]
            steps += int(summary["optimizer_step"])
            masked += summary["masked_phase_boundary_groups"]
            assert summary["tokens_used"] == sum(map(sum, allocation["loss_masks"]))
        assert tokens == state["train_tokens"] and generated == state["generated_tokens"]
        assert active == state["nonzero_advantage_tokens"] and steps == state["optimizer_steps"]
        assert masked == state["masked_phase_boundary_groups"] <= len(state["used_by_stage"])
        if cfg.get("book_suite"):
            expected = [{"draws": 0, "successes": 0} for s in frozen["branches"][index]["stages"] if s["kind"] == "curriculum"]
            for update in range(1, state["update"] + 1):
                summary = read_json(directory / "train" / f"update_{update:04d}_summary.json")
                if summary["stage_kind"] == "curriculum":
                    data = read_jsonl(directory / "train" / f"update_{update:04d}.jsonl")
                    counter = expected[summary["stage"]]
                    counter["draws"] += sum(r["prompt_role"] == "curriculum" for r in data)
                    counter["successes"] += sum(r["reward"] for r in data if r["prompt_role"] == "curriculum")
            assert expected == state["curriculum_stage_reward_counts"]
        raw = read_jsonl(directory / "target.jsonl")
        assert len(raw) == branch["target"]["rollouts"] == cfg["endpoint_instances"] * cfg["selection_samples_per_instance"]
        assert sum(r["reward"] for r in raw) == branch["target"]["counts"][-1]
        assert branch["interface_version"] == cfg["output_interface"]
        if cfg.get("independent_prompt_streams"):
            for kind in ("target", "scope"):
                validate_independent_batch(directory / f"{kind}.jsonl")
    assert decision == decide(frozen, branches)
    assert teacher["whole_proposal_advantages"] == decision["challenger_advantages"]
    assert len(teacher["likelihood_checks"]) == 3
    assert completion["rollouts"] == cfg["completion_instances"]
    if cfg.get("book_suite"):
        assert cfg["solver_rank"] == 16 and "v5_5" not in cfg["solver_start"]
        assert teacher["inherited_optimizer"]
        if cfg["book_arm"] == "frozen":
            assert not teacher["updated"] and teacher["cumulative_optimizer_steps"] == 0
    for checkpoint in (decision["selected"]["checkpoint"], teacher["checkpoint"]):
        assert (EXP_ROOT / checkpoint / "adapter_model.safetensors").exists()
    return {"validated": True, "version": VERSION, "acceptance_only": cfg["acceptance_only"]}


if __name__ == "__main__":
    import json
    frozen = read_json(ROOT / "round_frozen.json")
    branches = {b["index"]: read_json(ROOT / "branches" / str(b["index"]) / "complete.json") for b in frozen["branches"]}
    print(json.dumps(validate(frozen, branches, read_json(ROOT / "decision.json"),
                             read_json(ROOT / "challenger_update.json"), read_json(ROOT / "completion_profile.json"))))
