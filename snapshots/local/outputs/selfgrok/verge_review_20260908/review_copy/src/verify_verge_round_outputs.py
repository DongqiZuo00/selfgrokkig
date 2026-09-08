"""Validate the completed round's recorded artifacts; no model sampling or hash scans."""
import json
from common import EXP_ROOT, read_json, read_jsonl
from verge_round_core import ROOT, VERSION, parse_proposal
from verge_round_decision import decide


def check_profile(endpoint, path, expected):
    raw = read_jsonl(path)
    vectors = endpoint["keyed_vectors"]
    assert len(raw) == len(vectors) == endpoint["rollouts"] == expected
    keys = {f"{r['instance_id']}::{r['rollout_index']}" for r in raw}
    assert keys == set(vectors)
    counts = [sum(v[i] for v in vectors.values()) for i in range(11)]
    assert counts == endpoint["counts"]
    assert [c / expected for c in counts] == endpoint["rates"]
    assert all(all(a >= b for a, b in zip(v, v[1:])) for v in vectors.values())
    assert sum(r["reward"] for r in raw) == counts[-1]


def main():
    marker = read_json(EXP_ROOT / "manifests" / f"{VERSION}_complete.json")
    assert marker["rounds_completed"] == 1 and not marker["official_test_opened"]
    for relative in marker["artifacts"]:
        path = EXP_ROOT / relative
        assert path.is_file() and path.stat().st_size > 0
        if path.suffix == ".json":
            read_json(path)
    frozen = read_json(ROOT / "round_frozen.json")
    cfg = frozen["config"]
    assert cfg["backbone"] == "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    assert len(frozen["generation"]["candidates"]) == 3
    assert not cfg["official_test_opened"]
    check_profile(frozen["initial"]["target"], ROOT / "initial/target.jsonl", 512)
    branches = {}
    brief = []
    for b in frozen["branches"]:
        directory = ROOT / "branches" / str(b["index"])
        completed = read_json(directory / "complete.json")
        train = completed["training"]
        assert train["train_tokens"] == cfg["train_tokens_per_branch"] == 131072
        assert train["update"] <= 100
        assert completed["binary_rewards_only"] and completed["fresh_optimizer"]
        assert completed["replay_loss"] == completed["weight_decay"] == 0
        assert completed["kl_reference"] == cfg["solver_start"]
        sums = {"tokens_used": 0, "nonzero_advantage_tokens": 0, "optimizer_step": 0}
        target_successes, total_reward = 0, 0
        for i in range(1, train["update"] + 1):
            raw = read_jsonl(directory / "train" / f"update_{i:04d}.jsonl")
            summary = read_json(directory / "train" / f"update_{i:04d}_summary.json")
            assert len(raw) == 64 and set(r["reward"] for r in raw) <= {0, 1}
            assert sum(r["reward"] for r in raw) == summary["successes"]
            for k in sums:
                sums[k] += summary[k]
            target_successes += summary["target_successes"]
            total_reward += summary["successes"]
        assert sums["tokens_used"] == train["train_tokens"]
        assert sums["nonzero_advantage_tokens"] == train["nonzero_advantage_tokens"]
        assert sums["optimizer_step"] == train["optimizer_steps"]
        assert target_successes == train["target_successes"]
        check_profile(completed["target"], directory / "target.jsonl", 512)
        assert len(read_jsonl(directory / "scope.jsonl")) == 128
        branches[b["index"]] = completed
        brief.append({"index": b["index"], "updates": train["update"], "optimizer_steps": train["optimizer_steps"],
            "training_full_successes_all_tasks": total_reward, "training_target_successes": target_successes,
            "target_counts": completed["target"]["counts"]})
    decision = read_json(ROOT / "decision.json")
    assert decision == decide(frozen, branches)
    teacher = read_json(ROOT / "challenger_update.json")
    assert teacher["rewards"] == decision["challenger_rewards"]
    assert teacher["whole_proposal_advantages"] == decision["challenger_advantages"]
    assert teacher["checkpoint"] == marker["challenger_checkpoint"]
    assert decision["selected"]["checkpoint"] == marker["solver_checkpoint"]
    for relative in (marker["solver_checkpoint"], marker["challenger_checkpoint"]):
        checkpoint = EXP_ROOT / relative
        assert (checkpoint / "adapter_model.safetensors").is_file()
        assert "qwen" not in (checkpoint / "adapter_config.json").read_text().lower()
    completion = read_json(ROOT / "completion_profile.json")
    check_profile(completion, ROOT / "completion.jsonl", 64)
    print(json.dumps({"validated": True, "initial_target_counts": frozen["initial"]["target"]["counts"],
        "branches": brief, "completion_full_pass": completion["counts"][-1],
        "proposals": [{"index": p["index"], "valid": p["valid"], "text": p["text"],
            "error": p.get("validation_error")} for p in frozen["generation"]["candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
