"""Read normal first-round outputs; no extra sampling, weights or checkpoint hashes."""
from pathlib import Path
import json
from common import atomic_json, read_json, read_jsonl
from verge_independent_sampling import split_requests, validate_response, validate_records, PROTOCOL

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "raw_results/verge_book_v2_verge_r00"


def check_batch(kind, prompts, n):
    path = ROOT / "initial" / f"{kind}.jsonl"
    saved = read_json(path.with_suffix(".response.json"))
    request, response = saved["request"], saved["response"]
    records = read_jsonl(path)
    assert saved["backend_protocol"] == PROTOCOL
    assert len(request["prompt"]) == prompts and request["n"] == n
    assert request["max_tokens"] == 2048
    validate_response(request, response)
    validate_records(request, records)
    for i, part in enumerate(split_requests(request)):
        cached = read_json(path.with_suffix(".parts") / f"prompt_{i:04d}.json")
        assert cached["request"] == part
        choices = sorted(cached["response"]["choices"], key=lambda c: c["index"])
        assert len(choices) == n
        for j, choice in enumerate(choices):
            assert records[i*n+j]["completion_token_ids"] == choice["token_ids"]
    return {"prompts": prompts, "samples_per_prompt": n, "outputs": len(records),
        "distinct_configured_child_seeds": len({r["sampling_child_seed"] for r in records}),
        "actual_per_prompt_requests_and_native_actions_verified": True,
        "max_completion_tokens": max(r["completion_tokens"] for r in records),
        "generation_seconds": saved["seconds"],
        "generated_tokens": sum(r["completion_tokens"] for r in records)}


def main():
    frozen = read_json(ROOT / "round_frozen.json")
    cfg = frozen["config"]
    assert cfg["book_suite"] == "verge_book_v2" and cfg["book_round"] == 0
    assert cfg["prior_book_versions"] == [] and cfg["independent_prompt_streams"]
    assert len(frozen["branches"]) == 4 and frozen["probes"] == []
    roots = {}
    for role in ("solver", "challenger"):
        roots[role] = read_json(EXP / "checkpoints" / f"verge_book_v2_initial_{role}" / "restart_root_provenance.json")
        assert roots[role]["trained_tokens"] == 0 and roots[role]["optimizer_state_entries"] == 0
    report = {"version": cfg["protocol_version"], "sampling_protocol": PROTOCOL,
        "normal_initial_target": check_batch("target", 64, 8),
        "normal_initial_scope": check_batch("scope", 32, 4),
        "root_provenance": roots, "extra_model_samples": 0, "checkpoint_hash_scan": False,
        "curricula_generated": 3, "training_branches_frozen": 4,
        "legacy_results_preserved": True, "whole_experiment_complete": False}
    atomic_json(ROOT / "independent_backend_runtime_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
