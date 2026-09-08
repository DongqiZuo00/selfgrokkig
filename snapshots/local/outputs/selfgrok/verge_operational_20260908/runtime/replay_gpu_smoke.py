"""CPU-only replay of saved job 41369486 completions after extraction repair.

No generation, training, model import, new reward selection, or source-log edits.
The original pre-smoke verifier is replayed alongside the repaired adapter.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "benchmark"), str(ROOT / "runtime")]
import benchmark as b
from generated_budget import verify_journal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def logged_digest(preflight, suffix):
    matches = [v for k, v in preflight["sha256"].items() if k.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"ambiguous/missing original source digest: {suffix}")
    return matches[0]


def main(events_path, parser_path, output):
    if output.exists():
        raise FileExistsError("Use a fresh replay directory, preserving the original smoke")
    original_summary = events_path.parent / "summary.json"
    before = {str(p): digest(p) for p in (events_path, original_summary)}
    journal_records = verify_journal(events_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    preflight = next(e["payload"] for e in events if e["kind"] == "preflight")
    target_path = ROOT / "benchmark/generated/target_train.jsonl"
    previous_path = ROOT / "benchmark/revisions/pre_smoke_41369486.py"
    if digest(target_path) != logged_digest(preflight, "/benchmark/generated/target_train.jsonl"):
        raise ValueError("current train rows differ from the original smoke's frozen data")
    if digest(previous_path) != logged_digest(preflight, "/benchmark/benchmark.py"):
        raise ValueError("pre-smoke code snapshot differs from the actually executed verifier")
    previous = b.load_python(previous_path, "verge_pre_smoke_41369486")
    parser = b.load_python(parser_path, "verge_replay_actual_vendor_parser")
    rows = {row["id"]: row for row in (json.loads(line) for line in target_path.read_text(encoding="utf-8").splitlines())}
    groups = [e for e in events if e["kind"] == "sampled_group"]
    raw_groups = {e["payload"]["group"]: e["payload"] for e in events if e["kind"] == "raw_generation"}
    if len(groups) != 4 or sorted(e["payload"]["group"] for e in groups) != list(range(4)):
        raise ValueError("expected the original four complete sampled groups")
    output.mkdir(parents=True)
    records = []
    for event in groups:
        group = event["payload"]
        row = rows[group["instance_id"]]
        raw = raw_groups[group["group"]]
        assert group["completions"] == raw["completions"]
        assert group["completion_token_ids"] == raw["completion_token_ids"]
        assert row.get("hint") is None and len(group["completions"]) == 8
        for index, (completion, token_ids, saved) in enumerate(zip(group["completions"], group["completion_token_ids"], group["verification"])):
            old = previous.evaluate_program(completion, row["ground_truth"], parser.create_robot_factory)
            new = b.evaluate_program(completion, row["ground_truth"], parser.create_robot_factory)
            for key in ("parse_valid", "reward", "f", "class_rates"):
                assert old[key] == saved[key], (group["group"], index, key)
            assert [x["pass"] for x in old["per_test"]] == [x["pass"] for x in saved["per_test"]]
            assert len(old["per_test"]) == len(new["per_test"]) == 36
            extracted = b.extract_program(completion)
            records.append({"group": group["group"], "rollout": index, "instance_id": row["id"],
                "source_event_sequence": event["sequence"], "source_event_sha256": event["sha256"],
                "raw_completion": completion, "raw_completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "extracted_program": extracted, "extraction_changed_text": extracted != completion,
                "generated_tokens": len(token_ids), "at_original_512_token_cap": len(token_ids) == 512,
                "original_record_exactly_reproduced": True,
                "raw_verifier": old, "extracted_verifier": new})
    assert len(records) == 32
    after = {str(p): digest(p) for p in (events_path, original_summary)}
    assert before == after
    summary = {"status": "complete", "mode": "CPU_replay_of_original_saved_model_outputs",
        "original_job_id": 41369486, "new_model_generations": 0, "new_GPU_usage": False,
        "new_optimizer_steps": 0, "original_events": str(events_path), "original_journal_records_verified": journal_records,
        "source_files_unchanged": before == after, "source_sha256": before,
        "previous_verifier_sha256": digest(previous_path), "repaired_verifier_sha256": digest(Path(b.__file__)),
        "parser_path": str(parser_path), "parser_sha256": digest(parser_path),
        "original_train_data_sha256": digest(target_path), "groups": 4, "rollouts": len(records),
        "original_rewards_reproduced": all(x["original_record_exactly_reproduced"] for x in records),
        "original_generated_tokens": sum(x["generated_tokens"] for x in records),
        "rollouts_at_original_token_cap": sum(x["at_original_512_token_cap"] for x in records),
        "raw_parse_valid": sum(x["raw_verifier"]["parse_valid"] for x in records),
        "extracted_parse_valid": sum(x["extracted_verifier"]["parse_valid"] for x in records),
        "raw_binary_successes": sum(x["raw_verifier"]["reward"] for x in records),
        "extracted_binary_successes": sum(x["extracted_verifier"]["reward"] for x in records),
        "extracted_passing_tests": sum(r["pass"] for x in records for r in x["extracted_verifier"]["per_test"]),
        "total_target_tests_replayed": 32 * 36, "changed_extractions": sum(x["extraction_changed_text"] for x in records),
        "benchmark_regime_established": False, "original_smoke_summary_overwritten": False,
        "output_directory": str(output)}
    save(output / "replay_records.json", records)
    save(output / "replay_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--events", type=Path, default=ROOT / "runs/gpu_smoke_41369486/events.jsonl")
    cli.add_argument("--parser", type=Path, default=Path("/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py"))
    cli.add_argument("--output", type=Path, default=ROOT / "runs" / ("gpu_smoke_41369486_replay_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")))
    args = cli.parse_args()
    main(args.events, args.parser, args.output)
