"""Audit five immutable training groups without loading models or evaluations."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_catalogue as stages
import select_groups


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(parser, output):
    factory = stages.load_module(parser, "round2_training_actual_vendor").create_robot_factory if parser else None
    selection = read(HERE / "selection.json")
    cat_path = ROOT / "benchmarks/generated/naming_round_v2/catalogue.json"
    target_path = ROOT / "benchmarks/generated/naming_round_v2/target_rows.json"
    catalogue = read(cat_path)
    by_stage = {stage["stage_id"]: stage for stage in catalogue["stages"]}
    plan = read(HERE / "plan.json")
    assert select_groups.sha({k: v for k, v in plan.items() if k != "plan_sha256"}) == plan["plan_sha256"]
    probe_rows = {}
    for path in (ROOT / "evidence/naming_v2/job_41415498/samples").glob("*.json"):
        sample = read(path)
        probe_rows[sample["instance_id"]] = sample
    snapshots, launch_checks, journals = {}, {}, {}
    for branch in ("g1", "g2"):
        _, snapshot = select_groups.snapshot(branch)
        assert snapshot == selection["snapshots"][branch]
        snapshots[branch] = snapshot
        launch = read(HERE / branch / "LAUNCH.json")
        contract = read(HERE / branch / "runtime_contract.json")
        assert launch["branch"] == branch and launch["plan_sha256"] == plan["plan_sha256"]
        assert launch["plan_file_sha256"] == digest(HERE / "plan.json")
        assert launch["catalogue_sha256"] == digest(cat_path) == contract["files"]["benchmarks/generated/naming_round_v2/catalogue.json"]
        assert launch["target_rows_sha256"] == digest(target_path) == contract["files"]["benchmarks/generated/naming_round_v2/target_rows.json"]
        original_contract = ROOT / "configs" / Path(launch["runtime_contract"]["path"]).name
        assert launch["runtime_contract"]["sha256"] == digest(original_contract)
        # The launcher writes a JSON copy with LF; its source file used CRLF.
        # Verify the original byte hash and exact parsed copy separately.
        assert read(original_contract) == contract
        for file, expected in contract["files"].items():
            assert digest(ROOT / file) == expected, file
        launch_checks[branch] = {"catalogue_sha256": digest(cat_path), "target_rows_sha256": digest(target_path),
            "all_runtime_contract_file_hashes_match": True, "contract_files_checked": len(contract["files"]),
            "launch_sha256": digest(HERE / branch / "LAUNCH.json"), "plan_file_sha256": digest(HERE / "plan.json"),
            "slurm_job_id_reported_by_launch": launch["slurm_job_id"],
            "source_runtime_contract_sha256": digest(original_contract),
            "serialized_contract_copy_sha256": digest(HERE / branch / "runtime_contract.json"),
            "serialized_contract_content_identical": True}
        journals[branch] = [json.loads(line) for line in (HERE / branch / "training_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    groups, diagnostics, totals = [], [], Counter()
    for chosen in selection["selected"]:
        branch = chosen["branch"]
        raw_path = HERE / branch / chosen["filename"]
        raw = read(raw_path)
        source = chosen["source"]
        assert digest(raw_path) == source["payload"]["raw_sha256"]
        assert source == journals[branch][source["sequence"]]
        event = journals[branch][chosen["rollouts_sequence"]]
        assert event["kind"] == "rollouts"
        phase = next(p for p in plan["branches"][branch]["phases"] if p["phase_id"] == chosen["phase"])
        stage = by_stage[phase["stage_id"]]
        proposed = next(s for s in plan["proposal"]["curricula"][branch] if s["stage_id"] == stage["stage_id"])
        assert proposed["kind"] == stage["kind"] and proposed["spec"] == stage["spec"]
        assert stage["rows"] == stage["rows_by_variant"]["concise"]
        row = next(row for row in stage["rows"] if row["id"] == raw["instance_id"])
        assert raw["instance_id"] == chosen["instance_id"] and raw["instance_id"].endswith("__compact")
        assert raw["prompt"] == probe_rows[row["id"]]["prompt"]
        assert row["messages"][0]["content"] in raw["prompt"]
        assert all(r["messages"] != row["messages"] for r in stage["rows_by_variant"]["official"])
        assert raw["prompt"].count(catalogue["prompt_delta"]) == 1 and row.get("hint") is None
        assert raw["seed"] == source["payload"]["actual_batch_seed"]
        assert raw["n"] == source["payload"]["sample_count"] == chosen["n"] == 8
        assert raw["max_new_tokens"] == event["payload"]["cap"]
        assert not source["payload"]["target"] and not event["payload"]["drain"]
        assert raw["optimizer_steps_before"] == source["payload"]["optimizer_steps_before"]
        assert raw["decoding_policy"] == "dsl_grammar_v1"
        assert raw["generated_tokens"] == sum(map(len, raw["completion_token_ids"]))
        journal_items = event["payload"]["records"]
        expected_verifier = select_groups.sha({"ground_truth": row["ground_truth"], "task_kind": "stage"})
        reward_sum, parsed, test_passes, reasons = 0, 0, 0, Counter()
        for index, (text, ids, saved, item) in enumerate(zip(raw["verifier_completions"], raw["completion_token_ids"], raw["verification"], journal_items)):
            assert item["raw_completion"] == text and item["completion_token_ids"] == ids
            assert item["reward"] == saved["reward"] == raw["rewards"][index]
            assert item["verifier_digest"] == expected_verifier and item["prompt_id"] == row["id"] and item["task_kind"] == "stage"
            assert text == raw["format_prefix"] + raw["native_suffixes"][index]
            assert [c["test_id"] for c in saved["per_test"]] == [c["test_id"] for c in row["ground_truth"]]
            assert saved["reward"] == int(all(c["pass"] for c in saved["per_test"]))
            if factory:
                actual = stages.evaluate_stage(text, row, factory)
                assert actual["parse_valid"] == saved["parse_valid"] and actual["reward"] == saved["reward"]
                for new_case, old_case in zip(actual["per_test"], saved["per_test"]):
                    if new_case["reason"] == "output_mismatch" and old_case["reason"] == "exact_output_mismatch":
                        new_case = {**new_case, "reason": "exact_output_mismatch"}
                        totals["diagnostic_reason_aliases"] += 1
                    assert new_case == old_case
                totals["CPU_replayed_programs"] += 1
                totals["CPU_replayed_case_executions"] += len(row["ground_truth"])
            reward_sum += saved["reward"]
            parsed += saved["parse_valid"]
            test_passes += sum(c["pass"] for c in saved["per_test"])
            failure = Counter(c["reason"] for c in saved["per_test"] if not c["pass"])
            reasons.update(failure)
            diagnostics.append({"branch": branch, "file": chosen["filename"], "stage_id": stage["stage_id"], "sample_index": index,
                "parse_valid": saved["parse_valid"], "reward": saved["reward"], "failure_reasons": dict(failure),
                "program": saved["program"], "generated_tokens": len(ids)})
        assert reward_sum == raw["binary_successes"] == chosen["reward_sum"]
        assert parsed == raw["parse_valid"] and raw["mixed_group"] == (0 < reward_sum < raw["n"])
        totals.update(groups=1, programs=8, generated_tokens=raw["generated_tokens"], parse_valid=parsed,
            binary_successes=reward_sum, test_passes=test_passes, test_cases=len(row["ground_truth"]) * 8)
        groups.append({"branch": branch, "phase": chosen["phase"], "file": chosen["filename"], "stage_id": stage["stage_id"],
            "instance_id": row["id"], "n": 8, "cap": raw["max_new_tokens"], "generated_tokens": raw["generated_tokens"],
            "parse_valid": parsed, "binary_successes": reward_sum, "per_test_passes": test_passes,
            "per_test_total": len(row["ground_truth"]) * 8, "failure_reasons": dict(reasons),
            "actual_prompt_exactly_matches_frozen_concise_naming_probe": True, "official_prompt_used": False,
            "message_sha256": select_groups.sha(row["messages"]), "ground_truth_sha256": select_groups.sha(row["ground_truth"]),
            "raw_sha256": digest(raw_path), "journal_generation_source_sha256": source["sha256"],
            "journal_rollouts_sha256": event["sha256"]})
    result = {"status": "PASS", "mode": "CPU_vendor_actual_training_group_replay" if factory else "actual_training_group_raw_audit",
        "scope": "five completed training groups from fixed g1/g2 journal snapshots, not complete round results",
        "new_GPU_jobs": 0, "new_model_samples": 0, "new_optimizer_steps": 0, "heldout_test_model_results_read": False,
        "snapshots": snapshots, "launch_contract_checks": launch_checks, "totals": dict(totals), "groups": groups,
        "program_diagnostics": diagnostics, "parser_path": str(parser) if parser else None,
        "parser_sha256": digest(parser) if parser else None, "selection_sha256": digest(HERE / "selection.json")}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "totals": result["totals"], "groups": [{k: g[k] for k in ("branch", "stage_id", "n", "cap", "generated_tokens", "parse_valid", "binary_successes", "per_test_passes", "per_test_total")} for g in groups]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    main(args.parser, args.output)
