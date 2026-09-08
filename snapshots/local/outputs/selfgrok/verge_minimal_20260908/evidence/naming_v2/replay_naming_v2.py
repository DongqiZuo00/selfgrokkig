"""CPU replay and names-only pairing audit; never loads a model or held-out results."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_catalogue as stages
import freeze_naming_prompt_v2 as naming


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(value):
    return naming.digest_value(value)


def main(parser, output):
    factory = stages.load_module(parser, "naming_v2_actual_vendor_parser").create_robot_factory if parser else None
    v1_dir = ROOT / "evidence/stage_ignition/job_41412533"
    v2_dir = HERE / "job_41415498"
    audit = stages.load_module(ROOT / "evidence/stage_ignition/analyze_screens.py", "naming_v2_prior_screen_audit")
    prior = audit.analyze("41412533", v1_dir, factory)
    summary = read(v2_dir / "SUMMARY.json")
    frozen = read(v2_dir / "frozen_screen.json")
    progress = read(v2_dir / "screen_progress.json")
    task_file = ROOT / "benchmarks/generated/naming_prompt_v2/screen_tasks.json"
    screen = read(task_file)
    assert screen == frozen["screen"] and digest(task_file) == frozen["task_file_sha256"]
    assert summary["summaries"] == progress and len(progress) == len(screen["tasks"]) == 12
    assert frozen["original_adapter_sha256"] == prior["source_adapter_sha256_at_screen"]
    assert screen["prompt_delta"] == naming.PROMPT_DELTA
    totals, routes, failure_reasons = Counter(), Counter(), Counter()
    groups, programs, paired_checks, grammar_timing_deltas = [], [], [], []
    source_files = [v2_dir / name for name in ("SUMMARY.json", "screen_progress.json", "frozen_screen.json", "one_binary_update_proof.json", "pilot_state.json")]
    for gi, (task, group) in enumerate(zip(screen["tasks"], progress)):
        stage, variant = task["stage_id"], task["variant"]
        assert (stage, variant) == (group["stage_id"], group["variant"])
        row = task["row"]
        original = stages.make_stage_rows(stage, variant)[0]
        assert {k: v for k, v in row.items() if k != "messages"} == {k: v for k, v in original.items() if k != "messages"}
        assert row["messages"][0]["content"] == original["messages"][0]["content"] + "\n\n" + screen["prompt_delta"]
        assert all(stages.TARGET.split_of(c["input"]) == "train" for c in row["ground_truth"])
        path = v2_dir / "samples" / Path(group["raw_path"]).name
        source_files.append(path)
        raw = read(path)
        old = read(v1_dir / "samples" / path.name)
        shared = ("mode", "adapter", "instance_id", "seed", "n", "max_new_tokens", "format_prefix", "sampling_temperature", "top_p", "top_k", "optimizer_steps_before", "decoding_policy")
        assert all(raw[k] == old[k] for k in shared)
        assert raw["seed"] == task["seed"] and raw["max_new_tokens"] == task["cap"] and raw["n"] == 8
        assert raw["adapter"] == prior["source_adapter"] and raw["optimizer_steps_before"] == 0
        old_content, new_content = original["messages"][0]["content"], row["messages"][0]["content"]
        assert old["prompt"].count(old_content) == 1
        assert raw["prompt"] == old["prompt"].replace(old_content, new_content, 1)
        grammar_diff = {key: [old["grammar_metadata"].get(key, "<absent>"), raw["grammar_metadata"].get(key, "<absent>")]
            for key in old["grammar_metadata"].keys() | raw["grammar_metadata"].keys()
            if key not in old["grammar_metadata"] or key not in raw["grammar_metadata"] or old["grammar_metadata"][key] != raw["grammar_metadata"][key]}
        added_metadata = {"mode": "solver_dsl", "consumed_prefix": raw["format_prefix"], "structured_const_expansions": 0, "json_schema_path": None}
        for key, value in added_metadata.items():
            if key in grammar_diff:
                assert key not in old["grammar_metadata"] and raw["grammar_metadata"][key] == value
        assert set(grammar_diff) <= {"startup_seconds", "client_worker_startup_seconds", *added_metadata}, grammar_diff
        grammar_timing_deltas.append({"stage_id": stage, "variant": variant, "differences": grammar_diff})
        token_count = sum(map(len, raw["completion_token_ids"]))
        rewards = [v["reward"] for v in raw["verification"]]
        assert len(rewards) == 8 and set(rewards) <= {0, 1} and rewards == raw["rewards"]
        parsed, successes = sum(v["parse_valid"] for v in raw["verification"]), sum(rewards)
        mixed = 0 < successes < 8
        for key, value in (("generated_tokens", token_count), ("parse_valid", parsed), ("binary_successes", successes), ("mixed_group", mixed)):
            assert group[key] == raw[key] == value
        initial = []
        for ri, (suffix, completion, ids, saved) in enumerate(zip(raw["native_suffixes"], raw["verifier_completions"], raw["completion_token_ids"], raw["verification"])):
            assert completion == raw["format_prefix"] + suffix and len(ids) <= task["cap"]
            assert [c["test_id"] for c in saved["per_test"]] == [c["test_id"] for c in row["ground_truth"]]
            assert [c["input"] for c in saved["per_test"]] == [c["input"] for c in row["ground_truth"]]
            assert saved["reward"] == int(all(c["pass"] for c in saved["per_test"]))
            if factory:
                replay = stages.evaluate_stage(completion, row, factory)
                assert replay["parse_valid"] == saved["parse_valid"] and replay["reward"] == saved["reward"]
                for actual_case, saved_case in zip(replay["per_test"], saved["per_test"]):
                    if actual_case["reason"] == "output_mismatch" and saved_case["reason"] == "exact_output_mismatch":
                        actual_case = {**actual_case, "reason": "exact_output_mismatch"}
                        totals["diagnostic_reason_aliases"] += 1
                    assert actual_case == saved_case
                totals["CPU_replayed_programs"] += 1
                totals["CPU_replayed_cases"] += len(saved["per_test"])
            program = stages.TARGET.extract_program(completion)
            assert program == saved["program"]
            route = re.search(r"START start:\s*\n\s*NEXT\s+(\S+)", program).group(1)
            routes[route] += 1
            initial.append(route)
            reasons = Counter(c["reason"] for c in saved["per_test"] if not c["pass"])
            failure_reasons.update(reasons)
            passed = sum(c["pass"] for c in saved["per_test"])
            totals.update(test_cases=len(saved["per_test"]), per_test_passes=passed,
                at_cap=int(len(ids) == task["cap"]))
            programs.append({"file": path.name, "stage_id": stage, "variant": variant, "sample_index": ri,
                "first_route": route, "parse_valid": saved["parse_valid"], "binary_reward": saved["reward"],
                "passed_cases": passed, "test_cases": len(saved["per_test"]), "failure_reasons": dict(reasons),
                "generated_tokens": len(ids), "program": program})
        assert initial == group["first_routes"]
        totals.update(rollouts=8, generated_tokens=token_count, parse_valid=parsed, binary_successes=successes, mixed_groups=int(mixed))
        groups.append({**group, "raw_sha256": digest(path), "per_test_passes": sum(sum(c["pass"] for c in v["per_test"]) for v in raw["verification"])})
        paired_checks.append({"stage_id": stage, "variant": variant, "source_row_nonmessage_fields_identical": True,
            "actual_chat_prompt_changed_only_by_frozen_paragraph": True, "same_recorded_sampling_policy_and_model": True,
            "same_grammar_hash_and_shared_semantic_metadata": True, "old_prompt_sha256": sha(old["prompt"]), "new_prompt_sha256": sha(raw["prompt"]),
            "ground_truth_sha256": sha(row["ground_truth"]), "grammar_sha256": raw["grammar_metadata"]["grammar_sha256"],
            "prior_raw_sha256": digest(v1_dir / "samples" / path.name), "new_raw_sha256": digest(path)})
    for key in ("rollouts", "generated_tokens", "parse_valid", "binary_successes", "mixed_groups"):
        assert totals[key] == summary[key]
    proof = read(v2_dir / "one_binary_update_proof.json")
    assert summary["update_proof"] == proof and summary["actual_optimizer_steps"] == 1
    assert proof["selected_stage"] == progress[0] and proof["selected_stage"]["mixed_group"]
    assert proof["update"]["tokens_used"] == proof["update"]["nonzero_advantage_tokens"] == groups[0]["generated_tokens"]
    assert proof["update"]["parameters_changed"] and proof["update"]["optimizer_step"] and proof["update"]["binary_reward_only"]
    assert proof["update"]["beta"] == proof["update"]["weight_decay"] == 0
    assert not proof["round_base_changed"]
    state = read(v2_dir / "pilot_state.json")
    assert state["optimizer_steps"] == 1 and state["generated_tokens"] == totals["generated_tokens"]
    assert summary["source_adapter_unchanged"] and not summary["target_transfer_or_VERGE_gain_established"]
    result = {"status": "PASS", "mode": "CPU_replay_saved_names_only_paired_stage_screen" if factory else "saved_names_only_raw_audit",
        "new_GPU_jobs": 0, "new_model_generations": 0, "new_optimizer_steps": 0, "heldout_test_model_results_read": False,
        "v1_job": "41412533", "v2_job": "41415498", "v1_totals": prior["totals"], "v2_totals": dict(totals),
        "v1_first_route_counts": prior["parsed_initial_route_counts"], "v2_first_route_counts": dict(routes),
        "v2_failed_case_reason_counts": dict(failure_reasons), "v2_sampling_seconds": sum(g["sampling_seconds"] for g in groups),
        "paired_checks": paired_checks, "grammar_metadata_differences": grammar_timing_deltas,
        "source_adapter_sha256_at_both_screens": frozen["original_adapter_sha256"], "proof": proof,
        "v2_groups": groups, "v2_program_diagnostics": programs, "target_transfer_established": False,
        "parser_path": str(parser) if parser else None, "parser_sha256": digest(parser) if parser else None,
        "source_sha256": {str(p.relative_to(v2_dir)): digest(p) for p in source_files},
        "task_file_sha256": digest(task_file), "diagnostic_alias": {"catalogue": "output_mismatch", "backend": "exact_output_mismatch"}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "mode", "v1_totals", "v2_totals", "v1_first_route_counts", "v2_first_route_counts", "v2_failed_case_reason_counts")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    main(args.parser, args.output)
