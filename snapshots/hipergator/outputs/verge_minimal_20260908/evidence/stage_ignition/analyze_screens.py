"""Read-only audit/CPU replay of two saved train-stage screens; no model loading."""
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


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def analyze(job, directory, create_factory=None):
    summary = read(directory / "SUMMARY.json")
    frozen = read(directory / "frozen_screen.json")
    assert summary["summaries"] == read(directory / "screen_progress.json")
    assert len(summary["summaries"]) == len(frozen["tasks"]) == 12
    files = [directory / name for name in ("SUMMARY.json", "frozen_screen.json", "screen_progress.json")]
    totals = Counter()
    first_targets, parse_errors, test_failures = Counter(), Counter(), Counter()
    groups, examples = [], []
    for group_index, group in enumerate(summary["summaries"]):
        stage, variant = frozen["tasks"][group_index]
        assert (stage, variant) == (group["stage_id"], group["variant"])
        path = directory / "samples" / Path(group["raw_path"]).name
        files.append(path)
        raw = read(path)
        row = stages.make_stage_rows(stage, variant)[0]
        assert raw["instance_id"] == row["id"] == group["instance_id"]
        assert raw["n"] == frozen["samples_per_task"] == 8
        assert raw["max_new_tokens"] == frozen["caps"].get(stage, frozen["caps"]["default"])
        assert raw["optimizer_steps_before"] == 0
        assert raw["adapter"] == frozen["source_solver"]
        assert all(stages.TARGET.split_of(c["input"]) == "train" for c in row["ground_truth"])
        assert raw["sampling_temperature"] == raw["top_p"] == 1.0 and raw["top_k"] == 0
        assert len(raw["completion_token_ids"]) == len(raw["native_suffixes"]) == len(raw["verifier_completions"]) == len(raw["verification"]) == 8
        token_count = sum(len(x) for x in raw["completion_token_ids"])
        parsed = sum(x["parse_valid"] for x in raw["verification"])
        rewards = [x["reward"] for x in raw["verification"]]
        assert set(rewards) <= {0, 1} and rewards == raw["rewards"]
        successes = sum(rewards)
        mixed = 0 < successes < 8
        for key, value in (("generated_tokens", token_count), ("parse_valid", parsed), ("binary_successes", successes), ("mixed_group", mixed)):
            assert raw[key] == group[key] == value, (job, stage, key)
        per_test_passes = 0
        for index, (suffix, completion, saved, token_ids) in enumerate(zip(raw["native_suffixes"], raw["verifier_completions"], raw["verification"], raw["completion_token_ids"])):
            assert raw["format_prefix"] + suffix == completion
            assert len(token_ids) <= raw["max_new_tokens"]
            assert [x["test_id"] for x in saved["per_test"]] == [x["test_id"] for x in row["ground_truth"]]
            assert [x["input"] for x in saved["per_test"]] == [x["input"] for x in row["ground_truth"]]
            assert saved["reward"] == int(all(x["pass"] for x in saved["per_test"]))
            if create_factory is not None:
                replay = stages.evaluate_stage(completion, row, create_factory)
                assert replay["parse_valid"] == saved["parse_valid"]
                assert replay["reward"] == saved["reward"]
                # hf_backend.py uses exact_output_mismatch; stage_catalogue uses
                # output_mismatch. Record this diagnostic-name alias explicitly;
                # test IDs, inputs, pass bits and execution steps must be exact.
                for actual_case, saved_case in zip(replay["per_test"], saved["per_test"]):
                    if actual_case["reason"] == "output_mismatch" and saved_case["reason"] == "exact_output_mismatch":
                        actual_case = {**actual_case, "reason": "exact_output_mismatch"}
                        totals["diagnostic_reason_aliases"] += 1
                    assert actual_case == saved_case, (job, stage, index, actual_case, saved_case)
                totals["CPU_replayed_programs"] += 1
                totals["CPU_replayed_test_cases"] += len(saved["per_test"])
            program = stages.TARGET.extract_program(completion)
            assert program == saved["program"]
            route = re.search(r"\bSTART\s+([^:\s]+):\s*\n\s*NEXT\s+(\S+)", program)
            first_target = route.group(2) if route else "<unrecognized>"
            if saved["parse_valid"]:
                first_targets[first_target] += 1
            elif saved["per_test"]:
                error = re.sub(r"Line \d+: ", "", saved["per_test"][0]["reason"] or "<no reason>")
                # Keep exact messages in saved evidence; aggregate by parser category here.
                parse_errors[error.split(":", 1)[0]] += 1
            reasons = Counter()
            for case in saved["per_test"]:
                if case["pass"]:
                    totals["per_test_passes"] += 1
                    per_test_passes += 1
                else:
                    reasons[case["reason"] or "<no reason>"] += 1
                    if saved["parse_valid"]:
                        test_failures[case["reason"] or "<no reason>"] += 1
            totals["at_cap"] += int(len(token_ids) == raw["max_new_tokens"])
            totals["test_cases"] += len(saved["per_test"])
            examples.append({"file": path.name, "stage_id": stage, "variant": variant, "sample_index": index,
                "generated_tokens": len(token_ids), "parse_valid": saved["parse_valid"], "binary_reward": saved["reward"],
                "first_target": first_target, "passed_cases": sum(x["pass"] for x in saved["per_test"]),
                "case_count": len(saved["per_test"]), "failure_reasons": dict(reasons), "program": program})
        totals.update(rollouts=8, generated_tokens=token_count, parse_valid=parsed, binary_successes=successes, mixed_groups=int(mixed))
        groups.append({"stage_id": stage, "variant": variant, "rollouts": 8, "parse_valid": parsed,
            "binary_successes": successes, "mixed_group": mixed, "generated_tokens": token_count,
            "per_test_passes": per_test_passes, "seed": raw["seed"], "sampling_seconds": raw["sampling_seconds"],
            "prompt_sha256": text_digest(raw["prompt"]), "prompt_token_ids_sha256": text_digest(json.dumps(raw["prompt_token_ids"])),
            "max_new_tokens": raw["max_new_tokens"], "raw_file": path.name, "raw_sha256": digest(path)})
    assert totals["rollouts"] == summary["rollouts"] == 96
    assert totals["generated_tokens"] == summary["generated_tokens"]
    assert totals["binary_successes"] == summary["total_stage_successes"]
    assert totals["mixed_groups"] == summary["mixed_stage_groups"]
    proof = summary["update_proof"]
    if proof is not None:
        assert proof == read(directory / "one_binary_update_proof.json")
        state = read(directory / "pilot_state.json")
        assert state["optimizer_steps"] == summary["actual_optimizer_steps"] == 1
        assert state["generated_tokens"] == totals["generated_tokens"]
        selected = next(g for g in groups if g["stage_id"] == proof["selected_stage"]["stage_id"] and g["variant"] == proof["selected_stage"]["variant"])
        assert selected["mixed_group"]
        update = proof["update"]
        assert update["tokens_used"] == update["nonzero_advantage_tokens"] == selected["generated_tokens"]
        assert update["optimizer_step"] and update["parameters_changed"] and update["binary_reward_only"]
        assert update["beta"] == update["weight_decay"] == 0
        assert not proof["round_base_changed"]
        files += [directory / "one_binary_update_proof.json", directory / "pilot_state.json"]
    else:
        assert summary["actual_optimizer_steps"] == 0 and totals["mixed_groups"] == 0
    assert summary["source_adapter_unchanged"] and not summary["target_transfer_or_VERGE_gain_established"]
    return {"job_id": job, "totals": dict(totals), "optimizer_steps": summary["actual_optimizer_steps"],
        "source_adapter_sha256_at_screen": frozen["source_adapter_sha256"], "source_adapter": frozen["source_solver"],
        "source_sha256": {str(p.relative_to(directory)): digest(p) for p in files},
        "decoding_policy": frozen.get("decoding_policy"), "sampling_seconds": sum(g["sampling_seconds"] for g in groups),
        "parsed_initial_route_counts": dict(first_targets), "parse_error_category_counts": dict(parse_errors),
        "parsed_program_failed_test_reason_counts": dict(test_failures), "groups": groups, "program_diagnostics": examples,
        "update_proof": proof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", type=Path, default=HERE / "job_41410864")
    ap.add_argument("--run-b", type=Path, default=HERE / "job_41412533")
    ap.add_argument("--parser", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    factory = stages.load_module(args.parser, "stage_screen_replay_vendor_parser").create_robot_factory if args.parser else None
    a, b = analyze("41410864", args.run_a, factory), analyze("41412533", args.run_b, factory)
    assert a["source_adapter_sha256_at_screen"] == b["source_adapter_sha256_at_screen"]
    for ga, gb in zip(a["groups"], b["groups"]):
        for key in ("stage_id", "variant", "rollouts", "seed", "prompt_sha256", "prompt_token_ids_sha256", "max_new_tokens"):
            assert ga[key] == gb[key], key
    report = {"status": "PASS", "mode": "CPU_replay_saved_train_stage_outputs" if factory else "saved_train_stage_raw_record_audit",
        "model_loaded": False, "GPU_used": False, "new_optimizer_steps": 0,
        "heldout_or_test_model_results_read": False, "target_transfer_established": False,
        "matched_inputs_prompts_seeds_caps_initial_weights": True, "jobs": [a, b],
        "parser_path": str(args.parser) if args.parser else None, "parser_sha256": digest(args.parser) if args.parser else None,
        "stage_catalogue_sha256": digest(Path(stages.__file__)),
        "diagnostic_reason_alias": {"stage_catalogue": "output_mismatch", "hf_backend": "exact_output_mismatch"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode": report["mode"], "jobs": [{"job_id": x["job_id"], "totals": x["totals"], "optimizer_steps": x["optimizer_steps"]} for x in (a, b)]}, indent=2))


if __name__ == "__main__":
    main()
