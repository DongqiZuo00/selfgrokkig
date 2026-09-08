"""CPU replay of a real g1 mixed update, constant skip and saved LoRA difference."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAP = HERE / "snapshot_first_update"
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_catalogue as stages
from capture_key_evidence import sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def verify_journal(records):
    previous = "0" * 64
    for index, record in enumerate(records):
        assert record["sequence"] == index and record["previous_sha256"] == previous
        assert sha({k: v for k, v in record.items() if k != "sha256"}) == record["sha256"]
        previous = record["sha256"]


def compare_tensors(base_file, updated_file):
    # Read only adapter tensors; no model construction, inference or GPU work.
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    comparisons = []
    with safe_open(str(base_file), framework="pt", device="cpu") as a, safe_open(str(updated_file), framework="pt", device="cpu") as b:
        assert set(a.keys()) == set(b.keys())
        for key in sorted(a.keys()):
            x, y = a.get_tensor(key), b.get_tensor(key)
            assert x.shape == y.shape and x.dtype == y.dtype
            assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
            changed = not torch.equal(x, y)
            comparisons.append({"key": key, "shape": list(x.shape), "dtype": str(x.dtype), "changed": changed,
                "changed_elements": int(torch.count_nonzero(x != y)), "elements": x.numel(),
                "max_abs_difference": float((x.float() - y.float()).abs().max()) if changed else 0.0})
    assert any(item["changed"] for item in comparisons)
    return {"full_model_constructed": False, "adapter_tensors_read_on_CPU": True, "gpu_used": False,
        "base_file_sha256": digest(base_file), "updated_file_sha256": digest(updated_file),
        "tensor_count": len(comparisons), "changed_tensor_count": sum(x["changed"] for x in comparisons),
        "changed_element_count": sum(x["changed_elements"] for x in comparisons),
        "max_abs_difference": max(x["max_abs_difference"] for x in comparisons), "tensors": comparisons}


def main(parser, output):
    snapshot = read(SNAP / "snapshot.json")
    launch, contract = read(SNAP / "LAUNCH.json"), read(SNAP / "source_runtime_contract.json")
    assert read(SNAP / "runtime_contract.json") == contract
    assert digest(SNAP / "source_runtime_contract.json") == launch["runtime_contract"]["sha256"]
    for path, expected in contract["files"].items():
        assert digest(ROOT / path) == expected, path
    plan = read(SNAP / "plan.json")
    assert digest(SNAP / "plan.json") == launch["plan_file_sha256"]
    assert sha({k: v for k, v in plan.items() if k != "plan_sha256"}) == launch["plan_sha256"] == plan["plan_sha256"]
    cat_path = ROOT / "benchmarks/generated/naming_round_v2/catalogue.json"
    target_path = ROOT / "benchmarks/generated/naming_round_v2/target_rows.json"
    assert digest(cat_path) == launch["catalogue_sha256"] and digest(target_path) == launch["target_rows_sha256"]
    cat = {s["stage_id"]: s for s in read(cat_path)["stages"]}
    train_rows = {r["id"]: r for r in read(target_path)["train"]}
    records = [json.loads(line) for line in (SNAP / "training_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    verify_journal(records)
    assert digest(SNAP / "training_journal.jsonl") == snapshot["journal_snapshot_sha256"]
    factory = stages.load_module(parser, "round3_update_actual_vendor").create_robot_factory
    groups, totals = {}, Counter()
    for name in ("first_mixed", "later_constant"):
        group = snapshot[name]
        if group is None:
            continue
        source, rollout, accounting = group["source"], group["rollout_event"], group["accounting_event"]
        for event in (source, rollout, accounting):
            assert event == records[event["sequence"]]
        raw_path = SNAP / f"{name}.json"
        raw = read(raw_path)
        assert digest(raw_path) == source["payload"]["raw_sha256"]
        phase = next(p for p in plan["branches"]["g1"]["phases"] if p["phase_id"] == source["payload"]["phase"])
        target = source["payload"]["target"]
        if target:
            row = train_rows[raw["instance_id"]]
            kind = "target"
        else:
            stage = cat[phase["stage_id"]]
            proposed = next(s for s in plan["proposal"]["curricula"]["g1"] if s["stage_id"] == stage["stage_id"])
            assert proposed["kind"] == stage["kind"] and proposed["spec"] == stage["spec"]
            assert stage["rows"] == stage["rows_by_variant"]["concise"]
            row = next(r for r in stage["rows"] if r["id"] == raw["instance_id"])
            kind = "stage"
        assert row.get("hint") is None and row["messages"][0]["content"] in raw["prompt"]
        assert all(stages.TARGET.split_of(c["input"]) == "train" for c in row["ground_truth"])
        assert raw["n"] == source["payload"]["sample_count"] == len(rollout["payload"]["records"]) == 8
        assert raw["seed"] == source["payload"]["actual_batch_seed"]
        assert raw["max_new_tokens"] == rollout["payload"]["cap"] and not rollout["payload"]["drain"]
        assert raw["optimizer_steps_before"] == source["payload"]["optimizer_steps_before"]
        assert raw["decoding_policy"] == "dsl_grammar_v1"
        tokens = sum(map(len, raw["completion_token_ids"]))
        assert tokens == raw["generated_tokens"]
        diagnostic = []
        for index, (text, ids, saved, item) in enumerate(zip(raw["verifier_completions"], raw["completion_token_ids"], raw["verification"], rollout["payload"]["records"])):
            assert text == item["raw_completion"] and ids == item["completion_token_ids"]
            assert text == raw["format_prefix"] + raw["native_suffixes"][index]
            assert item["prompt_id"] == row["id"] and item["task_kind"] == kind
            assert item["verifier_digest"] == sha({"ground_truth": row["ground_truth"], "task_kind": kind})
            assert saved["reward"] == item["reward"] == raw["rewards"][index]
            if target:
                replay = stages.TARGET.evaluate_program(text, row["ground_truth"], factory)
                assert replay == saved
            else:
                replay = stages.evaluate_stage(text, row, factory)
                assert replay["reward"] == saved["reward"] and replay["parse_valid"] == saved["parse_valid"]
                for actual, recorded in zip(replay["per_test"], saved["per_test"]):
                    if actual["reason"] == "output_mismatch" and recorded["reason"] == "exact_output_mismatch":
                        actual = {**actual, "reason": "exact_output_mismatch"}
                        totals["diagnostic_reason_aliases"] += 1
                    assert actual == recorded
            assert saved["reward"] == int(all(c["pass"] for c in saved["per_test"]))
            totals.update(programs=1, test_case_executions=len(row["ground_truth"]), binary_successes=saved["reward"],
                parse_valid=saved["parse_valid"], per_test_passes=sum(c["pass"] for c in saved["per_test"]))
            diagnostic.append({"sample_index": index, "reward": saved["reward"], "parse_valid": saved["parse_valid"],
                "program": stages.TARGET.extract_program(text), "per_test": saved["per_test"], "generated_tokens": len(ids)})
        prior_accounting = next(r["payload"] for r in reversed(records[:source["sequence"]]) if r["kind"] == "accounting")
        after = accounting["payload"]
        assert after["name"] == prior_accounting["name"] == phase["phase_id"]
        assert after["generated_tokens"] - prior_accounting["generated_tokens"] == tokens
        assert after["loss_eligible_tokens"] - prior_accounting["loss_eligible_tokens"] == tokens
        actual_update_events = [r for r in records[source["sequence"]:accounting["sequence"] + 1] if r["kind"] == "actual_model_update"]
        if name == "first_mixed":
            assert kind == "stage" and stage["stage_id"] == "identity" and set(raw["rewards"]) == {0, 1}
            assert actual_update_events == [group["update_event"]]
            update = group["update_event"]["payload"]
            metric = update["metric"]
            assert update["raw_path"] == source["payload"]["raw_path"]
            assert metric["tokens_used"] == metric["nonzero_advantage_tokens"] == tokens
            assert metric["optimizer_step"] and metric["parameters_changed"] and metric["binary_reward_only"]
            assert metric["beta"] == metric["weight_decay"] == 0
            assert metric["loss_policy"] == "same_dsl_grammar_masked_native_suffix"
            assert after["optimizer_steps"] == prior_accounting["optimizer_steps"] + 1 == metric["optimizer_steps"]
            assert after["nonzero_advantage_tokens"] - prior_accounting["nonzero_advantage_tokens"] == tokens
            assert after["constant_groups"] == prior_accounting["constant_groups"]
        else:
            assert len(set(raw["rewards"])) == 1 and not actual_update_events and raw["optimizer_steps_before"] == 1
            assert after["optimizer_steps"] == prior_accounting["optimizer_steps"] == 1
            assert after["nonzero_advantage_tokens"] == prior_accounting["nonzero_advantage_tokens"]
            assert after["constant_groups"] == prior_accounting["constant_groups"] + 1
        groups[name] = {"task_kind": kind, "instance_id": row["id"], "rewards": raw["rewards"], "tokens": tokens,
            "optimizer_steps_before": raw["optimizer_steps_before"], "before_accounting": prior_accounting,
            "after_accounting": after, "actual_update_events": actual_update_events, "samples": diagnostic,
            "raw_sha256": digest(raw_path), "source_sequence": source["sequence"], "rollout_sequence": rollout["sequence"]}
    boundary = snapshot["checkpoint_boundary"]
    assert boundary == records[boundary["sequence"]]
    assert boundary["payload"]["optimizer_steps"] == 1 and boundary["payload"]["checkpoint_alias"] == "g1_m1"
    phase_updates = [r for r in records[:boundary["sequence"]] if r["kind"] == "actual_model_update"]
    assert phase_updates == [snapshot["first_mixed"]["update_event"]]
    base_file = Path(plan["base_checkpoint"]) / "adapter_model.safetensors"
    saved_file = Path(boundary["payload"]["checkpoint"]["weights_file"])
    weight_root = Path("/blue/du.j/jinjiaguo/self grok").resolve()
    assert base_file.resolve().is_relative_to(weight_root) and saved_file.resolve().is_relative_to(weight_root)
    assert digest(base_file) == "9596df4348ee3d959661bf451864f2b8dafe454212bd16928a6774eba5f5a321"
    assert digest(saved_file) == boundary["payload"]["checkpoint"]["weights_sha256"]
    weights = compare_tensors(base_file, saved_file)
    result = {"status": "PASS", "mode": "CPU_replay_real_round3_g1_update_and_lora_tensor_audit", "groups": groups,
        "totals": dict(totals), "tensor_comparison": weights, "checkpoint_boundary": boundary,
        "journal_records_verified": len(records), "runtime_contract_files_verified": len(contract["files"]),
        "source_contract_sha256": digest(SNAP / "source_runtime_contract.json"), "plan_sha256": plan["plan_sha256"],
        "parser_sha256": digest(parser), "base_weights_unchanged": True, "new_training_performed_by_audit": False,
        "constant_group_claim": "journal/accounting and frozen skip implementation; no per-group parameter snapshot exists for the constant group",
        "target_transfer_established": False, "heldout_test_model_results_read": False, "new_GPU_jobs": 0}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "totals": result["totals"], "mixed_tokens": groups["first_mixed"]["tokens"],
        "constant_tokens": groups.get("later_constant", {}).get("tokens"),
        "weights": {k: v for k, v in weights.items() if k != "tensors"}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    main(args.parser, args.output)
