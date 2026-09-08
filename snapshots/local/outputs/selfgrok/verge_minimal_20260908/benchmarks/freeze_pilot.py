"""Freeze the same target/tests under one compact, unhinted pilot prompt view.

This reads benchmark manifests and existing CPU verifier evidence only. It never
reads model results, selects a curriculum, submits work, or changes prior files.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import stage_catalogue as c

HERE = Path(__file__).resolve().parent
VIEW_VERSION = "verge_target_compact_pilot_view_v1_20260908"
SAMPLE_IDS = [f"rollout_{i:02d}" for i in range(32)]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(value):
    """Exactly the canonical JSON hash used by round/round_cli.py."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def frozen_write(path, value):
    """Write reproducibly; an identical rerun is allowed, changed bytes are not."""
    path = Path(path)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"frozen pilot artifact differs: {path}; choose a new view version/path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def contract(row):
    cases = row["ground_truth"]
    if len(cases) != 36 or Counter(x["test_class"] for x in cases) != Counter(dict.fromkeys(c.TARGET.CLASS_IDS, 6)):
        raise ValueError("target view must preserve all 36 tests and six classes")
    if len({case["test_id"] for case in cases}) != 36:
        raise ValueError("duplicate target test identity")
    for case in cases:
        if case["expected_output"] != c.TARGET.expected_output(case["input"]):
            raise ValueError("target expectation changed")
        if case["test_class"] != c.TARGET.test_class(case["input"]):
            raise ValueError("target class changed")
    return {"test_ids": [x["test_id"] for x in cases],
            "class_by_test": {x["test_id"]: x["test_class"] for x in cases},
            "expected_classes": list(c.TARGET.CLASS_IDS)}


def build_view():
    source = c.RELEASE / "benchmark/generated"
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train_path, selection_path = source / "target_train.jsonl", source / "target_selection.jsonl"
    train_hash, selection_hash = file_sha(train_path), file_sha(selection_path)
    if train_hash != source_manifest["split_rows"]["train"]["sha256"] or selection_hash != source_manifest["split_rows"]["selection"]["sha256"]:
        raise ValueError("original datasets no longer match their frozen source manifest")
    train, full_selection = read_jsonl(train_path), read_jsonl(selection_path)
    if len(train) != 128 or len(full_selection) != 64:
        raise ValueError("expected the actual 128-train/64-selection source")
    selection = full_selection[:8]
    full_target_spec = {"operation": "fixed_target", "max_length": 64}
    content = c.COMPACT_SYNTAX + c.task_description(full_target_spec)
    messages = [{"role": "user", "content": content}]
    if any(token in content for token in ("NEXT marker", "NEXT delimiter", "pending_r", "VERGE_TRAIN_HINT")):
        raise ValueError("compact target prompt contains a solution/hint marker")
    target_sets = {"train": train, "selection": selection}
    originals = {row["id"]: copy.deepcopy(row) for rows in target_sets.values() for row in rows}
    views = {split: [dict(copy.deepcopy(row), messages=copy.deepcopy(messages)) for row in rows]
             for split, rows in target_sets.items()}
    all_contracts = {}
    for split, rows in views.items():
        for row in rows:
            original = originals[row["id"]]
            if row.get("hint") is not None:
                raise ValueError("target view must be unhinted")
            if {k: v for k, v in row.items() if k != "messages"} != {k: v for k, v in original.items() if k != "messages"}:
                raise AssertionError("only messages may change in this prompt view")
            all_contracts[row["id"]] = contract(row)
    selection_ids = [row["id"] for row in selection]
    selection_contracts = {sid: all_contracts[sid] for sid in selection_ids}
    manifest_id = c.TARGET.VERSION + ":selection_first8:" + sha({"source_selection_sha256": selection_hash,
        "instance_ids": selection_ids, "test_contracts": selection_contracts})[:16]
    view_payload = {"version": VIEW_VERSION, "source_manifest_sha256": file_sha(source_manifest_path),
        "source_train_manifest_sha256": train_hash, "source_selection_manifest_sha256": selection_hash,
        "manifest_id": manifest_id, "train_instance_ids": [row["id"] for row in train],
        "selection_instance_ids": selection_ids, "messages": messages,
        "ground_truth_sha256_by_instance": {sid: sha(row["ground_truth"]) for sid, row in originals.items()}}
    view_hash = sha(view_payload)
    rows_document = {"manifest_id": manifest_id, "prompt_view_sha256": view_hash,
        "source_train_manifest_sha256": train_hash, "train": views["train"], "selection": views["selection"]}
    metadata = {"version": VIEW_VERSION, "mode": "frozen_prompt_view_no_model_results",
        "source_paths": {"manifest": "../verge_operational_20260908/benchmark/generated/manifest.json",
            "train": "../verge_operational_20260908/benchmark/generated/target_train.jsonl",
            "selection": "../verge_operational_20260908/benchmark/generated/target_selection.jsonl"},
        "source_manifest_sha256": file_sha(source_manifest_path), "source_train_manifest_sha256": train_hash,
        "source_selection_manifest_sha256": selection_hash, "source_benchmark_version": c.TARGET.VERSION,
        "fixed_target_parameters": c.TARGET.PARAMETERS, "input_alphabet": "RB", "input_length_range": [0, 64],
        "train_count": len(train), "selection_count": len(selection), "formal_selection_count": len(full_selection),
        "selection_rule": "first eight rows in frozen source selection file; no result-based filtering",
        "all_original_ids_unchanged": True, "only_messages_changed": True,
        "all_original_ground_truth_unchanged": True, "tests_per_instance": 36,
        "class_ids": list(c.TARGET.CLASS_IDS), "class_denominators": dict.fromkeys(c.TARGET.CLASS_IDS, 6),
        "messages_sha256": sha(messages), "prompt_text_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "original_messages_sha256_by_instance": {sid: sha(row["messages"]) for sid, row in originals.items()},
        "new_messages_sha256_by_instance": dict.fromkeys(originals, sha(messages)),
        "prompt_view_sha256": view_hash, "prompt_view_hash_payload": view_payload,
        "selection": {"manifest_id": manifest_id, "instance_ids": selection_ids, "sample_ids": SAMPLE_IDS,
            "prompt_view_sha256": view_hash, "test_contracts": selection_contracts},
        "source_scripts_sha256": {"benchmarks/freeze_pilot.py": file_sha(Path(__file__)),
                                  "benchmarks/stage_catalogue.py": file_sha(Path(c.__file__))},
        "hint_or_solution_in_prompt": False, "heldout_or_test_model_results_read": False,
        "formal_predictions_or_stage_sources_deleted": False}
    return rows_document, metadata


def stage_validation_evidence():
    """Run the real CPU validator; bind its specs to existing real oracle logs."""
    report_path = HERE / "generated/cpu_oracle_verification.json"
    tests_log = HERE / "cpu_tests.log"
    oracle_log = HERE / "cpu_oracle.log"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "CPU_reference_only_no_model" or report.get("all_pass") is not True:
        raise ValueError("stage catalogue does not have genuine CPU oracle evidence")
    if "Ran 11 tests" not in tests_log.read_text(encoding="utf-8") or "\nOK\n" not in tests_log.read_text(encoding="utf-8"):
        raise ValueError("expected the completed stage catalogue CPU regression log")
    actual_reports = {r["stage_id"]: r for r in report["candidates"]}
    target_def = {"benchmark_version": c.TARGET.VERSION, "parameters": c.TARGET.PARAMETERS,
                  "input_alphabet": "RB", "max_input_length": 64}
    protected, protected_inputs = [], set()
    for split in ("selection", "heldout", "test"):
        for row in read_jsonl(c.RELEASE / "benchmark/generated" / f"target_{split}.jsonl"):
            inputs = [case["input"] for case in row["ground_truth"]]
            protected.append(c.STAGES.instance_tuple_from_manifest(target_def, inputs))
            protected_inputs.update(inputs)
    registry = c.registered_families()
    validations = {}
    for definition in c.DEFINITIONS:
        sid = definition["id"]
        spec = c.stage_spec(sid)
        row = c.make_stage_rows(sid)[0]
        evidence = actual_reports[sid]
        actual_tests = evidence["training_suite"]["per_test"]
        expected_ids = [case["test_id"] for case in row["ground_truth"]]
        if [test["test_id"] for test in actual_tests] != expected_ids or any(test["pass"] != 1 for test in actual_tests):
            raise ValueError("CPU reference evidence does not bind the current stage test identities")
        inputs = [case["input"] for case in row["ground_truth"]]
        if set(inputs) & protected_inputs or any(c.TARGET.split_of(tape) != "train" for tape in inputs):
            raise ValueError("stage input overlaps a protected split")
        validated = c.STAGES.validate_stage(spec, registry=registry, protected_tuples=protected).to_dict()
        if not validated["accepted"]:
            raise ValueError(f"stage validator rejected {sid}: {validated}")
        validated.update(spec_sha256=sha(spec),
            evidence_id=f"benchmarks/generated/pilot_view_v1/stage_validation_evidence.json#{sid}",
            synthetic_fixture=False)
        validated["details"]["cpu_reference_evidence"] = {
            "path": "benchmarks/generated/cpu_oracle_verification.json", "sha256": file_sha(report_path),
            "stage_id": sid, "test_ids": expected_ids, "all_exact_pass": True,
            "oracle_only_not_model_training_or_transfer": True}
        validated["details"]["protected_actual_input_overlap"] = 0
        validations[sid] = validated
    return {"mode": "real_CPU_stage_validation_not_model_effectiveness", "stage_validations": validations,
        "specifications": {d["id"]: c.stage_spec(d["id"]) for d in c.DEFINITIONS},
        "evidence_sources": {"benchmarks/generated/cpu_oracle_verification.json": file_sha(report_path),
            "benchmarks/cpu_tests.log": file_sha(tests_log), "benchmarks/cpu_oracle.log": file_sha(oracle_log)},
        "protected_tuple_count": len(protected), "protected_tuple_sha256": sha(sorted(protected)),
        "hinted_target_certified_by_this_file": False,
        "hinted_target_note": "Requires its own verified train-program provenance and real 512-rollout paired hint-regret probe; CPU oracles do not supply that evidence."}


def freeze(output):
    rows, metadata = build_view()
    stages = stage_validation_evidence()
    output = Path(output)
    frozen_write(output / "target_rows.json", rows)
    metadata["target_rows_file_sha256"] = file_sha(output / "target_rows.json")
    frozen_write(output / "target_view_metadata.json", metadata)
    frozen_write(output / "stage_validation_evidence.json", stages)
    return {"status": "frozen", "output": str(output), "train_count": 128, "selection_count": 8,
            "prompt_view_sha256": metadata["prompt_view_sha256"], "manifest_id": rows["manifest_id"],
            "stage_validation_count": len(stages["stage_validations"]), "model_results_read": False}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=HERE / "generated/pilot_view_v1")
    args = cli.parse_args()
    print(json.dumps(freeze(args.output), indent=2))
