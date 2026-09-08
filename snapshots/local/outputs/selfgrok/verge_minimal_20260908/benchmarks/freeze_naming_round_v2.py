"""Freeze the same round inputs with the already frozen generic naming paragraph.

CPU only. Reads v1 input contracts and CPU stage evidence; no model results,
sampling, training, proposal selection, runtime changes, or GPU submission.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import freeze_pilot as pilot

HERE = Path(__file__).resolve().parent
VERSION = "verge_naming_round_prompt_view_v2_20260908"
CATALOGUE_VERSION = "verge_stage_catalogue_naming_view_v2_20260908"
V1 = HERE / "generated/pilot_view_v1"
NAMES = HERE / "generated/naming_prompt_v2/screen_tasks.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def changed_row(original, delta):
    row = copy.deepcopy(original)
    users = [m for m in row["messages"] if m["role"] == "user"]
    if not users:
        raise ValueError("expected an existing user prompt")
    for message in users:
        if delta in message["content"]:
            raise ValueError("naming paragraph is already present; do not append twice")
        message["content"] += "\n\n" + delta
    if {k: v for k, v in row.items() if k != "messages"} != {k: v for k, v in original.items() if k != "messages"}:
        raise AssertionError("only messages may change")
    if row.get("hint") is not None:
        raise ValueError("this names-only view must remain unhinted")
    return row


def build():
    original_rows = read(V1 / "target_rows.json")
    original_metadata = read(V1 / "target_view_metadata.json")
    original_catalogue = read(HERE / "generated/catalogue.json")
    original_evidence = read(V1 / "stage_validation_evidence.json")
    names = read(NAMES)
    delta = names["prompt_delta"]
    if read(NAMES.with_name("metadata.json"))["screen_tasks_sha256"] != pilot.sha(names):
        raise ValueError("names-only screen treatment no longer matches its frozen identity")
    if pilot.file_sha(V1 / "target_rows.json") != original_metadata["target_rows_file_sha256"]:
        raise ValueError("v1 target rows no longer match their frozen identity")
    if pilot.sha(original_metadata["prompt_view_hash_payload"]) != original_rows["prompt_view_sha256"]:
        raise ValueError("v1 target view identity is invalid")
    if original_rows["source_train_manifest_sha256"] != pilot.file_sha(pilot.c.RELEASE / "benchmark/generated/target_train.jsonl"):
        raise ValueError("original target train source changed")
    sources = {
        "benchmarks/generated/pilot_view_v1/target_rows.json": pilot.file_sha(V1 / "target_rows.json"),
        "benchmarks/generated/pilot_view_v1/target_view_metadata.json": pilot.file_sha(V1 / "target_view_metadata.json"),
        "benchmarks/generated/pilot_view_v1/stage_validation_evidence.json": pilot.file_sha(V1 / "stage_validation_evidence.json"),
        "benchmarks/generated/catalogue.json": pilot.file_sha(HERE / "generated/catalogue.json"),
        "benchmarks/generated/naming_prompt_v2/screen_tasks.json": pilot.file_sha(NAMES),
        "benchmarks/freeze_naming_round_v2.py": pilot.file_sha(Path(__file__)),
    }
    rows = copy.deepcopy(original_rows)
    for split, count in (("train", 128), ("selection", 8)):
        if len(rows[split]) != count:
            raise ValueError("expected the frozen 128 train / first 8 selection inputs")
        rows[split] = [changed_row(r, delta) for r in original_rows[split]]
        for row in rows[split]:
            pilot.contract(row)
    manifest_id = pilot.c.TARGET.VERSION + ":selection_first8:naming_v2:" + pilot.sha({
        "parent_manifest_id": original_rows["manifest_id"], "parent_prompt_view_sha256": original_rows["prompt_view_sha256"],
        "version": VERSION, "prompt_delta": delta})[:16]
    payload = copy.deepcopy(original_metadata["prompt_view_hash_payload"])
    payload.update(version=VERSION, manifest_id=manifest_id, messages=rows["train"][0]["messages"],
        parent_prompt_view_sha256=original_rows["prompt_view_sha256"],
        naming_screen_file_sha256=pilot.file_sha(NAMES), prompt_delta_sha256=pilot.sha(delta))
    view_hash = pilot.sha(payload)
    rows.update(manifest_id=manifest_id, prompt_view_sha256=view_hash)
    metadata = copy.deepcopy(original_metadata)
    metadata.update(version=VERSION, mode="frozen_names_only_round_view_no_model_results",
        source_view_version=original_metadata["version"], source_view_manifest_id=original_rows["manifest_id"],
        source_prompt_view_sha256=original_rows["prompt_view_sha256"], source_view_files_sha256=sources,
        prompt_delta=delta, prompt_delta_source_version=names["version"],
        prompt_view_hash_payload=payload, prompt_view_sha256=view_hash,
        messages_sha256=pilot.sha(payload["messages"]),
        prompt_text_sha256=__import__("hashlib").sha256(payload["messages"][0]["content"].encode("utf-8")).hexdigest(),
        original_messages_sha256_by_instance={r["id"]: pilot.sha(r["messages"]) for split in ("train", "selection") for r in original_rows[split]},
        new_messages_sha256_by_instance={r["id"]: pilot.sha(r["messages"]) for split in ("train", "selection") for r in rows[split]},
        task_and_test_contracts_unchanged=True, target_original_train_manifest_sha256_preserved=True,
        target_v1_files_modified=False, reference_or_solution_added=False,
        source_scripts_sha256={"benchmarks/freeze_naming_round_v2.py": pilot.file_sha(Path(__file__))})
    metadata["selection"].update(manifest_id=manifest_id, prompt_view_sha256=view_hash)
    metadata.pop("target_rows_file_sha256")

    catalogue = copy.deepcopy(original_catalogue)
    stage_messages = {}
    for stage, original in zip(catalogue["stages"], original_catalogue["stages"]):
        stage["rows"] = [changed_row(r, delta) for r in original["rows"]]
        stage["rows_by_variant"] = {variant: [changed_row(r, delta) for r in rs]
                                    for variant, rs in original["rows_by_variant"].items()}
        if stage["spec"] != original["spec"] or stage["kind"] != original["kind"]:
            raise AssertionError("stage semantics changed")
        stage_messages[stage["stage_id"]] = {"rows": [pilot.sha(r["messages"]) for r in stage["rows"]],
            "rows_by_variant": {variant: [pilot.sha(r["messages"]) for r in rs] for variant, rs in stage["rows_by_variant"].items()}}
    stage_payload = {"version": CATALOGUE_VERSION, "source_catalogue_file_sha256": sources["benchmarks/generated/catalogue.json"],
        "naming_screen_file_sha256": pilot.file_sha(NAMES), "prompt_delta": delta,
        "messages_sha256_by_stage": stage_messages}
    stage_view_hash = pilot.sha(stage_payload)
    catalogue.update(version=CATALOGUE_VERSION, source_catalogue_version=original_catalogue["version"],
        prompt_view_sha256=stage_view_hash, prompt_delta=delta, source_sha256=sources)
    metadata.update(stage_prompt_view_sha256=stage_view_hash, stage_prompt_view_hash_payload=stage_payload,
        stage_messages_sha256_by_stage=stage_messages)

    evidence = copy.deepcopy(original_evidence)
    evidence.update(prompt_view_version=CATALOGUE_VERSION, stage_prompt_view_sha256=stage_view_hash,
        target_prompt_view_sha256=view_hash, stage_messages_sha256_by_stage=stage_messages,
        spec_validation_reused_from="benchmarks/generated/pilot_view_v1/stage_validation_evidence.json",
        source_spec_validation_file_sha256=sources["benchmarks/generated/pilot_view_v1/stage_validation_evidence.json"],
        unchanged_stage_specs=True, prompt_treatment_model_effectiveness_established=False)
    for sid, validation in evidence["stage_validations"].items():
        # Keep the original evidence_id: it refers to genuine v1 CPU spec/oracle
        # validation. Only add explicit provenance for this new prompt view.
        validation["details"]["prompt_view_binding"] = {"version": CATALOGUE_VERSION,
            "stage_prompt_view_sha256": stage_view_hash, "messages_sha256": stage_messages[sid],
            "specs_tests_and_oracle_evidence_unchanged": True, "model_effectiveness_established": False}
    return rows, metadata, catalogue, evidence


def freeze(output):
    rows, metadata, catalogue, evidence = build()
    output = Path(output)
    pilot.frozen_write(output / "target_rows.json", rows)
    pilot.frozen_write(output / "catalogue.json", catalogue)
    pilot.frozen_write(output / "stage_validation_evidence.json", evidence)
    metadata.update(target_rows_file_sha256=pilot.file_sha(output / "target_rows.json"),
        catalogue_file_sha256=pilot.file_sha(output / "catalogue.json"),
        stage_validation_evidence_file_sha256=pilot.file_sha(output / "stage_validation_evidence.json"))
    pilot.frozen_write(output / "target_view_metadata.json", metadata)
    return {"status": "frozen_CPU_only_no_GPU_submission", "manifest_id": rows["manifest_id"],
        "prompt_view_sha256": rows["prompt_view_sha256"], "stage_prompt_view_sha256": catalogue["prompt_view_sha256"],
        "train_count": len(rows["train"]), "selection_count": len(rows["selection"]),
        "stage_count": len(catalogue["stages"]), "source_train_manifest_sha256": rows["source_train_manifest_sha256"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "generated/naming_round_v2")
    print(json.dumps(freeze(ap.parse_args().output), indent=2))
