"""Freeze a names-only prompt treatment; preserve v1 tasks, suites and topology freedom."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "verge_stage_naming_prompt_v2_20260908"
PROMPT_DELTA = "Use this canonical naming format. Keep START start first and END end last. If there are K intermediate nodes (0 <= K <= 30), name and declare them n0, n1, ..., n(K-1) in that exact order, without gaps or other node names. In node declarations, <node_id> is the next canonical name. In routes, <node_id> may be start, end, NONE, or any intermediate name declared in the program, including one declared later. Declaration order does not determine execution order. Use four-space route indentation. END has no colon."


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_value(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    catalogue_path = HERE / "generated/catalogue.json"
    screen_path = HERE.parent / "evidence/stage_ignition/job_41412533/frozen_screen.json"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    frozen = json.loads(screen_path.read_text(encoding="utf-8"))
    by_id = {s["stage_id"]: s for s in catalogue["stages"]}
    tasks, checks = [], []
    assert len(frozen["tasks"]) == 12 and frozen["samples_per_task"] == 8
    for index, (stage_id, variant) in enumerate(frozen["tasks"]):
        original = by_id[stage_id]["rows_by_variant"][variant][0]
        row = copy.deepcopy(original)
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        row["messages"][0]["content"] += "\n\n" + PROMPT_DELTA
        assert {k: v for k, v in row.items() if k != "messages"} == {k: v for k, v in original.items() if k != "messages"}
        assert row["hint"] is None
        tasks.append({"stage_id": stage_id, "variant": variant, "row": row,
            "cap": frozen["caps"].get(stage_id, frozen["caps"]["default"]), "seed": 2026090800 + index})
        checks.append({"stage_id": stage_id, "variant": variant, "instance_id": row["id"],
            "original_messages_sha256": digest_value(original["messages"]), "messages_sha256": digest_value(row["messages"]),
            "ground_truth_sha256": digest_value(row["ground_truth"]), "instance_tuple": row["instance_tuple"],
            "all_non_message_fields_identical": True})
    sources = {"generated/catalogue.json": digest(catalogue_path),
        "evidence/stage_ignition/job_41412533/frozen_screen.json": digest(screen_path),
        "freeze_naming_prompt_v2.py": digest(Path(__file__))}
    payload = {"version": VERSION, "prompt_delta": PROMPT_DELTA, "source_sha256": sources,
        "samples_per_task": 8, "tasks": tasks}
    metadata = {"version": VERSION, "status": "frozen_unmeasured_prompt_treatment", "source_sha256": sources,
        "screen_tasks_sha256": digest_value(payload), "prompt_delta_sha256": hashlib.sha256(PROMPT_DELTA.encode("utf-8")).hexdigest(),
        "task_count": 12, "samples_per_task": 8, "checks": checks,
        "only_messages_changed": True, "task_semantics_changed": False, "test_inputs_or_outputs_changed": False,
        "references_or_solution_supplied": False, "prescribes_start_route_or_task_topology": False,
        "heldout_or_test_model_results_read": False, "v1_target_view_modified": False,
        "target_view_followup": "The same task-independent naming paragraph is the candidate for a future target view; no target rows are generated or modified by this file. Freeze that view only after the bounded stage-probe decision."}
    return payload, metadata


def prepare(output):
    payload, metadata = build()
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("screen_tasks.json", payload), ("metadata.json", metadata)):
        data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Frozen view differs; choose a new directory: {path}")
        path.write_bytes(data)
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "generated/naming_prompt_v2")
    meta = prepare(ap.parse_args().output)
    print(json.dumps({"status": "PASS", "task_count": meta["task_count"], "screen_tasks_sha256": meta["screen_tasks_sha256"]}, indent=2))
