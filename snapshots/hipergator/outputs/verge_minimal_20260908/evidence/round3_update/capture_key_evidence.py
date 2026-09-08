"""Read only g1 training; freeze key evidence after a changed checkpoint exists."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/round3_g1_41420288_0"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect():
    journal = RUN / "execution/training_journal.jsonl"
    if not journal.exists():
        return {"status": "waiting_for_training_journal", "groups": 0}, [], []
    data = journal.read_bytes()
    records, lines = [], []
    for line in data.splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        assert record["sequence"] == len(records)
        assert record["previous_sha256"] == (records[-1]["sha256"] if records else "0" * 64)
        assert record["sha256"] == sha({k: v for k, v in record.items() if k != "sha256"})
        records.append(record)
        lines.append(line)
    sources, groups, latest_source = {}, [], None
    for record in records:
        if record["kind"] == "generation_source":
            latest_source = record
            sources[record["payload"]["raw_path"]] = record
        elif record["kind"] == "rollouts":
            group = {"source": latest_source, "rollout_event": record}
            groups.append(group)
        elif record["kind"] == "actual_model_update":
            groups[-1]["update_event"] = record
        elif record["kind"] == "accounting" and groups:
            groups[-1]["accounting_event"] = record
    complete = [g for g in groups if "accounting_event" in g]
    updated = [g for g in complete if "update_event" in g]
    checkpoints = [r for r in records if r["kind"] == "checkpoint_boundary"]
    result = {"status": "waiting_for_mixed_update", "journal_records": len(records), "groups": len(complete),
        "stage_successes": sum(s["reward"] for g in complete for s in g["rollout_event"]["payload"]["records"] if s["task_kind"] == "stage"),
        "actual_updates": len(updated), "checkpoint_boundaries": len(checkpoints),
        "complete_file_exists": (RUN / "COMPLETE.json").exists(), "failed_file_exists": (RUN / "FAILED.json").exists()}
    if not updated:
        return result, records, lines
    first = updated[0]
    boundary = next((r for r in checkpoints if r["payload"]["phase"] == first["source"]["payload"]["phase"]), None)
    result["status"] = "waiting_for_changed_phase_checkpoint"
    if boundary is None:
        return result, records, lines
    constant = next((g for g in complete if g["source"]["sequence"] > first["update_event"]["sequence"]
        and g["source"]["payload"]["phase"] == first["source"]["payload"]["phase"]
        and "update_event" not in g and not g["rollout_event"]["payload"]["drain"]), None)
    result.update(status="ready", first_mixed=first, later_constant=constant, checkpoint_boundary=boundary)
    return result, records, lines


def freeze(output):
    status, records, lines = inspect()
    if status["status"] != "ready":
        print(json.dumps(status, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "training_journal.jsonl").write_bytes(b"".join(lines))
    for name in ("LAUNCH.json", "runtime_contract.json"):
        (output / name).write_bytes((RUN / name).read_bytes())
    launch = read(RUN / "LAUNCH.json")
    source_plan = Path(launch["preparation_root"]) / "plan.json"
    (output / "plan.json").write_bytes(source_plan.read_bytes())
    contract = Path(launch["runtime_contract"]["path"])
    (output / "source_runtime_contract.json").write_bytes(contract.read_bytes())
    for key in ("first_mixed", "later_constant"):
        group = status[key]
        if group is None:
            continue
        path = Path(group["source"]["payload"]["raw_path"])
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == group["source"]["payload"]["raw_sha256"]
        (output / f"{key}.json").write_bytes(data)
    status.update(source_run=str(RUN), journal_snapshot_sha256=hashlib.sha256(b"".join(lines)).hexdigest(),
        final_target_evaluations_read=False, heldout_test_model_results_read=False)
    (output / "snapshot.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "path": str(output), "groups": status["groups"],
        "actual_updates": status["actual_updates"], "stage_successes": status["stage_successes"],
        "has_later_constant": status["later_constant"] is not None,
        "checkpoint": status["checkpoint_boundary"]["payload"]["checkpoint"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path)
    args = ap.parse_args()
    if args.freeze:
        freeze(args.freeze)
    else:
        value, _, _ = inspect()
        print(json.dumps({k: v for k, v in value.items() if k not in {"first_mixed", "later_constant", "checkpoint_boundary"}}, indent=2))
