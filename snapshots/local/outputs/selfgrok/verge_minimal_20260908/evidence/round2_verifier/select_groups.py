"""Choose five completed training groups from immutable journal snapshots."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def snapshot(branch):
    path = HERE / branch / "training_journal.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    previous = "0" * 64
    groups, source = [], None
    for index, record in enumerate(records):
        unsigned = {k: v for k, v in record.items() if k != "sha256"}
        assert sha(unsigned) == record["sha256"] and record["previous_sha256"] == previous and record["sequence"] == index
        previous = record["sha256"]
        if record["kind"] == "generation_source":
            source = record
        elif record["kind"] == "rollouts":
            items = record["payload"]["records"]
            groups.append({"branch": branch, "source": source, "rollouts_sequence": index,
                "filename": Path(source["payload"]["raw_path"]).name, "phase": record["payload"]["phase"],
                "instance_id": items[0]["prompt_id"], "n": len(items), "reward_sum": sum(x["reward"] for x in items),
                "target": source["payload"]["target"], "drain": record["payload"]["drain"]})
    return groups, {"journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "record_count": len(records),
        "last_sequence": len(records) - 1, "last_record_sha256": previous, "complete_generation_groups": len(groups),
        "positive_groups": sum(x["reward_sum"] > 0 for x in groups),
        "actual_model_update_events": sum(r["kind"] == "actual_model_update" for r in records)}


def main():
    all_groups, snapshots = [], {}
    for branch in ("g1", "g2"):
        groups, evidence = snapshot(branch)
        all_groups += groups
        snapshots[branch] = evidence
    candidates = [g for g in all_groups if not g["target"] and not g["drain"] and g["n"] == 8]
    chosen = [g for g in candidates if g["reward_sum"] > 0][:5]
    seen = {g["instance_id"] for g in chosen}
    for group in candidates:
        if len(chosen) == 5:
            break
        if group["instance_id"] not in seen:
            chosen.append(group)
            seen.add(group["instance_id"])
    assert len(chosen) == 5
    result = {"mode": "CPU_training_verifier_audit_selection", "snapshot_scope": "g1/g2 training journals only; running jobs may progress afterward",
        "selection_rule": "Prioritize positive full 8-sample stage groups, then first full group of each distinct stage ID; at most five groups.",
        "snapshots": snapshots, "selected": chosen, "heldout_test_results_read": False}
    path = HERE / "selection.json"
    encoded = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError("Preserve the first snapshot audit selection")
    path.write_bytes(encoded)
    print(json.dumps({"snapshots": snapshots, "selected": [{k: g[k] for k in ("branch", "filename", "instance_id", "reward_sum")} for g in chosen]}, indent=2))


if __name__ == "__main__":
    main()
