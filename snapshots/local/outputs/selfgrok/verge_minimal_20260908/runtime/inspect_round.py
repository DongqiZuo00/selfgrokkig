"""Read the explicitly named minimum round's on-disk evidence, without model work."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARRAY_ID = "41414619"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect():
    branches = []
    for index, branch in enumerate(("g1", "g2", "g3", "direct")):
        output = ROOT / "runs" / f"round1_{branch}_{ARRAY_ID}_{index}"
        item = {"branch": branch, "path": str(output), "status_on_disk": "not_started"}
        if (output / "COMPLETE.json").exists():
            item.update(status_on_disk="complete", execution=read(output / "COMPLETE.json")["execution"])
        elif (output / "FAILED.json").exists():
            item.update(status_on_disk="failed", failure=read(output / "FAILED.json"))
        elif (output / "LAUNCH.json").exists():
            item["status_on_disk"] = "started_not_complete"
        journals = list((output / "execution").glob("*.jsonl"))
        if journals:
            rows = []
            for line in journals[0].read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    item["last_journal_line_in_progress"] = True
            ledger = [r["payload"] for r in rows if r["kind"] == "accounting"]
            updates = [r for r in rows if r["kind"] == "actual_model_update"]
            records = [sample for r in rows if r["kind"] == "rollouts" for sample in r["payload"]["records"]]
            item.update(last_accounting=ledger[-1] if ledger else None,
                        actual_model_updates_recorded=len(updates),
                        actual_training_generated_tokens=sum(len(r["completion_token_ids"]) for r in records),
                        recorded_stage_successes=sum(r["reward"] for r in records if r["task_kind"] == "stage"),
                        recorded_target_successes=sum(r["reward"] for r in records if r["task_kind"] == "target"))
        branches.append(item)
    return {"mode": "in_progress_disk_evidence_not_final_score", "array_id": ARRAY_ID, "branches": branches}


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))
