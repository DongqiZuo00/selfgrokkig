"""Inspect or finalize the explicitly frozen third real round; never submits jobs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "runs/round_prepare_ignition_41419999"
ARRAY = "41420288"
BRANCHES = ("g1", "g2", "g3", "direct")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def branch_dir(index, branch):
    return ROOT / "runs" / f"round3_{branch}_{ARRAY}_{index}"


def inspect():
    result = []
    for index, branch in enumerate(BRANCHES):
        path = branch_dir(index, branch)
        item = {"branch": branch, "path": str(path), "status": "not_started"}
        if (path / "COMPLETE.json").exists():
            item.update(status="complete", execution=read(path / "COMPLETE.json")["execution"])
        elif (path / "FAILED.json").exists():
            item.update(status="failed", failure=read(path / "FAILED.json"))
        elif (path / "LAUNCH.json").exists():
            item["status"] = "running"
        rows = []
        for journal in (path / "execution").glob("*.jsonl"):
            for line in journal.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    item["journal_last_line_in_progress"] = True
        samples = [sample for row in rows if row["kind"] == "rollouts"
                   for sample in row["payload"]["records"]]
        accounting = [row["payload"] for row in rows if row["kind"] == "accounting"]
        item.update(training_generated_tokens=sum(len(s["completion_token_ids"]) for s in samples),
                    stage_successes=sum(s["reward"] for s in samples if s["task_kind"] == "stage"),
                    target_successes=sum(s["reward"] for s in samples if s["task_kind"] == "target"),
                    actual_updates=sum(row["kind"] == "actual_model_update" for row in rows),
                    last_accounting=accounting[-1] if accounting else None)
        evaluation = [read(raw) for raw in (path / "execution/raw_evaluation").glob("*.json")]
        item["new_eval_rollouts"] = sum(len(raw["completion_token_ids"]) for raw in evaluation)
        item["new_eval_tokens"] = sum(len(ids) for raw in evaluation for ids in raw["completion_token_ids"])
        result.append(item)
    return {"array": ARRAY, "mode": "actual_disk_evidence", "branches": result}


def finalize():
    from execute_round import checked_plan, file_sha, merge_fragments, require
    plan = checked_plan(PREP / "plan.json")
    base = read(PREP / "base/fragment.json")
    fragments = []
    for index, branch in enumerate(BRANCHES):
        path = branch_dir(index, branch)
        complete = read(path / "COMPLETE.json")
        require(complete["status"] == "complete", "only merge complete real branches")
        require(complete["fragment_sha256"] == file_sha(path / "fragment.json"), "fragment changed")
        fragments.append(read(path / "fragment.json"))
    result_path = ROOT / "runs/round3_result"
    require(not result_path.exists(), "preserve any existing result")
    merge_fragments(plan, base, fragments, result_path)
    return {"status": "complete", "result": str(result_path), "summary": read(result_path / "summary.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = finalize() if args.finalize else inspect()
    if args.compact and not args.finalize:
        fields = ("branch", "status", "training_generated_tokens", "stage_successes", "target_successes", "actual_updates",
                  "new_eval_rollouts", "new_eval_tokens")
        result["branches"] = [{k: row[k] for k in fields} for row in result["branches"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))
