"""Merge only four completed real branches and independently score their evidence."""
from pathlib import Path
from execute_round import ROOT, checked_plan, load_json, merge_fragments, require, file_sha

if __name__ == "__main__":
    preparation = ROOT / "runs/round_prepare_41413994"
    plan = checked_plan(preparation / "plan.json")
    base = load_json(preparation / "base/fragment.json")
    fragments = []
    for index, branch in enumerate(("g1", "g2", "g3", "direct")):
        directory = ROOT / "runs" / f"round1_{branch}_41414619_{index}"
        completion = load_json(directory / "COMPLETE.json")
        path = directory / "fragment.json"
        require(completion["status"] == "complete" and completion["fragment_sha256"] == file_sha(path),
                "require actual completed, unchanged branch evidence")
        fragments.append(load_json(path))
    result = merge_fragments(plan, base, fragments, ROOT / "runs/round1_result")
    print("Completed real-evidence merge and score under runs/round1_result")
