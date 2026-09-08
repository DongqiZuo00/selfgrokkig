"""Compact read-only progress, using produced files only."""
import json
import os
from pathlib import Path

exp = Path(__file__).resolve().parents[1]
version = os.environ.get("VERGE_VERSION", "verge_mistral_repair_acceptance")
if version not in ("verge_mistral_repair_acceptance", "verge_mistral_repair_v2"):
    raise ValueError("Unexpected namespace")
root = exp / "raw_results" / version
def load(path):
    return json.loads(path.read_text()) if path.exists() else None
frozen = load(root / "round_frozen.json")
result = {"version":version, "frozen":frozen is not None, "branches":[],
          "acceptance":load(root / "ENGINEERING_ACCEPTANCE.json"),
          "completion_manifest":load(exp / "manifests" / f"{version}_complete.json")}
if frozen:
    for b in frozen["branches"]:
        directory = root / "branches" / str(b["index"])
        complete = load(directory / "complete.json")
        summaries = sorted((directory / "train").glob("update_*_summary.json"))
        progress = load(summaries[-1]) if summaries else None
        state = complete["training"] if complete else progress["cumulative"] if progress else {}
        result["branches"].append({"index":b["index"], "complete":complete is not None,
            "updates":state.get("update",0), "optimizer_steps":state.get("optimizer_steps",0),
            "training_tokens":state.get("train_tokens",0), "nonzero_advantage_tokens":state.get("nonzero_advantage_tokens",0),
            "training_target_successes":state.get("target_successes",0),
            "endpoint_full_pass":None if not complete else {
                "count":complete["target"]["counts"][-1], "draws":complete["target"]["rollouts"]},
            "loss_token_budget":frozen["config"]["train_tokens_per_branch"]})
result["teacher"] = load(root / "challenger_update.json")
print(json.dumps(result, indent=2))
