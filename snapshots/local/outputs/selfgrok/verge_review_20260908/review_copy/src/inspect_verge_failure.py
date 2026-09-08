"""Summarize existing rollouts and budgets; no new samples, verifier runs or hash scans."""
import json
from collections import Counter
from pathlib import Path
from common import EXP_ROOT, read_json, read_jsonl, group_advantages
from verge_round_core import ROOT, allocate_tokens


def summary(path):
    rows = read_jsonl(path)
    lengths = sorted(r["completion_tokens"] for r in rows)
    return {"n": len(rows), "successes": sum(r["reward"] for r in rows),
        "parse": sum(r["parse_valid"] for r in rows), "finish": dict(Counter(r["finish_reason"] for r in rows)),
        "length_mean": sum(lengths) / len(lengths), "length_p50": lengths[len(lengths)//2],
        "errors": Counter(r["verifier_message"] for r in rows if not r["parse_valid"]).most_common(5)}


print("EVALUATIONS")
for p in [ROOT / "initial/target.jsonl", ROOT / "branches/0/target.jsonl", ROOT / "branches/2/target.jsonl",
          EXP_ROOT / "raw_results/recipe_v1/branches/decisive_mistral_v5_5__oracle/grounded_reward/update_0004.jsonl"]:
    print(p.relative_to(EXP_ROOT), json.dumps(summary(p)))
print("TRAINING")
for index in (0, 2):
    root = ROOT / "branches" / str(index)
    complete = read_json(root / "complete.json")
    for u in range(1, complete["training"]["update"] + 1):
        raw = read_jsonl(root / "train" / f"update_{u:04d}.jsonl")
        stats = read_json(root / "train" / f"update_{u:04d}_summary.json")
        quota = allocate_tokens([len(r["completion_token_ids"]) for r in raw], stats["tokens_used"])
        usable_successes = sum(r["reward"] for r, n in zip(raw, quota) if n > 0)
        print(json.dumps({"branch": index, "update": u, "phase": stats["stage"], "kind": stats["stage_kind"],
            "successes_generated": sum(r["reward"] for r in raw), "successes_used": usable_successes,
            "active_tokens": stats["nonzero_advantage_tokens"], "loss_tokens": sum(quota),
            "generated_tokens": sum(r["completion_tokens"] for r in raw),
            "completions_used": sum(n > 0 for n in quota), "completions_full": sum(n == r["completion_tokens"] for n, r in zip(quota, raw)),
            "target_tokens_used": sum(n for n, r in zip(quota, raw) if "target_train" in r["instance_id"]),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in raw))}))
print("RAW_TARGET_EXCERPTS")
raw = read_jsonl(ROOT / "initial/target.jsonl")
for reason in ("length", "stop"):
    selected = next((r for r in raw if r["finish_reason"] == reason and not r["parse_valid"]), None)
    if selected:
        print(json.dumps({"finish_reason": reason, "error": selected["verifier_message"],
            "head": selected["completion"][:1800], "tail": selected["completion"][-800:]}))
print("STAGES", json.dumps(read_json(ROOT / "round_frozen.json")["probes"]))
