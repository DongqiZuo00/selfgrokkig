"""Inspect existing endpoint request structure without generating or changing samples."""
import argparse
import json
from pathlib import Path


def inspect(root):
    raw = root / "target.response.json"
    payload = json.loads(raw.read_text())
    request = payload["request"]
    prompts, n = request["prompt"], request["n"]
    unique = []
    sizes = []
    for prompt in prompts:
        for i, existing in enumerate(unique):
            if prompt == existing:
                sizes[i] += 1
                break
        else:
            unique.append(prompt)
            sizes.append(1)
    choices = sorted(payload["response"]["choices"], key=lambda x: x["index"])
    groups = [[c["token_ids"] for c in choices[i*n:(i+1)*n]] for i in range(len(prompts))]
    equal_prompt_pairs = equal_action_pairs = exact_cohort_pairs = 0
    examples = []
    for i in range(len(prompts)):
        for j in range(i):
            if prompts[i] != prompts[j]:
                continue
            equal_prompt_pairs += 1
            matches = sum(a == b for a, b in zip(groups[i], groups[j]))
            equal_action_pairs += matches
            exact_cohort_pairs += int(matches == n)
            if len(examples) < 5:
                examples.append({"prompt_indices": [j, i], "matched_native_actions": matches})
    records = [json.loads(line) for line in (root / "target.jsonl").read_text().splitlines() if line]
    outcomes_by_draw = [sum(r["reward"] for r in records if r["rollout_index"] == k) for k in range(n)]
    return {"path": str(root), "prompt_count": len(prompts), "distinct_prompt_token_lists": len(unique),
        "prompt_multiplicities": sizes, "samples_per_prompt": n, "single_request_seed": request.get("seed"),
        "equal_prompt_pairs": equal_prompt_pairs, "same_draw_exact_native_action_matches": equal_action_pairs,
        "exact_matching_cohort_pairs": exact_cohort_pairs, "examples": examples,
        "full_successes_by_draw_index": outcomes_by_draw,
        "new_sampling": False, "checkpoint_reads": False, "hash_scans": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    exp = Path(__file__).resolve().parents[1]
    for relative in args.paths:
        root = (exp / relative).resolve()
        if not root.is_relative_to((exp / "raw_results").resolve()):
            raise ValueError("Expected an existing raw-results endpoint")
        print(json.dumps(inspect(root), indent=2))
