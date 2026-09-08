from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Callable

from datasets import load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, load_config, stable_int, verify_full, write_jsonl
from generation import InferenceEngine, summarize_rollouts
from train_branch import build_model, get_branch, latest_checkpoint


def prepend_oracle(name: str) -> tuple[str, Callable[[str], str]]:
    match = re.fullmatch(r"Prepend ([RBYG]+)(?: after (.+))?", name)
    if not match:
        raise ValueError(f"cannot parse official PREPEND name: {name}")
    prefix, mutation = match.groups()
    if mutation is None:
        transform = lambda value: value
    elif mutation == "Remove B":
        transform = lambda value: value.replace("B", "")
    elif mutation == "Remove R":
        transform = lambda value: value.replace("R", "")
    elif mutation == "Swap color R&B":
        transform = lambda value: value.translate(str.maketrans({"R": "B", "B": "R"}))
    elif (found := re.fullmatch(r"Change ([BR]) to ([RBYG]+)", mutation)):
        source, target = found.groups()
        transform = lambda value, source=source, target=target: value.replace(source, target)
    elif (found := re.fullmatch(r"Remove patterns ([RBYG]+)", mutation)):
        source = found.group(1)
        transform = lambda value, source=source: value.replace(source, "")
    elif (found := re.fullmatch(r"Replace patterns ([RBYG]+) with ([RBYG]+)", mutation)):
        source, target = found.groups()
        transform = lambda value, source=source, target=target: value.replace(source, target)
    else:
        raise ValueError(f"unknown official PREPEND mutation: {mutation}")
    return prefix, lambda value: prefix + transform(value)


def hidden_cases(row: dict[str, Any], count: int) -> list[dict[str, Any]]:
    _, oracle = prepend_oracle(row["name"])
    rng = random.Random(stable_int("hidden_prepend", row["id"]))
    seen = {case["input"] for case in row["ground_truth"]}
    values: list[str] = []
    alphabet = "RB"
    fixed = ["", "R", "B", "RR", "BB", "RB", "BR", "R" * 24, "B" * 24, "RB" * 12]
    for value in fixed:
        if value not in seen and value not in values:
            values.append(value)
    while len(values) < count:
        length = rng.randint(0, 32)
        value = "".join(rng.choice(alphabet) for _ in range(length))
        if value not in seen and value not in values:
            values.append(value)
    return [
        {
            "input": value,
            "expected_output": oracle(value),
            "expected_accepted": True,
            "check_output": True,
            "description": "deterministic additional hidden PREPEND test",
        }
        for value in values[:count]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--split", choices=["confirmation", "official_test"], required=True)
    args = parser.parse_args()
    config = load_config()
    _, branch = get_branch(args.branch_id)
    checkpoint_update, checkpoint = latest_checkpoint(args.branch_id, args.budget)
    if checkpoint is None or checkpoint_update != args.budget:
        raise RuntimeError(f"missing exact checkpoint at common budget {args.budget}")
    tokenizer, model = build_model(int(branch["seed"]), checkpoint)
    if args.split == "confirmation":
        data_path = DATA_ROOT / "target_splits" / "confirmation"
    else:
        data_path = DATA_ROOT / "official_sealed" / "prepend_sequence_test"
        if not (EXP_ROOT / "manifests" / "official_test_unsealed.json").exists():
            raise RuntimeError("sealed official test has not been protocol-unsealed")
    rows = [dict(row) for row in load_from_disk(str(data_path))]
    output_root = (
        EXP_ROOT / "raw_results" / "branches" / args.branch_id / args.split / f"budget_{args.budget}"
    )
    raw_path = output_root.with_suffix(".jsonl")
    engine = InferenceEngine(model=model, tokenizer=tokenizer)
    scored = engine.score_rows(
        rows,
        int(config["evaluation"]["rollouts_per_instance"]),
        stable_int(args.split, branch["seed"], branch.get("candidate_id"), args.budget),
        raw_path,
        int(config["training"]["generation_batch_size"]),
    )
    summary = summarize_rollouts(scored)
    summary.update(
        {
            "branch_id": args.branch_id,
            "kind": branch["kind"],
            "candidate_id": branch.get("candidate_id"),
            "seed": int(branch["seed"]),
            "budget": args.budget,
            "split": args.split,
        }
    )
    if args.split == "confirmation":
        augmented = []
        confirmed = 0
        hidden_count = int(config["evaluation"]["hidden_tests_per_confirmed_program"])
        for item in scored:
            hidden_reward = 0
            if item["reward"]:
                row = rows[int(item["instance_index"])]
                hidden_reward = verify_full(item["completion"], hidden_cases(row, hidden_count)).reward
            value = {**item, "hidden_full_pass": hidden_reward, "confirmed_reward": int(item["reward"] and hidden_reward)}
            augmented.append(value)
            confirmed += value["confirmed_reward"]
        write_jsonl(output_root.with_name(output_root.name + "_hidden.jsonl"), augmented)
        summary["confirmed_full_pass_count"] = confirmed
        summary["confirmed_full_pass_rate"] = confirmed / len(scored)
        summary["hidden_tests_per_official_success"] = hidden_count
    atomic_json(output_root.with_name(output_root.name + "_summary.json"), summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
