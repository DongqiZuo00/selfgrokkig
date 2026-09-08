from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from common import EXP_ROOT, atomic_json, read_json, read_jsonl


RAW_ROOT = EXP_ROOT / "raw_results" / "protocol_v2" / "branches"


def endpoint_summary(branch: dict[str, Any]) -> dict[str, Any]:
    split = branch["endpoint_split"]
    path = RAW_ROOT / branch["branch_id"] / split / "update_0100_summary.json"
    return read_json(path)


def branch_lookup() -> tuple[list[dict[str, Any]], dict[tuple[str, int, str | None], dict[str, Any]]]:
    branches = read_json(EXP_ROOT / "manifests" / "fast_branches.json")["branches"]
    lookup = {
        (item["stage"], int(item["seed"]), item.get("candidate_id")): item for item in branches
    }
    return branches, lookup


def discovery_rows(batch: int) -> list[dict[str, Any]]:
    discovery = read_json(EXP_ROOT / "manifests" / "discovery_candidates.json")
    selected_batch = next(
        item for item in discovery["discovery_batches"] if int(item["batch"]) == batch
    )
    _, lookup = branch_lookup()
    direct = lookup[("discovery", 42, None)]
    direct_summary = endpoint_summary(direct)
    direct_rate = float(direct_summary["full_pass_rate"])
    finalist = {item["candidate_id"]: item for item in discovery["finalists"]}
    rows = []
    for pair_index, pair in enumerate(selected_batch["pairs"], start=1):
        for role, candidate_id in (
            ("A", pair["candidate_a"]),
            ("B", pair["candidate_b"]),
        ):
            branch = lookup[("discovery", 42, candidate_id)]
            summary = endpoint_summary(branch)
            candidate = finalist[candidate_id]
            rows.append(
                {
                    "batch": batch,
                    "pair": pair_index,
                    "role": role,
                    "branch_id": branch["branch_id"],
                    "candidate_id": candidate_id,
                    "family": candidate["family"],
                    "base_success_rate": candidate["base_success_rate"],
                    "solver_uncertainty": candidate["solver_uncertainty"],
                    "target_full_pass_count": summary["full_pass_count"],
                    "target_pass_at_1": summary["full_pass_rate"],
                    "direct_pass_at_1": direct_rate,
                    "target_gain": float(summary["full_pass_rate"]) - direct_rate,
                }
            )
    return rows


def write_discovery_table(all_rows: list[dict[str, Any]]) -> None:
    path = EXP_ROOT / "aggregated_results" / "discovery_branch_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "pair",
        "role",
        "branch_id",
        "candidate_id",
        "family",
        "base_success_rate",
        "solver_uncertainty",
        "target_full_pass_count",
        "target_pass_at_1",
        "direct_pass_at_1",
        "target_gain",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def select_discovery_pair(batch: int) -> dict[str, Any] | None:
    existing_path = EXP_ROOT / "manifests" / "confirmed_pair_manifest.json"
    if existing_path.exists():
        return read_json(existing_path)
    current = discovery_rows(batch)
    previous = []
    table_path = EXP_ROOT / "aggregated_results" / "discovery_branch_results.csv"
    if table_path.exists():
        with table_path.open("r", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
    merged = [
        row for row in previous if int(row["batch"]) != batch
    ] + current
    write_discovery_table(merged)
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in current:
        by_pair.setdefault(int(row["pair"]), {})[row["role"]] = row
    eligible = []
    for pair_index, pair_rows in by_pair.items():
        a, b = pair_rows["A"], pair_rows["B"]
        gap = float(a["target_gain"]) - float(b["target_gain"])
        any_success = int(a["target_full_pass_count"]) + int(
            b["target_full_pass_count"]
        ) > 0
        if any_success and abs(gap) >= 0.02:
            eligible.append((abs(gap), pair_index, gap, a, b))
    if not eligible:
        atomic_json(
            EXP_ROOT / "manifests" / f"discovery_batch_{batch}_decision.json",
            {
                "batch": batch,
                "pair_found": False,
                "rule": "at least one verified target success and absolute target-gain gap >= 0.02",
            },
        )
        return None
    _, pair_index, gap, a, b = max(eligible, key=lambda item: (item[0], -item[1]))
    direction = 1 if gap > 0 else -1
    manifest = {
        "protocol_version": 2,
        "discovery_batch": batch,
        "pair_index": pair_index,
        "candidate_a": a["candidate_id"],
        "candidate_b": b["candidate_id"],
        "candidate_a_uncertainty": a["solver_uncertainty"],
        "candidate_b_uncertainty": b["solver_uncertainty"],
        "candidate_a_base_success_rate": a["base_success_rate"],
        "candidate_b_base_success_rate": b["base_success_rate"],
        "discovery_target_gain_a": a["target_gain"],
        "discovery_target_gain_b": b["target_gain"],
        "discovery_a_minus_b": gap,
        "frozen_direction": direction,
        "fixed_endpoint": 100,
        "confirmation_seeds": [123, 2026, 31415],
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260826,
        "selection_uses_confirmation": False,
        "selection_uses_official_test": False,
    }
    atomic_json(existing_path, manifest)
    return manifest


def rewards_by_instance(branch: dict[str, Any], split: str) -> dict[str, list[int]]:
    path = RAW_ROOT / branch["branch_id"] / split / "update_0100.jsonl"
    result: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        result.setdefault(str(row["instance_id"]), []).append(int(row["reward"]))
    return result


def mean_rewards(values: dict[str, list[int]], instance_ids: list[str]) -> float:
    rewards = [reward for instance_id in instance_ids for reward in values[instance_id]]
    return sum(rewards) / len(rewards)


def hierarchical_bootstrap(
    seed_data: dict[int, tuple[dict[str, list[int]], dict[str, list[int]]]],
    replicates: int = 10000,
    seed: int = 20260826,
) -> tuple[float, float, list[float]]:
    rng = random.Random(seed)
    seeds = sorted(seed_data)
    estimates = []
    for _ in range(replicates):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        differences = []
        for sampled_seed in sampled_seeds:
            a, b = seed_data[sampled_seed]
            ids = sorted(set(a) & set(b))
            sampled_ids = [rng.choice(ids) for _ in ids]
            differences.append(
                mean_rewards(a, sampled_ids) - mean_rewards(b, sampled_ids)
            )
        estimates.append(sum(differences) / len(differences))
    estimates.sort()
    lower = estimates[math.floor(0.025 * replicates)]
    upper = estimates[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return lower, upper, estimates


def analyze_confirmation() -> dict[str, Any]:
    pair = read_json(EXP_ROOT / "manifests" / "confirmed_pair_manifest.json")
    _, lookup = branch_lookup()
    a_id = pair["candidate_a"]
    b_id = pair["candidate_b"]
    direction = int(pair["frozen_direction"])
    seed_rows = []
    bootstrap_data = {}
    for seed in (123, 2026, 31415):
        direct = lookup[("confirmation", seed, None)]
        a_branch = lookup[("confirmation", seed, a_id)]
        b_branch = lookup[("confirmation", seed, b_id)]
        direct_rate = float(endpoint_summary(direct)["full_pass_rate"])
        a_rate = float(endpoint_summary(a_branch)["full_pass_rate"])
        b_rate = float(endpoint_summary(b_branch)["full_pass_rate"])
        seed_rows.append(
            {
                "seed": seed,
                "candidate_a": a_id,
                "candidate_b": b_id,
                "candidate_a_pass_at_1": a_rate,
                "candidate_b_pass_at_1": b_rate,
                "direct_pass_at_1": direct_rate,
                "candidate_a_target_gain": a_rate - direct_rate,
                "candidate_b_target_gain": b_rate - direct_rate,
                "a_minus_b": a_rate - b_rate,
            }
        )
        bootstrap_data[seed] = (
            rewards_by_instance(a_branch, "confirmation"),
            rewards_by_instance(b_branch, "confirmation"),
        )
    lower, upper, _ = hierarchical_bootstrap(bootstrap_data)
    direction_consistent = all(direction * row["a_minus_b"] > 0 for row in seed_rows)
    ci_excludes_zero = lower > 0 or upper < 0
    any_positive_gain = any(
        row["candidate_a_target_gain"] > 0 or row["candidate_b_target_gain"] > 0
        for row in seed_rows
    )
    supported = direction_consistent and ci_excludes_zero and any_positive_gain
    table_path = EXP_ROOT / "aggregated_results" / "seed_level_target_gain.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    confirmed_path = EXP_ROOT / "aggregated_results" / "confirmed_pair_results.csv"
    with confirmed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    result = {
        "protocol_version": 2,
        "candidate_a": a_id,
        "candidate_b": b_id,
        "seed_results": seed_rows,
        "frozen_direction": direction,
        "direction_consistent": direction_consistent,
        "hierarchical_paired_bootstrap_95_ci": [lower, upper],
        "ci_excludes_zero": ci_excludes_zero,
        "at_least_one_positive_target_gain": any_positive_gain,
        "confirmation_supported": supported,
        "official_test_authorized": supported,
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "statistical_summary.json", result)
    return result


def write_null_summary(reason: str, branches_run: int) -> None:
    atomic_json(
        EXP_ROOT / "aggregated_results" / "statistical_summary.json",
        {
            "protocol_version": 2,
            "confirmation_supported": False,
            "official_test_authorized": False,
            "outcome": "not supported",
            "reason": reason,
            "training_branches_run": branches_run,
        },
    )
