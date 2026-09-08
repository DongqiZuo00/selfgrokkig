from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from common import (
    DATA_ROOT,
    EXP_ROOT,
    atomic_json,
    canonical_sha256,
    load_config,
    read_json,
    read_jsonl,
    uncertainty_matched,
)
from generation import summarize_rollouts
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


RAW_ROOT = EXP_ROOT / "raw_results" / "protocol_v2"


def rows_at(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_from_disk(str(path))]


def length_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(row["completion_tokens"]) for row in rows)
    wall = max(float(row["request_wall_seconds"]) for row in rows)
    telemetry_path = (
        RAW_ROOT / "throughput" / "optimized.telemetry.jsonl"
    )
    telemetry = read_jsonl(telemetry_path) if telemetry_path.exists() else []
    gpu_values = [
        float(item["gpu_utilization"]) for item in telemetry if "gpu_utilization" in item
    ]
    active = [float(item.get("active_sequences", 0)) for item in telemetry]
    waiting = [float(item.get("waiting_sequences", 0)) for item in telemetry]
    verifier_wall = max(float(row.get("verifier_wall_seconds", 0)) for row in rows)
    return {
        "completed_rollouts": len(rows),
        "wall_seconds": wall,
        "seconds_per_rollout": wall / len(rows),
        "output_tokens": sum(lengths),
        "output_tokens_per_second": sum(lengths) / wall,
        "mean_completion_tokens": statistics.fmean(lengths),
        "median_completion_tokens": statistics.median(lengths),
        "p95_completion_tokens": lengths[max(0, math.ceil(0.95 * len(lengths)) - 1)],
        "finish_reason_length_rate": sum(
            item.get("finish_reason") == "length" for item in rows
        )
        / len(rows),
        "cap_hit_rate": sum(length >= 2048 for length in lengths) / len(lengths),
        "think_tag_count": sum("<think>" in item["completion"] for item in rows),
        "parse_valid_rate": sum(bool(item["parse_valid"]) for item in rows) / len(rows),
        "verifier_wall_seconds": verifier_wall,
        "mean_verifier_seconds": statistics.fmean(
            float(item.get("verifier_seconds", 0)) for item in rows
        ),
        "mean_gpu_utilization": statistics.fmean(gpu_values) if gpu_values else None,
        "max_active_sequences": max(active, default=None),
        "max_waiting_sequences": max(waiting, default=None),
        "active_concurrency_submitted": len(rows),
    }


def benchmark(client: VLLMRolloutClient) -> dict[str, Any]:
    output = RAW_ROOT / "throughput" / "optimized.jsonl"
    rows = rows_at(DATA_ROOT / "target_splits" / "initial_audit")[:8]
    scored = client.score_rows(
        rows,
        8,
        920260826,
        output,
        RAW_ROOT / "throughput" / "optimized.telemetry.jsonl",
    )
    legacy_path = EXP_ROOT / "raw_results" / "target_audit" / "stratum_0.jsonl"
    legacy = read_jsonl(legacy_path)
    legacy_elapsed = 1764.0
    baseline = {
        "job_id": "40336174",
        "completed_rollouts": len(legacy),
        "wall_seconds": legacy_elapsed,
        "seconds_per_rollout": legacy_elapsed / len(legacy),
        "output_tokens": sum(int(row["completion_tokens"]) for row in legacy),
        "output_tokens_per_second": sum(int(row["completion_tokens"]) for row in legacy)
        / legacy_elapsed,
        "mean_completion_tokens": statistics.fmean(
            int(row["completion_tokens"]) for row in legacy
        ),
        "median_completion_tokens": statistics.median(
            int(row["completion_tokens"]) for row in legacy
        ),
        "p95_completion_tokens": sorted(int(row["completion_tokens"]) for row in legacy)[
            max(0, math.ceil(0.95 * len(legacy)) - 1)
        ],
        "cap_hit_rate": sum(int(row["completion_tokens"]) >= 4096 for row in legacy)
        / len(legacy),
        "think_tag_count": sum("<think>" in row["completion"] for row in legacy),
        "generation_mode": "transformers batch_size=16, max_new_tokens=4096",
    }
    report = {
        "protocol_version": 2,
        "thinking_disabled_argument": {
            "chat_template_kwargs": {"enable_thinking": False},
            "observed_think_tags_legacy": baseline["think_tag_count"],
            "observed_think_tags_optimized": sum("<think>" in row["completion"] for row in scored),
        },
        "baseline": baseline,
        "optimized": length_summary(scored),
        "selected": "vllm",
        "completion_cap": 2048,
        "program_stop": "first complete END end followed by closing manufactoria fence",
        "benchmark_finished_at": time.time(),
    }
    atomic_json(EXP_ROOT / "aggregated_results" / "throughput_report.json", report)
    return report


def target_audit(clients: list[VLLMRolloutClient]) -> dict[str, Any]:
    manifest_path = EXP_ROOT / "manifests" / "target_stratum_fast.json"
    if manifest_path.exists():
        return read_json(manifest_path)
    rows = rows_at(DATA_ROOT / "target_splits" / "initial_audit")

    def run_shard(index: int) -> list[dict[str, Any]]:
        shard = rows[index::2]
        return clients[index].score_rows(
            shard,
            128,
            20260826 + index,
            RAW_ROOT / "target_audit" / f"gpu{index}.jsonl",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        shards = list(executor.map(run_shard, (0, 1)))
    scored = [item for shard in shards for item in shard]
    summary = summarize_rollouts(scored)
    result = {
        "protocol_version": 2,
        "stratum_index": 0,
        "source": "frozen manufactoria/prepend_sequence_train initial-audit split",
        "audit_data_path": str(DATA_ROOT / "target_splits" / "initial_audit"),
        "thinking_disabled": True,
        "completion_cap": 2048,
        "two_gpu_shards": True,
        **summary,
        "criterion": "Frozen target retained; zero-success audit recorded without escalation.",
    }
    atomic_json(manifest_path, result)
    return result


def candidate_summary(
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    summary = summarize_rollouts(rows)
    p = float(summary["full_pass_rate"])
    return {
        "candidate_id": entry["candidate_id"],
        "family": entry["family"],
        "configuration": entry["configuration"],
        "screen_stage": stage,
        **summary,
        "base_success_rate": p,
        "solver_uncertainty": uncertainty_matched(p),
    }


def run_entries(
    clients: list[VLLMRolloutClient],
    entries: list[dict[str, Any]],
    stage: str,
    instance_slice: slice,
) -> dict[str, list[dict[str, Any]]]:
    partitions = [entries[index::2] for index in range(2)]

    def worker(index: int) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for entry_index, entry in enumerate(partitions[index]):
            rows = rows_at(Path(entry["audit_data_path"]))[instance_slice]
            output = (
                RAW_ROOT
                / "candidate_screening"
                / stage
                / f"{entry['candidate_id']}.jsonl"
            )
            result[entry["candidate_id"]] = clients[index].score_rows(
                rows,
                8,
                314159 + stable_entry_index(entry["candidate_id"]) + entry_index,
                output,
            )
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        partial = list(executor.map(worker, (0, 1)))
    combined: dict[str, list[dict[str, Any]]] = {}
    for mapping in partial:
        combined.update(mapping)
    return combined


def stable_entry_index(candidate_id: str) -> int:
    return int(canonical_sha256(candidate_id)[:8], 16) % 10_000_000


def matching_batches(finalists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = {item["candidate_id"]: item for item in finalists}
    ids = sorted(candidates)
    pairings = []
    for chosen in itertools.combinations(ids, 4):
        a, b, c, d = chosen
        for left, right in (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))):
            pairs = []
            for first, second in (left, right):
                x, y = candidates[first], candidates[second]
                p_gap = abs(x["base_success_rate"] - y["base_success_rate"])
                u_gap = abs(x["solver_uncertainty"] - y["solver_uncertainty"])
                pairs.append(
                    {
                        "candidate_a": first,
                        "candidate_b": second,
                        "success_rate_gap": p_gap,
                        "uncertainty_gap": u_gap,
                        "strict_match": p_gap <= 0.05 and u_gap <= 0.05,
                        "different_families": x["family"] != y["family"],
                    }
                )
            score = (
                -sum(pair["strict_match"] for pair in pairs),
                max(pair["success_rate_gap"] for pair in pairs),
                sum(pair["success_rate_gap"] for pair in pairs),
                max(pair["uncertainty_gap"] for pair in pairs),
                sum(pair["uncertainty_gap"] for pair in pairs),
                -sum(pair["different_families"] for pair in pairs),
                tuple(chosen),
            )
            pairings.append((score, pairs, set(chosen)))
    pairings.sort(key=lambda item: item[0])
    first_score, first_pairs, first_ids = pairings[0]
    remaining = [item for item in finalists if item["candidate_id"] not in first_ids]
    remaining.sort(key=lambda item: item["candidate_id"])
    second_pairs = []
    for offset in (0, 2):
        x, y = remaining[offset], remaining[offset + 1]
        p_gap = abs(x["base_success_rate"] - y["base_success_rate"])
        u_gap = abs(x["solver_uncertainty"] - y["solver_uncertainty"])
        second_pairs.append(
            {
                "candidate_a": x["candidate_id"],
                "candidate_b": y["candidate_id"],
                "success_rate_gap": p_gap,
                "uncertainty_gap": u_gap,
                "strict_match": p_gap <= 0.05 and u_gap <= 0.05,
                "different_families": x["family"] != y["family"],
            }
        )
    return [
        {"batch": 1, "pairs": first_pairs, "candidate_ids": sorted(first_ids)},
        {
            "batch": 2,
            "pairs": second_pairs,
            "candidate_ids": [item["candidate_id"] for item in remaining],
        },
    ]


def screen_candidates(clients: list[VLLMRolloutClient]) -> dict[str, Any]:
    frozen_path = EXP_ROOT / "manifests" / "discovery_candidates.json"
    if frozen_path.exists():
        return read_json(frozen_path)
    pool = read_json(EXP_ROOT / "manifests" / "candidate_pool_generation_manifest.json")
    entries = list(pool["entries"])
    first_rows = run_entries(clients, entries, "stage1_64", slice(0, 8))
    stage1 = [
        candidate_summary(entry, first_rows[entry["candidate_id"]], "64")
        for entry in entries
    ]
    stage1.sort(
        key=lambda item: (
            -item["solver_uncertainty"],
            abs(item["base_success_rate"] - 0.5),
            item["candidate_id"],
        )
    )
    finalists_stage1 = stage1[:8]
    finalist_ids = {item["candidate_id"] for item in finalists_stage1}
    finalist_entries = [entry for entry in entries if entry["candidate_id"] in finalist_ids]
    remainder = run_entries(clients, finalist_entries, "stage2_remaining_192", slice(8, 32))
    final = []
    for entry in finalist_entries:
        candidate_id = entry["candidate_id"]
        all_rows = first_rows[candidate_id] + remainder[candidate_id]
        final.append(candidate_summary(entry, all_rows, "256"))
    final.sort(
        key=lambda item: (
            -item["solver_uncertainty"],
            abs(item["base_success_rate"] - 0.5),
            item["candidate_id"],
        )
    )
    table_path = EXP_ROOT / "aggregated_results" / "candidate_uncertainty.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "family",
                "screen_stage",
                "rollouts",
                "full_pass_count",
                "base_success_rate",
                "solver_uncertainty",
                "selected_finalist",
            ],
        )
        writer.writeheader()
        final_by_id = {item["candidate_id"]: item for item in final}
        for initial in sorted(stage1, key=lambda item: item["candidate_id"]):
            row = final_by_id.get(initial["candidate_id"], initial)
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key != "selected_finalist"},
                    "selected_finalist": initial["candidate_id"] in finalist_ids,
                }
            )
    batches = matching_batches(final)
    manifest = {
        "protocol_version": 2,
        "selection_is_target_blind": True,
        "uncertainty_definition": "U = 4 p (1-p)",
        "stage1_candidates": len(entries),
        "stage1_rollouts_each": 64,
        "finalists": final,
        "finalist_rollouts_each": 256,
        "discovery_batches": batches,
        "selection_hash": canonical_sha256(
            {
                "finalists": [
                    (item["candidate_id"], item["base_success_rate"], item["solver_uncertainty"])
                    for item in final
                ],
                "batches": batches,
            }
        ),
        "target_metrics_consulted": False,
        "frozen_at": time.time(),
    }
    atomic_json(frozen_path, manifest)
    return manifest


def run_all(urls: list[str]) -> dict[str, Any]:
    clients = [VLLMRolloutClient(url, gpu=index) for index, url in enumerate(urls)]
    throughput = benchmark(clients[0])
    target = target_audit(clients)
    discovery = screen_candidates(clients)
    result = {"throughput": throughput, "target": target, "discovery": discovery}
    atomic_json(EXP_ROOT / "manifests" / "fast_screen_complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-urls", nargs=2)
    args = parser.parse_args()
    if args.server_urls:
        run_all(args.server_urls)
        return
    with VLLMServerPool() as urls:
        run_all(urls)


if __name__ == "__main__":
    main()
