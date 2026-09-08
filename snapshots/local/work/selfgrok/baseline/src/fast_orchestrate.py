from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, load_config, read_json, stable_int
from fast_analysis import analyze_confirmation, select_discovery_pair, write_null_summary
from fast_official import prepare_sealed
from fast_screen import run_all
from generation import summarize_rollouts
from prepare_data import materialize_selected_training
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


PYTHON = sys.executable
SRC = EXP_ROOT / "src"
RAW_ROOT = EXP_ROOT / "raw_results" / "protocol_v2"


def safe_candidate(candidate_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id)


def make_branch(
    stage: str,
    kind: str,
    seed: int,
    candidate_id: str | None,
    endpoint_split: str,
    training_paths: dict[str, str],
) -> dict[str, Any]:
    suffix = f"direct__seed{seed}" if candidate_id is None else (
        f"candidate__{safe_candidate(candidate_id)}__seed{seed}"
    )
    return {
        "branch_id": f"fast__{stage}__{suffix}",
        "stage": stage,
        "kind": kind,
        "seed": seed,
        "candidate_id": candidate_id,
        "endpoint_split": endpoint_split,
        "training_data_path": training_paths.get(candidate_id) if candidate_id else None,
        "fixed_updates": 100,
        "extension_allowed": False,
    }


def materialize_training(candidate_ids: list[str]) -> dict[str, str]:
    result = {}
    for candidate_id in candidate_ids:
        path = DATA_ROOT / "candidate_pool" / candidate_id / "training"
        if not path.exists():
            path = materialize_selected_training(candidate_id, 480)
        result[candidate_id] = str(path)
    return result


def initialize_branch_manifest(discovery: dict[str, Any]) -> dict[str, Any]:
    path = EXP_ROOT / "manifests" / "fast_branches.json"
    if path.exists():
        return read_json(path)
    candidate_ids = [item["candidate_id"] for item in discovery["finalists"]]
    training_paths = materialize_training(candidate_ids)
    branches = [
        make_branch("discovery", "direct", 42, None, "development", training_paths)
    ]
    branches.extend(
        make_branch(
            "discovery",
            "candidate",
            42,
            candidate_id,
            "development",
            training_paths,
        )
        for candidate_id in candidate_ids
    )
    manifest = {
        "protocol_version": 2,
        "branches": branches,
        "confirmation_appended": False,
        "possible_discovery_branches": 9,
        "initial_discovery_limit": 5,
    }
    atomic_json(path, manifest)
    return manifest


def append_confirmation_branches(pair: dict[str, Any]) -> dict[str, Any]:
    path = EXP_ROOT / "manifests" / "fast_branches.json"
    manifest = read_json(path)
    if manifest.get("confirmation_appended"):
        return manifest
    candidate_ids = [pair["candidate_a"], pair["candidate_b"]]
    training_paths = materialize_training(candidate_ids)
    for seed in (123, 2026, 31415):
        manifest["branches"].append(
            make_branch(
                "confirmation", "direct", seed, None, "confirmation", training_paths
            )
        )
        for candidate_id in candidate_ids:
            manifest["branches"].append(
                make_branch(
                    "confirmation",
                    "candidate",
                    seed,
                    candidate_id,
                    "confirmation",
                    training_paths,
                )
            )
    manifest["confirmation_appended"] = True
    manifest["confirmed_candidate_ids"] = candidate_ids
    manifest["confirmation_branch_count"] = 9
    atomic_json(path, manifest)
    return manifest


def branch_complete(branch: dict[str, Any]) -> bool:
    path = RAW_ROOT / "branches" / branch["branch_id"] / "status.json"
    return path.exists() and int(read_json(path).get("completed_updates", -1)) == 100


def official_complete(branch: dict[str, Any]) -> bool:
    path = (
        RAW_ROOT
        / "branches"
        / branch["branch_id"]
        / "official_test"
        / "update_0100_summary.json"
    )
    return path.exists()


def parallel_queue(
    branches: list[dict[str, Any]],
    urls: list[str],
    label: str,
    official: bool = False,
) -> None:
    pending = deque(
        branch
        for branch in branches
        if not (official_complete(branch) if official else branch_complete(branch))
    )
    running: dict[int, tuple[subprocess.Popen, dict[str, Any], Any, Any]] = {}
    retries: dict[str, int] = {}
    while pending or running:
        for gpu in range(len(urls)):
            if gpu in running or not pending:
                continue
            branch = pending.popleft()
            branch_id = branch["branch_id"]
            log_dir = EXP_ROOT / "logs" / branch_id
            log_dir.mkdir(parents=True, exist_ok=True)
            attempt = retries.get(branch_id, 0) + 1
            stdout = (log_dir / f"{label}_attempt{attempt}.out").open(
                "a", encoding="utf-8"
            )
            stderr = (log_dir / f"{label}_attempt{attempt}.err").open(
                "a", encoding="utf-8"
            )
            script = "fast_official.py" if official else "fast_train_branch.py"
            command = [
                PYTHON,
                str(SRC / script),
                "--branch-id",
                branch_id,
                "--server-url",
                urls[gpu],
                "--gpu",
                str(gpu),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            process = subprocess.Popen(
                command,
                cwd=str(EXP_ROOT.parents[1]),
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            running[gpu] = (process, branch, stdout, stderr)
            print(f"{label}: GPU {gpu} started {branch_id}", flush=True)
        time.sleep(10)
        for gpu, value in list(running.items()):
            process, branch, stdout, stderr = value
            code = process.poll()
            if code is None:
                continue
            stdout.close()
            stderr.close()
            del running[gpu]
            branch_id = branch["branch_id"]
            if code == 0:
                print(f"{label}: GPU {gpu} completed {branch_id}", flush=True)
                continue
            retries[branch_id] = retries.get(branch_id, 0) + 1
            if retries[branch_id] >= 3:
                raise RuntimeError(f"{label}: {branch_id} failed three times")
            print(f"{label}: retrying {branch_id} after exit {code}", flush=True)
            pending.appendleft(branch)


def shared_base_development(client: VLLMRolloutClient) -> dict[str, Any]:
    summary_path = RAW_ROOT / "base_shared" / "development" / "update_0000_summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    rows = [
        dict(row)
        for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))
    ]
    client.use_base()
    raw_path = RAW_ROOT / "base_shared" / "development" / "update_0000.jsonl"
    scored = client.score_rows(rows, 8, stable_int("fast_base_development"), raw_path)
    summary = {
        "shared_across_all_discovery_branches": True,
        "split": "development",
        "update": 0,
        "instances": len(rows),
        "samples_per_instance": 8,
        **summarize_rollouts(scored),
    }
    atomic_json(summary_path, summary)
    return summary


def branches_for_discovery_batch(
    branch_manifest: dict[str, Any],
    discovery: dict[str, Any],
    batch: int,
) -> list[dict[str, Any]]:
    selected = next(
        item for item in discovery["discovery_batches"] if int(item["batch"]) == batch
    )
    ids = set(selected["candidate_ids"])
    result = [
        branch
        for branch in branch_manifest["branches"]
        if branch["stage"] == "discovery" and branch.get("candidate_id") in ids
    ]
    if batch == 1:
        direct = next(
            branch
            for branch in branch_manifest["branches"]
            if branch["stage"] == "discovery" and branch["kind"] == "direct"
        )
        return [direct] + sorted(result, key=lambda item: item["candidate_id"])
    return sorted(result, key=lambda item: item["candidate_id"])


def confirmation_branches(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [branch for branch in manifest["branches"] if branch["stage"] == "confirmation"]


def run() -> None:
    started = time.time()
    atomic_json(
        EXP_ROOT / "manifests" / "fast_protocol_runtime.json",
        {
            "protocol_version": 2,
            "started_at": started,
            "original_eta_days": [10, 15],
            "old_job_id": "40336174",
            "old_job_cancelled_after_rollouts": 272,
        },
    )
    gpu_count = int(os.environ.get("SELF_GROK_GPU_COUNT", "2"))
    base_port = int(os.environ.get("SELF_GROK_VLLM_BASE_PORT", "8100"))
    only_discovery_branch = os.environ.get("SELF_GROK_ONLY_DISCOVERY_BRANCH")
    only_confirmation_index = os.environ.get("SELF_GROK_ONLY_CONFIRMATION_INDEX")
    with VLLMServerPool(
        gpus=tuple(range(gpu_count)), base_port=base_port
    ) as urls:
        screened = run_all(urls)
        discovery = screened["discovery"]
        branch_manifest = initialize_branch_manifest(discovery)
        shared_base_development(VLLMRolloutClient(urls[0], 0))
        pair = None
        branches_run = 0
        for batch in (1, 2):
            tasks = branches_for_discovery_batch(
                branch_manifest, discovery, batch
            )
            if only_discovery_branch:
                tasks = [
                    branch for branch in tasks
                    if branch["branch_id"] == only_discovery_branch
                ]
                if not tasks:
                    continue
            parallel_queue(tasks, urls, f"discovery_batch_{batch}")
            if only_discovery_branch:
                return
            branches_run += sum(1 for branch in tasks if branch["kind"] == "candidate")
            if batch == 1:
                branches_run += 1
            pair = select_discovery_pair(batch)
            if pair is not None:
                break
        if pair is not None and only_confirmation_index is not None:
            branch_manifest = append_confirmation_branches(pair)
            confirm = confirmation_branches(branch_manifest)
            index = int(only_confirmation_index)
            if not 0 <= index < len(confirm):
                raise ValueError(f"confirmation index out of range: {index}")
            parallel_queue([confirm[index]], urls, "confirmation")
            return
        if pair is not None and os.environ.get("SELF_GROK_STOP_AFTER_DISCOVERY") == "1":
            branch_manifest = append_confirmation_branches(pair)
            atomic_json(
                EXP_ROOT / "manifests" / "stage1_handoff_ready.json",
                {"pair": pair, "branches_run": branches_run, "finished_at": time.time()},
            )
            return
        if pair is None:
            write_null_summary(
                "No pre-frozen matched pair showed verified target success plus a >=0.02 target-gain gap at update 100.",
                branches_run,
            )
            subprocess.run(
                [PYTHON, str(SRC / "fast_report.py")],
                cwd=str(EXP_ROOT.parents[1]),
                check=True,
            )
            atomic_json(
                EXP_ROOT / "manifests" / "fast_experiment_complete.json",
                {
                    "outcome": "not supported",
                    "stage1_complete": True,
                    "stage2_started": False,
                    "training_branches": branches_run,
                    "wall_seconds": time.time() - started,
                },
            )
            return
        branch_manifest = append_confirmation_branches(pair)
        confirm = confirmation_branches(branch_manifest)
        parallel_queue(confirm, urls, "confirmation")
        branches_run += len(confirm)
        statistical = analyze_confirmation()
        if statistical["official_test_authorized"]:
            prepare_sealed()
            parallel_queue(confirm, urls, "official_test", official=True)
        subprocess.run(
            [PYTHON, str(SRC / "fast_report.py")],
            cwd=str(EXP_ROOT.parents[1]),
            check=True,
        )
        atomic_json(
            EXP_ROOT / "manifests" / "fast_experiment_complete.json",
            {
                "outcome": (
                    "supported" if statistical["confirmation_supported"] else "not supported"
                ),
                "stage1_complete": True,
                "stage2_complete": True,
                "official_test_complete": bool(statistical["official_test_authorized"]),
                "training_branches": branches_run,
                "wall_seconds": time.time() - started,
            },
        )


if __name__ == "__main__":
    run()
