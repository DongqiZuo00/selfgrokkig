from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from common import EXP_ROOT, atomic_json, load_config, read_json


PYTHON = sys.executable
SRC = EXP_ROOT / "src"


def complete_to_budget(branch_id: str, budget: int) -> bool:
    path = EXP_ROOT / "raw_results" / "branches" / branch_id / "status.json"
    return path.exists() and int(read_json(path).get("completed_updates", -1)) >= budget


def evaluation_complete(branch_id: str, split: str, budget: int) -> bool:
    path = (
        EXP_ROOT
        / "raw_results"
        / "branches"
        / branch_id
        / split
        / f"budget_{budget}_summary.json"
    )
    return path.exists()


def parallel_queue(tasks: list[dict[str, Any]], command_builder, label: str) -> None:
    pending = deque(tasks)
    running: dict[int, tuple[subprocess.Popen, dict[str, Any], Any, Any]] = {}
    retries: dict[str, int] = {}
    while pending or running:
        for gpu in (0, 1):
            if gpu in running or not pending:
                continue
            task = pending.popleft()
            task_id = task["branch_id"]
            log_dir = EXP_ROOT / "logs" / task_id
            log_dir.mkdir(parents=True, exist_ok=True)
            attempt = retries.get(task_id, 0) + 1
            stdout = (log_dir / f"{label}_attempt{attempt}.out").open("a", encoding="utf-8")
            stderr = (log_dir / f"{label}_attempt{attempt}.err").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["HF_HOME"] = str(EXP_ROOT.parents[1] / "caches" / "huggingface")
            command = command_builder(task)
            process = subprocess.Popen(
                command,
                cwd=str(EXP_ROOT.parents[1]),
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            running[gpu] = (process, task, stdout, stderr)
            print(f"{label}: GPU {gpu} started {task_id}: {' '.join(command)}", flush=True)
        time.sleep(10)
        for gpu, value in list(running.items()):
            process, task, stdout, stderr = value
            code = process.poll()
            if code is None:
                continue
            stdout.close()
            stderr.close()
            del running[gpu]
            task_id = task["branch_id"]
            if code == 0:
                print(f"{label}: GPU {gpu} completed {task_id}", flush=True)
                continue
            retries[task_id] = retries.get(task_id, 0) + 1
            if retries[task_id] >= 3:
                raise RuntimeError(f"{label}: {task_id} failed three times; inspect branch log")
            print(f"{label}: retrying {task_id} after exit {code}", flush=True)
            pending.appendleft(task)


def train_all(frozen: dict[str, Any], budget: int) -> None:
    tasks = [
        branch for branch in frozen["branches"] if not complete_to_budget(branch["branch_id"], budget)
    ]
    parallel_queue(
        tasks,
        lambda branch: [
            PYTHON,
            str(SRC / "train_branch.py"),
            "--branch-id",
            branch["branch_id"],
            "--goal-updates",
            str(budget),
        ],
        f"train_to_{budget}",
    )


def evaluate_all(frozen: dict[str, Any], split: str, budget: int) -> None:
    tasks = [
        branch
        for branch in frozen["branches"]
        if not evaluation_complete(branch["branch_id"], split, budget)
    ]
    parallel_queue(
        tasks,
        lambda branch: [
            PYTHON,
            str(SRC / "evaluate_branch.py"),
            "--branch-id",
            branch["branch_id"],
            "--budget",
            str(budget),
            "--split",
            split,
        ],
        f"{split}_{budget}",
    )


def candidate_confirmed(frozen: dict[str, Any], budget: int) -> bool:
    for branch in frozen["branches"]:
        if branch["kind"] != "candidate":
            continue
        path = (
            EXP_ROOT
            / "raw_results"
            / "branches"
            / branch["branch_id"]
            / "confirmation"
            / f"budget_{budget}_summary.json"
        )
        if int(read_json(path).get("confirmed_full_pass_count", 0)) > 0:
            return True
    return False


def run_stage(script: str) -> None:
    subprocess.run(
        [PYTHON, str(SRC / script)],
        cwd=str(EXP_ROOT.parents[1]),
        check=True,
        env=os.environ.copy(),
    )


def main() -> None:
    config = load_config()
    run_stage("prepare_data.py")
    run_stage("audit_and_select.py")
    frozen = read_json(EXP_ROOT / "manifests" / "frozen_manifest.json")
    if len(frozen["branches"]) != 27:
        raise RuntimeError(f"protocol requires 27 branches, got {len(frozen['branches'])}")
    budget = int(config["training"]["initial_total_updates"])
    final_path = EXP_ROOT / "manifests" / "final_common_budget.json"
    if final_path.exists():
        budget = int(read_json(final_path)["budget"])
    else:
        while True:
            train_all(frozen, budget)
            evaluate_all(frozen, "confirmation", budget)
            if candidate_confirmed(frozen, budget):
                atomic_json(
                    final_path,
                    {
                        "budget": budget,
                        "stopping_reason": "at least one candidate-to-PREPEND branch has nonzero hidden-confirmed success",
                        "all_branches_trained_to_budget": True,
                    },
                )
                break
            budget += int(config["training"]["extension_block_updates"])
            print(f"no candidate confirmed at common budget; extending all branches to {budget}", flush=True)
    run_stage("prepare_sealed_test.py")
    evaluate_all(frozen, "official_test", budget)
    run_stage("analyze.py")
    atomic_json(
        EXP_ROOT / "manifests" / "experiment_complete.json",
        {"budget": budget, "branches": 27, "status": "complete"},
    )


if __name__ == "__main__":
    main()
