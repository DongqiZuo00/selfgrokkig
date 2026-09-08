"""Slurm allocation ledger for registered book jobs; no samples, hashes or mutations of jobs."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from verge_book_controller import EXP, ROOT, read, write, schedule
from verge_book_protocol import version, SUITE

FIELDS = ("JobID", "State", "ElapsedRaw", "AllocCPUS", "AllocTRES", "MaxRSS")


def memory_bytes(value):
    if not value:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTPE]?)(?:[nc])?", value)
    if not match:
        raise ValueError(f"Unknown Slurm memory unit: {value}")
    return float(match[1]) * 1024 ** ("KMGTPE".index(match[2]) + 1 if match[2] else 0)


def tres(value):
    result = {}
    for item in value.split(","):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = val
    return result


def parse_sacct(output):
    result = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(FIELDS):
            raise ValueError("Unexpected sacct columns")
        row = dict(zip(FIELDS, values))
        result[row["JobID"]] = row
    return result


def summarize(accounting, registered):
    allocations, missing = [], []
    for job_id, metadata in registered.items():
        expected = [f"{job_id}_{i}" for i in range(4)] if metadata["stage"] == "branches" else [job_id]
        for identity in expected:
            row = accounting.get(identity)
            if row is None:
                # Pending array ranges need not yet have individual sacct rows.
                missing.append(identity)
                continue
            resources = tres(row["AllocTRES"])
            total_gpu = resources.get("gres/gpu")
            typed_gpus = [int(v) for k, v in resources.items() if k.startswith("gres/gpu:")]
            gpu = int(total_gpu) if total_gpu is not None else sum(typed_gpus)
            elapsed = int(row["ElapsedRaw"] or 0)
            cpus = int(row["AllocCPUS"] or 0)
            batch = accounting.get(identity + ".batch", {})
            peak = memory_bytes(batch.get("MaxRSS", ""))
            allocations.append({**metadata, "job_id": identity, "state": row["State"],
                "elapsed_seconds": elapsed, "allocated_cpus": cpus, "allocated_gpus": gpu,
                "allocated_cpu_memory_bytes": memory_bytes(resources.get("mem", "")),
                "gpu_allocation_hours": gpu * elapsed / 3600,
                "cpu_allocation_core_hours": cpus * elapsed / 3600,
                "batch_max_rss_bytes": peak})
    return {"allocations": allocations, "missing_individual_accounting_rows": missing,
        "gpu_allocation_hours": sum(r["gpu_allocation_hours"] for r in allocations),
        "cpu_allocation_core_hours": sum(r["cpu_allocation_core_hours"] for r in allocations),
        "accounting_complete": not missing and all(r["state"].split()[0] in
            {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"} for r in allocations),
        "all_registered_jobs_successful": not missing and all(r["state"] == "COMPLETED" for r in allocations),
        "cost_definition": "Allocated GPU-hours and CPU-core-hours; not active utilization, FLOPs or matched end-to-end compute",
        "memory_definition": "batch MaxRSS is Slurm's reported maximum task RSS, not guaranteed job-cgroup peak; do not sum historical peaks as simultaneous memory use"}


def registry():
    jobs = {}
    for arm, round_index in schedule():
        name = version(arm, round_index)
        path = ROOT / "jobs" / f"{name}.json"
        if not path.exists():
            continue
        record = read(path)
        for stage in ("prepare", "branches", "finish"):
            identity = record.get(stage)
            if identity is None:
                continue
            if not isinstance(identity, str) or not identity.isdigit():
                raise RuntimeError("Registered job ID must be numeric")
            if identity in jobs:
                raise RuntimeError("Job registered under multiple waves")
            jobs[identity] = {"version": name, "arm": arm, "round": round_index, "stage": stage}
    return jobs


def collect():
    jobs = registry()
    if not jobs:
        raise RuntimeError("No book jobs registered")
    response = subprocess.run(["sacct", "--noheader", "--parsable2", "--units=K",
        "-j", ",".join(jobs), "--format=" + ",".join(FIELDS)],
        cwd=EXP, capture_output=True, text=True, check=True)
    result = summarize(parse_sacct(response.stdout), jobs)
    result.update(observed_at_unix=time.time(), registered_parent_jobs=len(jobs),
        scope=f"Only the primary book jobs recorded in raw_results/{SUITE}/jobs; unrelated account jobs excluded",
        suite_complete=False, new_model_draws=0, checkpoint_hash_scans=0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Write a current resource snapshot inside the book output directory")
    args = parser.parse_args()
    result = collect()
    if args.save:
        write(ROOT / "resource_accounting_latest.json", result)
    print(json.dumps(result, indent=2))
