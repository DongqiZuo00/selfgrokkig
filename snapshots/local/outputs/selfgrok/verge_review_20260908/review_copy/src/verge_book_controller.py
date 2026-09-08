"""Durable, bounded Slurm dispatch. Only one primary wave can own GPUs at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from verge_book_protocol import SUITE, PRIMARY_ARMS, MAX_ROUNDS, make_config, version

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "raw_results" / SUITE


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)


def schedule():
    return [(arm, r) for r in range(MAX_ROUNDS) for arm in PRIMARY_ARMS]


def verified_round(arm, r):
    name = version(arm, r)
    record = read(EXP / "manifests" / f"{name}_complete.json")
    if (record.get("acceptance_only") or record.get("book_suite") != SUITE
            or record.get("book_arm") != arm or record.get("book_round") != r
            or record.get("official_test_opened") is not False):
        raise RuntimeError("Invalid completed book round")
    if len(record.get("artifacts", [])) < 12:
        raise RuntimeError("Incomplete per-round artifact inventory")
    root = (EXP / "raw_results" / name).resolve()
    for relative in record["artifacts"]:
        path = (EXP / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing, empty or foreign round artifact")
    for field in ("solver_checkpoint", "challenger_checkpoint"):
        path = (EXP / record[field]).resolve()
        if not path.is_relative_to((EXP / "checkpoints").resolve()):
            raise RuntimeError("Checkpoint outside experiment")
        if not (path / "verge_committed.json").is_file() or not (path / "adapter_model.safetensors").is_file():
            raise RuntimeError("Checkpoint is not committed")
    return record


def previous_pair(arm, r):
    if r == 0:
        return None
    name = version(arm, r - 1)
    complete = verified_round(arm, r - 1)
    prior = read(EXP / "manifests" / f"{name}.json")
    if complete.get("acceptance_only") or complete.get("book_suite") != SUITE:
        raise RuntimeError("Uncommitted or foreign predecessor")
    return {"config": prior, "selected_solver": complete["solver_checkpoint"],
            "selected_challenger": complete["challenger_checkpoint"]}


def dispatch(after=None):
    import fcntl
    incident = EXP / "manifests" / f"{SUITE}_sampling_incident.json"
    if incident.exists() and read(incident).get("block_dispatch", False):
        raise RuntimeError("Book dispatch blocked: shared endpoint RNG incident; see the incident record and obtain recovery authorization")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        protocol = read(EXP / "manifests" / f"{SUITE}_protocol.json")
        if SUITE == "verge_book_v2":
            approval = read(EXP / "manifests/verge_book_v2_restart_authorization.json")
            if approval.get("restart_authorized") is not True or protocol.get("sampling_protocol") != "per_prompt_disjoint_seed_ranges_v1":
                raise RuntimeError("Corrected restart lacks approval or independent-stream protocol")
        frozen_protocol = ROOT / "protocol_frozen.json"
        if frozen_protocol.exists() and read(frozen_protocol) != protocol:
            raise RuntimeError("Suite protocol changed after submission")
        if not frozen_protocol.exists():
            write(frozen_protocol, protocol)
        for arm, r in schedule():
            name = version(arm, r)
            marker = EXP / "manifests" / f"{name}_complete.json"
            if marker.exists():
                verified_round(arm, r)
                continue
            cfg = make_config(EXP, arm, r, previous_pair(arm, r))
            cfg_path = EXP / "manifests" / f"{name}.json"
            if cfg_path.exists() and read(cfg_path) != cfg:
                raise RuntimeError("Round configuration differs from the frozen predecessor")
            if not cfg_path.exists():
                write(cfg_path, cfg)
            path = ROOT / "jobs" / f"{name}.json"
            jobs = read(path) if path.exists() else {"version": name, "arm": arm, "round": r}
            # A manual continuation also waits for the previous finish allocation
            # to leave. Never overlap a new wave with an old finalizer.
            active = ROOT / "active_wave.json"
            if after is None and active.exists():
                prior_wave = read(active)
                if prior_wave["version"] != name:
                    after = prior_wave["finish"]
            def submit(stage, options):
                args = ["sbatch", "--parsable", f"--job-name=vb-{arm}-r{r}-{stage}",
                        f"--export=ALL,VERGE_BOOK_SUITE={SUITE}"] + options
                args += [str(EXP / "scripts/verge_book_stage.sbatch"), name, stage]
                answer = subprocess.run(args, cwd=EXP.parents[1], text=True, capture_output=True, check=True)
                job_id = answer.stdout.strip().split(";")[0]
                if not job_id.isdigit():
                    raise RuntimeError("Uncertain sbatch response; inspect Slurm before retrying")
                return job_id
            if "prepare" not in jobs:
                dependencies = [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"] if after else []
                jobs["prepare"] = submit("prepare", dependencies)
                write(path, jobs)
            if "branches" not in jobs:
                jobs["branches"] = submit("train", ["--array=0-3%2", "--kill-on-invalid-dep=yes",
                                                     f"--dependency=afterok:{jobs['prepare']}"])
                write(path, jobs)
            if "finish" not in jobs:
                jobs["finish"] = submit("finish", ["--kill-on-invalid-dep=yes", f"--dependency=afterok:{jobs['branches']}"])
                write(path, jobs)
            write(ROOT / "active_wave.json", jobs)
            print(json.dumps(jobs))
            return jobs
        # This marker deliberately does NOT finish the manuscript experiment matrix.
        record = {"primary_block_complete": True, "suite_complete": False,
            "completed_at": time.time(), "outer_rounds": len(schedule()),
            "next_required_matrix": [x["id"] for x in protocol["required_matrix"] if x["id"] != "primary_multiround"]}
        write(ROOT / "PRIMARY_BLOCK_COMPLETE.json", record)
        print(json.dumps(record))
        return record


def status():
    result = {"suite": SUITE, "completed_rounds": [], "active_wave": None, "suite_complete": False}
    incident = EXP / "manifests" / f"{SUITE}_sampling_incident.json"
    if incident.exists():
        result["sampling_incident"] = read(incident)
        result["valid_for_final_statistical_inference"] = False
    for arm, r in schedule():
        name = version(arm, r)
        marker = EXP / "manifests" / f"{name}_complete.json"
        if marker.exists():
            record = verified_round(arm, r)
            result["completed_rounds"].append({"arm": arm, "round": r,
                "completed_at": record["completed_at"], "solver_checkpoint": record["solver_checkpoint"],
                "challenger_checkpoint": record["challenger_checkpoint"], "artifacts_verified": True})
    active = ROOT / "active_wave.json"
    if active.exists():
        result["active_wave"] = read(active)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "dispatch", "status"))
    parser.add_argument("--after")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps({"suite": SUITE, "schedule": schedule(), "submitted": False}, indent=2))
    elif args.action == "dispatch":
        dispatch(args.after)
    else:
        status()
