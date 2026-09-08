"""Three bounded direct-control arms, six array waves, no primary overlap.

Requires an explicitly written plan; this module never chooses warm-up length.
It is not invoked by the running primary controller. Each array element runs its
own prepare/train sequence and releases its one GPU when done.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from verge_control_block import LABELS, validate_block, phases_for, validate_launch, validate_prepared
from verge_control_protocol import segment_config

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "raw_results/verge_control_v1"
PLAN = "manifests/verge_control_v1_direct_protocol.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name+".")
    with os.fdopen(fd,"w",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name,path)


def version(label, round_index):
    if label not in LABELS or type(round_index) is not int or not 0 <= round_index < 6:
        raise ValueError("Unknown or unbounded direct-control segment")
    return f"verge_control_v1_{label}_r{round_index:02d}"


def primary_finished():
    marker = EXP / "raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json"
    if not marker.is_file() or read(marker).get("primary_block_complete") is not True or read(marker).get("outer_rounds") != 24:
        raise RuntimeError("Primary block is still active; no direct-control submission")
    finish = read(EXP / "raw_results/verge_book_v2/active_wave.json")["finish"]
    if not isinstance(finish,str) or not finish.isdigit():
        raise RuntimeError("Missing final primary dependency ID")
    return finish


def completed(label, r, *, deep=False):
    name = version(label,r)
    directory = EXP / "raw_results" / name
    path = directory / "branches/0/complete.json"
    if not path.is_file():
        return None
    cfg = read(EXP / "manifests" / (name+".json"))
    validate_launch(EXP,cfg)
    frozen, result = read(directory / "round_frozen.json"), read(path)
    validate_prepared(cfg,frozen)
    expected_prefix = f"checkpoints/{name}_direct/resume_u"
    checkpoint = result.get("checkpoint","")
    if (result.get("control_segment_verified") is not True or result.get("control_label") != label
            or result.get("control_comparison_complete") is not False
            or result["training"]["train_tokens"] != cfg["train_tokens_per_branch"]
            or not 0 < result["training"]["update"] <= 100
            or not checkpoint.startswith(expected_prefix)
            or len(checkpoint) != len(expected_prefix)+4 or not checkpoint[-4:].isdigit()):
        raise RuntimeError("Invalid direct-control completion record")
    source = (EXP / checkpoint).resolve()
    if not source.is_relative_to((EXP / "checkpoints").resolve()):
        raise RuntimeError("Foreign control output checkpoint")
    for path in (source / "verge_committed.json", source / "adapter_model.safetensors", source / "state.pt",
                 directory / "branches/0/target.jsonl", directory / "branches/0/scope.jsonl"):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Incomplete control output artifacts")
    if (int(checkpoint[-4:]) != result["training"]["update"]
            or read(source / "verge_committed.json").get("update") != result["training"]["update"]):
        raise RuntimeError("Control checkpoint and completed training state disagree")
    if deep:
        from verge_control_verify import validate_segment
        validate_segment(frozen,result,directory / "branches/0")
    return result


@contextmanager
def dispatch_lock():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT / "controller.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def submit(stage,r,dependency):
    version("binary",r)  # Validate the shared bounded round index.
    if not isinstance(dependency,str) or not dependency.isdigit():
        raise ValueError("Every new control wave requires an exact prior job dependency")
    args = ["sbatch","--parsable",f"--dependency=afterok:{dependency}","--kill-on-invalid-dep=yes",
            "--export=ALL,VERGE_BOOK_SUITE=verge_book_v2,VERGE_CONTROL_SUITE=verge_control_v1",
            f"--job-name=vc-r{r}-{stage}"]
    if stage == "workers":
        args += ["--array=0-2%2",str(EXP / "scripts/verge_control_workers.sbatch"),str(r)]
    elif stage == "coordinator":
        args += [str(EXP / "scripts/verge_control_coordinate.sbatch"),str(r)]
    else:
        raise ValueError("Unknown control stage")
    intent = ROOT / "submission_intents" / f"round_{r:02d}_{stage}.json"
    if intent.exists():
        saved = read(intent)
        if saved.get("arguments") != args:
            raise RuntimeError("Submission arguments changed after a recorded intent")
        if isinstance(saved.get("confirmed_job_id"),str) and saved["confirmed_job_id"].isdigit():
            return saved["confirmed_job_id"]
        raise RuntimeError("Unresolved prior submission; inspect Slurm before any retry")
    # If sbatch succeeds but the process/connection dies, do not blindly submit
    # another allocation. A subsequent monitor can reconcile this recorded intent.
    write(intent,{"arguments":args,"created_at":time.time(),"confirmed_job_id":None})
    response = subprocess.run(args,cwd=EXP.parents[1],capture_output=True,text=True,check=True)
    job = response.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Uncertain sbatch response; inspect Slurm before retrying")
    write(intent,{"arguments":args,"confirmed_job_id":job,"confirmed_at":time.time()})
    return job


def dispatch():
    if os.environ.get("VERGE_BOOK_SUITE") != "verge_book_v2" or os.environ.get("VERGE_CONTROL_SUITE") != "verge_control_v1":
        raise RuntimeError("Both explicit suite variables are required")
    dependency = primary_finished()
    if not (EXP / PLAN).is_file():
        raise RuntimeError("No pre-execution direct-control plan; no implicit warm-up choice")
    block = validate_block(read(EXP / PLAN))
    with dispatch_lock():
        frozen_plan = ROOT / "protocol_frozen.json"
        if frozen_plan.exists() and read(frozen_plan) != block:
            raise RuntimeError("Direct-control plan changed after first dispatch")
        if not frozen_plan.exists():
            write(frozen_plan,block)
        for r in range(6):
            outcomes = [completed(label,r) for label in LABELS]
            jobs_path = ROOT / "jobs" / f"round_{r:02d}.json"
            if all(outcomes):
                # This CPU coordinator must finish before the next GPU wave.
                dependency = read(jobs_path)["coordinator"]
                continue
            for label in LABELS:
                start = "checkpoints/verge_book_v2_initial_solver/resume_u0000" if r == 0 else completed(label,r-1)["checkpoint"]
                cfg = segment_config(EXP,label,r,phases=phases_for(block,label,r),
                    starting_checkpoint=start,block_protocol=PLAN)
                validate_launch(EXP,cfg)
                path = EXP / "manifests" / (version(label,r)+".json")
                if path.exists() and read(path) != cfg:
                    raise RuntimeError("Control segment changed during recovery")
                if not path.exists():
                    write(path,cfg)
            jobs = read(jobs_path) if jobs_path.exists() else {"round":r,"labels":list(LABELS)}
            if "workers" not in jobs:
                jobs["workers"] = submit("workers",r,dependency)
                write(jobs_path,jobs)
            if "coordinator" not in jobs:
                jobs["coordinator"] = submit("coordinator",r,jobs["workers"])
                write(jobs_path,jobs)
            write(ROOT / "active_wave.json",jobs)
            return jobs
        # Full finalization is an explicit later step, never mistaken for book completion.
        record = {"all_direct_control_segments_finished":True,"segments":18,
                  "control_block_complete":False,"suite_complete":False,"observed_at":time.time()}
        write(ROOT / "SEGMENTS_FINISHED.json",record)
        return record


def coordinate(r):
    if type(r) is not int or not 0 <= r < 6:
        raise ValueError("Invalid control round")
    for label in LABELS:
        if completed(label,r,deep=True) is None:
            raise RuntimeError("Do not advance an incomplete control wave")
    return dispatch()


def status():
    return {"completed_segments":[version(label,r) for r in range(6) for label in LABELS if completed(label,r)],
            "required_segments":18,"active_wave":read(ROOT / "active_wave.json") if (ROOT / "active_wave.json").exists() else None,
            "control_block_complete":False,"suite_complete":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("status","dispatch","coordinate"))
    parser.add_argument("--round",type=int)
    args = parser.parse_args()
    result = status() if args.action == "status" else dispatch() if args.action == "dispatch" else coordinate(args.round)
    print(json.dumps(result,indent=2))
