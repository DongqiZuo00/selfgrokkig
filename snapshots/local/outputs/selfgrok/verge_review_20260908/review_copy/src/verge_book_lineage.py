"""Cross-round pointer and optimizer-counter checks; no weight comparisons or model sampling."""
import argparse
from verge_book_controller import EXP, read, write, verified_round
from verge_book_protocol import SUITE, version


def optimizer_counters(saved, expected_prior, expected_lr):
    runtime = saved["runtime_state"]
    if runtime["teacher_steps_prior"] != expected_prior:
        raise RuntimeError("Teacher cumulative-step counter did not cross the round boundary")
    opt = saved["optimizer"]
    steps = [int(s["step"]) for s in opt["state"].values()]
    if expected_prior == 0 and steps:
        raise RuntimeError("An untrained teacher unexpectedly has optimizer moments")
    if expected_prior and (not steps or max(steps) != expected_prior or min(steps) < 1):
        raise RuntimeError("Teacher moment-step counters disagree with its retained history")
    if any(abs(g["lr"] - expected_lr) > 1e-15 or g["weight_decay"] != 0 for g in opt["param_groups"]):
        raise RuntimeError("Teacher optimizer hyperparameters changed during inheritance")
    return {"expected_cumulative_steps": expected_prior, "parameter_states": len(steps),
            "distinct_parameter_step_counters": sorted(set(steps))}


def check(name):
    cfg = read(EXP / "manifests" / f"{name}.json")
    arm, r = cfg["book_arm"], cfg["book_round"]
    if version(arm, r) != name or cfg["book_suite"] != SUITE:
        raise RuntimeError("Foreign or unbounded round")
    if cfg["prior_book_versions"] != [version(arm, j) for j in range(r)]:
        raise RuntimeError("Archive does not contain exactly the preceding same-arm rounds")
    prior_steps = 0
    if r:
        previous = verified_round(arm, r - 1)
        if cfg["solver_start"] != previous["solver_checkpoint"] or cfg["challenger_start_checkpoint"] != previous["challenger_checkpoint"]:
            raise RuntimeError("Round starts from a different committed pair")
        teacher = read(EXP / "raw_results" / version(arm, r - 1) / "challenger_update.json")
        prior_steps = teacher["cumulative_optimizer_steps"]
    else:
        if cfg["solver_start"] != f"checkpoints/{SUITE}_initial_solver/resume_u0000" or cfg["challenger_start_checkpoint"] != f"checkpoints/{SUITE}_initial_challenger/resume_u0000":
            raise RuntimeError("Round zero is not the fresh autonomous root")
    teacher_path = EXP / "checkpoints" / (name + "_challenger") / "resume_u0000"
    if not (teacher_path / "verge_committed.json").exists():
        return {"version": name, "ready": False, "reason": "teacher initialization not committed yet"}
    import torch
    saved = torch.load(teacher_path / "state.pt", map_location="cpu", weights_only=False)
    counters = optimizer_counters(saved, prior_steps, cfg["challenger_learning_rate"])
    source = str(EXP / cfg["challenger_start_checkpoint"])
    if saved["runtime_state"]["source"] != source:
        raise RuntimeError("Teacher initialization source is not its declared checkpoint")
    return {"version": name, "ready": True, "same_arm_checkpoint_pair": True,
        "archive_rounds": len(cfg["prior_book_versions"]), "teacher_optimizer": counters,
        "new_model_draws": 0, "weight_tensors_compared": False, "checkpoint_hash_scans": 0,
        "scope": "Checkpoint pointers and retained optimizer counters, not a comparison of model weights"}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if not args.version.startswith(SUITE + "_") or "/" in args.version or "\\" in args.version or "." in args.version:
        raise ValueError("Expected a bounded book round name")
    result = check(args.version)
    if args.save:
        write(EXP / "raw_results" / args.version / "cross_round_state_check.json", result)
    print(json.dumps(result, indent=2))
