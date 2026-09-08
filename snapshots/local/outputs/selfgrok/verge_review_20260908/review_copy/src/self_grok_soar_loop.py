from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, read_json
from prepare_data import generate_distribution


PROTOCOL_PATH = EXP_ROOT / "manifests" / "self_grok_soar_mistral_v1.json"
STATE_PATH = EXP_ROOT / "manifests" / "self_grok_soar_mistral_state.json"
BRANCHES_PATH = EXP_ROOT / "manifests" / "self_grok_soar_mistral_runtime_branches.json"
SOAR_DATA_ROOT = DATA_ROOT / "self_grok_soar_mistral"
SOAR_RAW_ROOT = EXP_ROOT / "raw_results" / "self_grok_soar_mistral"


def protocol() -> dict[str, Any]:
    return read_json(PROTOCOL_PATH)


def state() -> dict[str, Any]:
    return read_json(STATE_PATH)


def outer_dir(outer: int) -> Path:
    return SOAR_RAW_ROOT / f"outer_{outer:03d}"


def generation_path(outer: int) -> Path:
    return outer_dir(outer) / "teacher_generation.json"


def decision_path(outer: int) -> Path:
    return outer_dir(outer) / "decision.json"


def validation_path(outer: int, role: str) -> Path:
    return outer_dir(outer) / "validation" / f"{role}.json"


def run_sbatch(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["sbatch", "--parsable", *arguments],
        # All Slurm log paths follow the repository's established convention
        # and are relative to the self-grok work root, not the experiment root.
        cwd=str(EXP_ROOT.parents[1]),
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip().split(";", 1)[0]


def submit_outer(outer: int) -> str:
    dispatch_path = outer_dir(outer) / "outer_submission.json"
    if dispatch_path.exists():
        return str(read_json(dispatch_path)["job_id"])
    job_id = run_sbatch(
        [str(EXP_ROOT / "scripts" / "self_grok_soar_outer.sbatch"), str(outer)]
    )
    atomic_json(
        dispatch_path,
        {"outer": outer, "job_id": job_id, "submitted_at_unix": time.time()},
    )
    return job_id


def dispatch_students(outer: int) -> dict[str, Any]:
    path = outer_dir(outer) / "student_submission.json"
    if path.exists():
        return read_json(path)
    cfg = protocol()
    count = int(cfg["teacher"]["candidates_per_outer_step"]) + 1
    concurrency = int(cfg["resources"]["student_array_concurrency"])
    array_job = run_sbatch(
        [
            f"--array=0-{count - 1}%{concurrency}",
            str(EXP_ROOT / "scripts" / "self_grok_soar_students.sbatch"),
            str(outer),
        ]
    )
    decision_job = run_sbatch(
        [
            f"--dependency=afterok:{array_job}",
            str(EXP_ROOT / "scripts" / "self_grok_soar_decide.sbatch"),
            str(outer),
        ]
    )
    result = {
        "outer": outer,
        "array_job_id": array_job,
        "decision_job_id": decision_job,
        "array": f"0-{count - 1}%{concurrency}",
        "submitted_at_unix": time.time(),
    }
    atomic_json(path, result)
    return result


def dispatch_validation(outer: int) -> dict[str, Any]:
    path = outer_dir(outer) / "validation_submission.json"
    if path.exists():
        return read_json(path)
    array_job = run_sbatch(
        [
            "--array=0-1%2",
            str(EXP_ROOT / "scripts" / "self_grok_soar_validate.sbatch"),
            str(outer),
        ]
    )
    analysis_job = run_sbatch(
        [
            f"--dependency=afterok:{array_job}",
            str(EXP_ROOT / "scripts" / "self_grok_soar_after_validate.sbatch"),
            str(outer),
        ]
    )
    result = {
        "outer": outer,
        "array_job_id": array_job,
        "analysis_job_id": analysis_job,
        "submitted_at_unix": time.time(),
    }
    atomic_json(path, result)
    return result


def initialize() -> None:
    cfg = protocol()
    SOAR_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SOAR_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    reward_path = SOAR_DATA_ROOT / "grounded_reward_train64"
    if not reward_path.exists():
        target = load_from_disk(str(DATA_ROOT / "target_splits" / "target_train"))
        count = int(cfg["inner_loop"]["grounded_reward_instances"])
        Dataset.from_list([dict(target[index]) for index in range(count)]).save_to_disk(
            str(reward_path)
        )
    if not BRANCHES_PATH.exists():
        initial = cfg["initial_student"]
        atomic_json(
            BRANCHES_PATH,
            {
                "protocol_version": cfg["protocol_version"],
                "backbone": cfg["backbone"],
                "branches": [
                    {
                        "branch_id": initial["branch_id"],
                        "kind": "root",
                        "role": "Mistral scope-preserving root",
                        "seed": 42,
                        "goal_updates": int(initial["scientific_update"]),
                        "training_data_path": str(DATA_ROOT / "root" / "basic_mix"),
                        "evaluation_updates": [int(initial["scientific_update"])],
                        "skip_development_evaluation": True,
                    }
                ],
            },
        )
    if not STATE_PATH.exists():
        initial = cfg["initial_student"]
        atomic_json(
            STATE_PATH,
            {
                "protocol_version": cfg["protocol_version"],
                "status": "active",
                "current_student_checkpoint": initial["checkpoint"],
                "current_student_scientific_update": int(initial["scientific_update"]),
                "current_student_branch_id": initial["branch_id"],
                "promotion_count": 0,
                "promotion_evidence": [],
                "completed_outer_steps": [],
                "created_at_unix": time.time(),
            },
        )
    print(json.dumps(state(), indent=2), flush=True)


def _allocate_counts(components: list[dict[str, Any]], total: int) -> list[int]:
    weights = [int(item["weight"]) for item in components]
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    counts = [max(1, int(value)) for value in raw]
    while sum(counts) < total:
        index = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    while sum(counts) > total:
        choices = [i for i, value in enumerate(counts) if value > 1]
        index = min(choices, key=lambda i: raw[i] - counts[i])
        counts[index] -= 1
    return counts


def materialize_curriculum(outer: int, candidate: int, spec: dict[str, Any]) -> Path:
    cfg = protocol()
    total = int(cfg["teacher"]["curriculum_rows"])
    components = list(spec["components"])
    counts = _allocate_counts(components, total)
    rows: list[dict[str, Any]] = []
    for component_index, (component, count) in enumerate(zip(components, counts)):
        level = str(component["level"])
        generated = generate_distribution(
            str(component["family"]),
            dict(cfg["teacher"]["levels"][level]),
            count,
            10_000_000 + outer * 10_000 + candidate * 100 + component_index,
            f"soar_o{outer:03d}_c{candidate:02d}",
        )
        rows.extend(generated)
    rows.sort(key=lambda row: str(row["id"]))
    path = SOAR_DATA_ROOT / f"outer_{outer:03d}" / f"candidate_{candidate:02d}"
    if not path.exists():
        Dataset.from_list(rows).save_to_disk(str(path))
    return path


def register_branches(outer: int, candidates: list[dict[str, Any]]) -> list[str]:
    cfg = protocol()
    current = state()
    inner = cfg["inner_loop"]
    goal = int(inner["updates"])
    common = {
        "seed": 42,
        "goal_updates": goal,
        "parent_checkpoint": current["current_student_checkpoint"],
        "parent_scientific_update": int(current["current_student_scientific_update"]),
        "inherit_optimizer_state": bool(inner["inherit_student_optimizer"]),
        "prompts_per_update": int(inner["prompts_per_update"]),
        "rollouts_per_prompt": int(inner["rollouts_per_prompt"]),
        "partial_weight": float(inner["partial_reward_weight"]),
        "evaluation_updates": [goal],
        "skip_development_evaluation": True,
        "grounded_reward_path": "data/self_grok_soar/grounded_reward_train64",
        "grounded_reward_instances": int(inner["grounded_reward_instances"]),
        "grounded_reward_rollouts_per_instance": int(
            inner["grounded_reward_rollouts_per_instance"]
        ),
        "scope_probe_path": "data/root/basic_mix",
        "scope_probe_instances": int(inner["scope_instances"]),
        "scope_probe_rollouts_per_instance": int(inner["scope_rollouts_per_instance"]),
        "training_rng_key": f"soar_o{outer:03d}_shared_train",
        "evaluation_rng_key": f"soar_o{outer:03d}_shared_eval",
    }
    direct_id = f"mistral_soar_o{outer:03d}__direct"
    branches: list[dict[str, Any]] = [
        {
            **common,
            "branch_id": direct_id,
            "kind": "grounded_meta_direct",
            "role": "matched target-only control",
            "target_update_offset": outer * goal,
            "evaluate_grounded_at_start": True,
        }
    ]
    branch_ids = [direct_id]
    for index, candidate in enumerate(candidates):
        branch_id = f"mistral_soar_o{outer:03d}__candidate_{index:02d}"
        data_path = materialize_curriculum(outer, index, candidate["spec"])
        branches.append(
            {
                **common,
                "branch_id": branch_id,
                "kind": "grounded_meta_candidate",
                "role": "target-blind Challenger curriculum",
                "candidate_id": f"soar_o{outer:03d}_c{index:02d}",
                "training_data_path": str(data_path),
                "evaluate_grounded_at_start": False,
            }
        )
        branch_ids.append(branch_id)

    manifest = read_json(BRANCHES_PATH)
    existing = {item["branch_id"] for item in manifest["branches"]}
    manifest["branches"].extend(item for item in branches if item["branch_id"] not in existing)
    atomic_json(BRANCHES_PATH, manifest)
    return branch_ids


def _summary(branch_id: str, split: str, update: int) -> dict[str, Any]:
    path = (
        EXP_ROOT
        / "raw_results"
        / "recipe_v1"
        / "branches"
        / branch_id
        / split
        / f"update_{update:04d}_summary.json"
    )
    if not path.exists():
        raise RuntimeError(f"missing completed summary: {path}")
    return read_json(path)


def _score_candidate(
    candidate_id: str,
    direct_target: dict[str, Any],
    direct_scope: dict[str, Any],
    goal: int,
) -> dict[str, Any]:
    cfg = protocol()
    reward_cfg = cfg["grounded_reward"]
    target = _summary(candidate_id, "grounded_reward", goal)
    scope = _summary(candidate_id, "scope_probe", goal)
    strict_gain = float(target["full_pass_rate"]) - float(direct_target["full_pass_rate"])
    soft_gain = float(target["mean_case_fraction"]) - float(direct_target["mean_case_fraction"])
    scope_strict_gain = float(scope["full_pass_rate"]) - float(direct_scope["full_pass_rate"])
    scope_soft_gain = float(scope["mean_case_fraction"]) - float(
        direct_scope["mean_case_fraction"]
    )
    scope_safe = (
        scope_strict_gain >= float(reward_cfg["scope_minimum_strict_gain"])
        and scope_soft_gain >= float(reward_cfg["scope_minimum_soft_gain"])
    )
    raw_reward = strict_gain + float(reward_cfg["soft_coefficient"]) * soft_gain
    teacher_reward = raw_reward if scope_safe else float(reward_cfg["scope_violation_reward"])
    return {
        "branch_id": candidate_id,
        "target_strict_rate": float(target["full_pass_rate"]),
        "target_soft_rate": float(target["mean_case_fraction"]),
        "strict_gain_vs_direct": strict_gain,
        "soft_gain_vs_direct": soft_gain,
        "scope_strict_gain_vs_direct": scope_strict_gain,
        "scope_soft_gain_vs_direct": scope_soft_gain,
        "scope_safe": scope_safe,
        "raw_grounded_reward": raw_reward,
        "teacher_reward": teacher_reward,
        "eligible": scope_safe,
    }


def decide(outer: int) -> None:
    path = decision_path(outer)
    if path.exists():
        print(json.dumps(read_json(path), indent=2), flush=True)
        return
    cfg = protocol()
    generation = read_json(generation_path(outer))
    goal = int(cfg["inner_loop"]["updates"])
    direct_id = generation["branch_ids"][0]
    direct_target = _summary(direct_id, "grounded_reward", goal)
    direct_scope = _summary(direct_id, "scope_probe", goal)
    scores = [
        _score_candidate(branch_id, direct_target, direct_scope, goal)
        for branch_id in generation["branch_ids"][1:]
    ]
    eligible = [item for item in scores if item["eligible"]]
    best = max(eligible, key=lambda item: item["teacher_reward"]) if eligible else None

    current = state()
    evidence = list(current.get("promotion_evidence", []))
    evidence.append(float(best["teacher_reward"]) if best else -1.0)
    promotion_cfg = cfg["promotion"]
    window_size = int(promotion_cfg["rolling_window"])
    window = evidence[-window_size:]
    enough_positive = sum(value > 0.0 for value in window) >= int(
        promotion_cfg["minimum_positive_steps"]
    )
    rolling_ready = (
        len(window) == window_size
        and enough_positive
        and sum(window) / len(window) > float(promotion_cfg["reward_threshold"])
    )
    immediate = bool(
        best
        and promotion_cfg["immediate_on_positive_strict_gain"]
        and float(best["strict_gain_vs_direct"]) > 0.0
    )
    promote = bool(best and (rolling_ready or immediate))
    previous_student = current["current_student_checkpoint"]
    promotion = None
    if promote:
        current["current_student_checkpoint"] = (
            f"checkpoints/{best['branch_id']}/resume_u{goal:04d}"
        )
        current["current_student_scientific_update"] = goal
        current["current_student_branch_id"] = best["branch_id"]
        current["promotion_count"] = int(current["promotion_count"]) + 1
        current["promotion_evidence"] = []
        promotion = {
            "branch_id": best["branch_id"],
            "checkpoint": current["current_student_checkpoint"],
            "reason": "positive strict gain" if immediate else "rolling grounded gain",
            "previous_student_checkpoint": previous_student,
        }
    else:
        current["promotion_evidence"] = evidence
    completed = list(current.get("completed_outer_steps", []))
    completed.append(outer)
    current["completed_outer_steps"] = sorted(set(completed))
    current["last_decided_outer"] = outer
    atomic_json(STATE_PATH, current)

    result = {
        "outer": outer,
        "direct_branch_id": direct_id,
        "direct_target_strict_rate": float(direct_target["full_pass_rate"]),
        "direct_target_soft_rate": float(direct_target["mean_case_fraction"]),
        "candidates": scores,
        "best_candidate": best,
        "promotion_window": window,
        "rolling_ready": rolling_ready,
        "promotion": promotion,
        "teacher_rewards": [float(item["teacher_reward"]) for item in scores],
    }
    atomic_json(path, result)
    if promote:
        dispatch_validation(outer)
    else:
        cleanup_outer(outer, keep_checkpoint=None)
        continue_or_finish(outer, decisive=False)
    print(json.dumps(result, indent=2), flush=True)


def _compress_rollouts(branch_id: str) -> None:
    root = EXP_ROOT / "raw_results" / "recipe_v1" / "branches" / branch_id
    if not root.exists():
        return
    for path in root.rglob("*.jsonl"):
        compressed = path.with_suffix(path.suffix + ".gz")
        if compressed.exists():
            path.unlink()
            continue
        with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target)
        path.unlink()


def cleanup_outer(outer: int, keep_checkpoint: str | None) -> None:
    generation = read_json(generation_path(outer))
    checkpoint_root = (EXP_ROOT / "checkpoints").resolve()
    for branch_id in generation["branch_ids"]:
        _compress_rollouts(branch_id)
        if branch_id == keep_checkpoint:
            continue
        path = (checkpoint_root / branch_id).resolve()
        if checkpoint_root not in path.parents:
            raise RuntimeError(f"refusing checkpoint cleanup outside root: {path}")
        if path.exists():
            shutil.rmtree(path)


def continue_or_finish(outer: int, decisive: bool) -> None:
    cfg = protocol()
    current = state()
    maximum = int(cfg["teacher"]["maximum_outer_steps"])
    if decisive:
        current["status"] = "decisive_held_out_success"
        atomic_json(STATE_PATH, current)
        atomic_json(
            EXP_ROOT / "manifests" / "self_grok_soar_mistral_loop_complete.json",
            {"status": current["status"], "outer": outer, "state": current},
        )
        return
    next_outer = outer + 1
    if next_outer >= maximum:
        current["status"] = "outer_budget_exhausted"
        atomic_json(STATE_PATH, current)
        atomic_json(
            EXP_ROOT / "manifests" / "self_grok_soar_mistral_loop_complete.json",
            {"status": current["status"], "outer": outer, "state": current},
        )
        return
    submit_outer(next_outer)


def after_validate(outer: int) -> None:
    output = outer_dir(outer) / "validation_decision.json"
    if output.exists():
        print(json.dumps(read_json(output), indent=2), flush=True)
        return
    decision = read_json(decision_path(outer))
    direct = read_json(validation_path(outer, "direct"))
    promoted = read_json(validation_path(outer, "promoted"))
    cfg = protocol()
    reward_cfg = cfg["grounded_reward"]
    strict_gain = float(promoted["development"]["full_pass_rate"]) - float(
        direct["development"]["full_pass_rate"]
    )
    soft_gain = float(promoted["development"]["mean_case_fraction"]) - float(
        direct["development"]["mean_case_fraction"]
    )
    scope_strict_gain = float(promoted["scope"]["full_pass_rate"]) - float(
        direct["scope"]["full_pass_rate"]
    )
    scope_soft_gain = float(promoted["scope"]["mean_case_fraction"]) - float(
        direct["scope"]["mean_case_fraction"]
    )
    scope_safe = (
        scope_strict_gain >= float(reward_cfg["scope_minimum_strict_gain"])
        and scope_soft_gain >= float(reward_cfg["scope_minimum_soft_gain"])
    )
    decisive = bool(
        int(promoted["development"]["full_pass_count"]) > 0
        and strict_gain > 0.0
        and scope_safe
    )
    result = {
        "outer": outer,
        "promoted_branch_id": decision["promotion"]["branch_id"],
        "development_strict_gain_vs_direct": strict_gain,
        "development_soft_gain_vs_direct": soft_gain,
        "scope_strict_gain_vs_direct": scope_strict_gain,
        "scope_soft_gain_vs_direct": scope_soft_gain,
        "scope_safe": scope_safe,
        "decisive": decisive,
    }
    atomic_json(output, result)
    cleanup_outer(outer, keep_checkpoint=decision["promotion"]["branch_id"])
    continue_or_finish(outer, decisive=decisive)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    dispatch_parser = subparsers.add_parser("dispatch-students")
    dispatch_parser.add_argument("--outer", type=int, required=True)
    decide_parser = subparsers.add_parser("decide")
    decide_parser.add_argument("--outer", type=int, required=True)
    after_parser = subparsers.add_parser("after-validate")
    after_parser.add_argument("--outer", type=int, required=True)
    submit_parser = subparsers.add_parser("submit-outer")
    submit_parser.add_argument("--outer", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize()
    elif args.command == "dispatch-students":
        print(json.dumps(dispatch_students(args.outer), indent=2), flush=True)
    elif args.command == "decide":
        decide(args.outer)
    elif args.command == "after-validate":
        after_validate(args.outer)
    elif args.command == "submit-outer":
        print(submit_outer(args.outer), flush=True)


if __name__ == "__main__":
    main()
