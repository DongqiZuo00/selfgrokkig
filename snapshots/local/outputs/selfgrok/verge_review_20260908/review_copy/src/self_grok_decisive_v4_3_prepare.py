from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from common import EXP_ROOT, atomic_json, read_json, stable_int
from generation import summarize_rollouts
from prepend_conditions import bottleneck_index, condition_profile, required_prefix
from prepare_data import generate_distribution
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


VERSION = os.environ.get(
    "SELF_GROK_DECISIVE_VERSION", "self_grok_decisive_mistral_v4_3"
)
VERSION_TAG = VERSION.rsplit("mistral_", 1)[-1]
BRANCH_PREFIX = VERSION.removeprefix("self_grok_")
CFG_PATH = EXP_ROOT / "manifests" / f"{VERSION}.json"
RUNTIME_PATH = EXP_ROOT / "manifests" / f"{VERSION}_runtime_branches.json"
ROOT = EXP_ROOT / "raw_results" / VERSION
DATA = EXP_ROOT / "data" / VERSION


def apply_prepend_curriculum(
    rows: list[dict[str, Any]], spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Materialize an explicit finite-input tape task without hidden test slicing.

    ``prepend_curriculum_operation`` defaults to the original PREPEND transform.
    The prerequisite operations are deliberately complete tasks with their own
    strict verifier. ``consume`` removes the allowed input, and ``replace`` removes
    it before writing the requested one-symbol prefix.  The two branch operations
    retain true PREPEND on the opposite-color input while growing the same-color
    route from consume-only to one emitted prefix.  They expose conditional routing
    separately from the final replay step.
    """
    raw_inputs = spec.get("prepend_curriculum_inputs")
    if raw_inputs is None:
        return rows
    rewritten = []
    for source in rows:
        row = dict(source)
        prefix = required_prefix(row)
        if len(prefix) != 1 or prefix not in {"R", "B"}:
            raise RuntimeError(
                "finite PREPEND curriculum currently requires a one-symbol R/B prefix"
            )
        opposite = "B" if prefix == "R" else "R"
        inputs = []
        for raw in raw_inputs:
            token = str(raw)
            if token == "$opposite":
                value = opposite
            elif token == "$prefix":
                value = prefix
            elif token == "$opposite_then_prefix":
                value = opposite + prefix
            else:
                value = token
            if any(color not in {"R", "B"} for color in value):
                raise RuntimeError(f"invalid finite PREPEND input: {value!r}")
            if value not in inputs:
                inputs.append(value)
        if not inputs:
            raise RuntimeError("finite PREPEND curriculum must include at least one input")
        operation = str(spec.get("prepend_curriculum_operation", "prepend"))
        if operation == "consume":
            outputs = {value: "" for value in inputs}
            task_body = (
                "Remove the complete input tape and then accept the robot. The expected "
                "output is the empty tape for every allowed input."
            )
        elif operation == "replace":
            outputs = {value: prefix for value in inputs}
            task_body = (
                f"Replace the complete input tape with exactly '{prefix}' and then "
                "accept the robot."
            )
        elif operation == "replace_repeat":
            outputs = {value: prefix + prefix for value in inputs}
            task_body = (
                f"Replace the complete input tape with exactly '{prefix + prefix}' "
                "and then accept the robot."
            )
        elif operation == "branch_consume_same":
            required = {opposite, prefix}
            if set(inputs) != required:
                raise RuntimeError(
                    "branch_consume_same requires exactly $opposite and $prefix"
                )
            outputs = {
                value: ("" if value == prefix else prefix + value)
                for value in inputs
            }
            task_body = (
                "Use separate input-color routes. Preserve true PREPEND on the "
                "opposite-color input and consume the same-color input."
            )
        elif operation == "branch_replace_same":
            required = {opposite, prefix}
            if set(inputs) != required:
                raise RuntimeError(
                    "branch_replace_same requires exactly $opposite and $prefix"
                )
            outputs = {
                value: (prefix if value == prefix else prefix + value)
                for value in inputs
            }
            task_body = (
                "Use separate input-color routes. Preserve true PREPEND on the "
                "opposite-color input and emit one prefix symbol on the same-color "
                "input."
            )
        elif operation == "prepend":
            outputs = {value: prefix + value for value in inputs}
            task_body = (
                f"Put '{prefix}' at the beginning of the tape while preserving the "
                "complete input tape in its original order."
            )
        else:
            raise RuntimeError(f"unknown finite PREPEND curriculum operation: {operation}")
        row["ground_truth"] = [
            {
                "input": value,
                "expected_output": outputs[value],
                "expected_accepted": True,
                "check_output": True,
                "description": (
                    f"Allowed input {value!r} -> output {outputs[value]!r}"
                ),
            }
            for value in inputs
        ]
        rendered_inputs = ", ".join("<empty>" if not value else value for value in inputs)
        prompt_mode = str(spec.get("finite_curriculum_prompt_mode", "operation"))
        if prompt_mode == "io_table":
            table_lines = [
                (
                    f"- Input {'<empty>' if not value else value} -> "
                    f"Output {'<empty>' if not outputs[value] else outputs[value]}"
                )
                for value in inputs
            ]
            task_body = (
                "Transform each input tape according to this exact finite input-output "
                "table:\n" + "\n".join(table_lines) +
                "\nEvery listed input must reach END with exactly its listed output."
            )
            if bool(spec.get("prepend_curriculum_semantic_label", False)):
                task_body += (
                    f" This table is the finite-input form of PREPEND {prefix}: place "
                    f"'{prefix}' at the beginning while preserving the input order."
                )
            task = (
                "# Task\n"
                "Your task is to design a factory in the required code format.\n\n"
                f"{task_body} The input alphabet is red/blue. The allowed inputs are "
                f"exactly: {rendered_inputs}. Correctness outside this stated finite "
                "set is not required."
            )
        elif prompt_mode == "operation":
            task = (
                "# Task\n"
                "Your task is to design a factory with code with the following functionality:\n\n"
                f"{task_body} The input tape carries only red/blue "
                f"colors. For this finite curriculum task, the allowed inputs are exactly: "
                f"{rendered_inputs}. Your factory only needs to be correct for this stated "
                "finite input set."
            )
        else:
            raise RuntimeError(f"unknown finite curriculum prompt mode: {prompt_mode}")
        messages = [dict(message) for message in row["messages"]]
        user_indices = [
            index for index, message in enumerate(messages) if message["role"] == "user"
        ]
        if not user_indices:
            raise RuntimeError("PREPEND curriculum row has no user message")
        message_index = user_indices[-1]
        original = str(messages[message_index]["content"])
        prefix_text = original.split("# Task", 1)[0]
        messages[message_index]["content"] = prefix_text + task
        row["messages"] = messages
        row["name"] = (
            f"Prepend {prefix} prerequisite {operation} on finite inputs "
            f"[{rendered_inputs}]"
        )
        row["difficulty"] = str(spec["name"])
        row["candidate_id"] = f"prepend_sequence__{spec['name']}"
        generator_config = json.loads(str(row["generator_config"]))
        generator_config["finite_curriculum_inputs"] = inputs
        generator_config["finite_curriculum_prompt_explicit"] = True
        generator_config["finite_curriculum_operation"] = operation
        generator_config["finite_curriculum_prompt_mode"] = prompt_mode
        generator_config["finite_curriculum_semantic_label"] = bool(
            spec.get("prepend_curriculum_semantic_label", False)
        )
        row["generator_config"] = json.dumps(
            generator_config, sort_keys=True, default=str
        )
        rewritten.append(row)
    return rewritten


def materialize_stage(index: int, spec: dict[str, Any]) -> Path:
    path = DATA / "oracle" / f"stage_{index:02d}"
    if path.exists():
        return path
    cohort = spec.get("instance_cohort")
    base_path = DATA / "oracle_bases" / str(cohort) if cohort else None
    if base_path is not None and base_path.exists():
        rows = [dict(row) for row in load_from_disk(str(base_path))]
    else:
        generation_version = str(spec.get("generation_version", VERSION))
        split_name = str(
            spec.get(
                "generation_split_name",
                f"oracle_cohort_{cohort}" if cohort else f"oracle_stage_{index:02d}",
            )
        )
        seed_key: Any = spec.get("generation_seed_key", cohort if cohort else index)
        rows = generate_distribution(
            spec["family"],
            spec["generator"],
            256,
            stable_int(generation_version, "oracle", seed_key),
            split_name,
        )
        if base_path is not None:
            Dataset.from_list(rows).save_to_disk(str(base_path))
    rows = apply_prepend_curriculum(rows, spec)
    keep = spec["test_cases"]
    if keep != "all":
        for row in rows:
            row["ground_truth"] = list(row["ground_truth"])[: int(keep)]
    Dataset.from_list(rows).save_to_disk(str(path))
    return path


def mixed_groups(rows: list[dict[str, Any]]) -> int:
    by_instance: dict[int, list[int]] = {}
    for row in rows:
        by_instance.setdefault(int(row["instance_index"]), []).append(int(row["reward"]))
    return sum(len(set(values)) > 1 for values in by_instance.values())


def verifier_case_sets(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    rollouts: int,
    stage_index: int,
    seed: int,
) -> list[list[list[dict[str, Any]]]] | None:
    raw_counts = spec.get("verification_case_counts")
    if raw_counts is None:
        return None
    counts = [int(value) for value in raw_counts]
    if len(counts) != rollouts:
        raise RuntimeError(
            "verification_case_counts must contain exactly one entry per rollout"
        )
    result: list[list[list[dict[str, Any]]]] = []
    for instance_index, row in enumerate(rows):
        cases = list(row["ground_truth"])
        assigned = counts.copy()
        random.Random(
            seed * 1_000_003 + stage_index * 1_009 + instance_index * 257
        ).shuffle(assigned)
        if min(assigned) < 1 or max(assigned) > len(cases):
            raise RuntimeError(
                f"invalid verifier counts {assigned} for {len(cases)} cases"
            )
        instance_sets: list[list[dict[str, Any]]] = []
        occurrences: dict[int, int] = {}
        subset_mode = str(spec.get("verification_subset_mode", "prefix"))
        for count in assigned:
            if subset_mode == "prefix" or count == len(cases):
                indices = tuple(range(count))
            elif subset_mode == "rotating_combinations":
                choices = list(itertools.combinations(range(len(cases)), count))
                occurrence = occurrences.get(count, 0)
                offset = (seed + stage_index * 17 + instance_index) % len(choices)
                indices = choices[(offset + occurrence) % len(choices)]
                occurrences[count] = occurrence + 1
            elif subset_mode == "required_anchor_combinations":
                required = int(spec["verification_required_case_index"])
                if required < 0 or required >= len(cases):
                    raise RuntimeError(
                        f"invalid required verifier case index {required} for "
                        f"{len(cases)} cases"
                    )
                other_indices = [index for index in range(len(cases)) if index != required]
                choices = [
                    tuple(sorted((required, *others)))
                    for others in itertools.combinations(other_indices, count - 1)
                ]
                occurrence = occurrences.get(count, 0)
                offset = (seed + stage_index * 17 + instance_index) % len(choices)
                indices = choices[(offset + occurrence) % len(choices)]
                occurrences[count] = occurrence + 1
            else:
                raise RuntimeError(f"unknown verification_subset_mode: {subset_mode}")
            instance_sets.append([cases[index] for index in indices])
        result.append(instance_sets)
    return result


def build_runtime(cfg: dict[str, Any], stage_paths: list[Path]) -> None:
    solver = cfg["solver"]
    common = {
        "seed": int(cfg["single_start"]["seed"]),
        "goal_updates": int(solver["matched_updates"]),
        "parent_checkpoint": cfg["single_start"]["checkpoint"],
        "parent_scientific_update": int(cfg["single_start"]["scientific_update"]),
        "inherit_optimizer_state": False,
        "prompts_per_update": int(solver["prompts_per_update"]),
        "rollouts_per_prompt": int(solver["rollouts_per_prompt"]),
        "partial_weight": 0.0,
        "evaluation_updates": [int(solver["matched_updates"])],
        "skip_development_evaluation": True,
        "grounded_reward_path": cfg["target"]["selection_split"],
        "grounded_reward_instances": int(cfg["target"]["selection_instances"]),
        "grounded_reward_rollouts_per_instance": int(cfg["target"]["selection_rollouts_per_instance"]),
        "scope_probe_path": cfg["scope"]["split"],
        "scope_probe_instances": int(cfg["scope"]["instances"]),
        "scope_probe_rollouts_per_instance": int(cfg["scope"]["rollouts_per_instance"]),
        "evaluation_rng_key": str(
            cfg.get("evaluation_rng_key", f"{VERSION}_endpoint_aligned")
        ),
        "align_evaluation_across_updates": True,
        "training_rng_key": f"{VERSION}_training_fixed",
        "target_training_path": cfg["target"]["training_split"],
    }
    reused_direct = cfg.get("reused_direct_branch_id")
    direct = {
        **common,
        "branch_id": str(reused_direct or f"{BRANCH_PREFIX}__direct"),
        "kind": "direct",
        "role": (
            "prespecified completed matched target-only null control"
            if reused_direct
            else "matched target-only null control"
        ),
        **(
            {"reused_from_completed_protocol": str(cfg["reused_direct_protocol"])}
            if reused_direct
            else {}
        ),
    }
    through = 0
    stages = []
    for index, (spec, path) in enumerate(zip(cfg["oracle_ladder"], stage_paths)):
        through += int(spec["updates"])
        stages.append(
            {
                "name": spec["name"],
                "kind": "bridge",
                "training_data_path": str(path.relative_to(EXP_ROOT)),
                "through_update": through,
                "target_mix_fraction": float(solver["target_mix_fraction"]),
                "partial_weight": 0.0,
                **(
                    {"gate_seed_branch_id": spec["gate_seed_branch_id"]}
                    if "gate_seed_branch_id" in spec
                    else {}
                ),
                **(
                    {"gate_seed_stage_index": int(spec["gate_seed_stage_index"])}
                    if "gate_seed_stage_index" in spec
                    else {}
                ),
                **(
                    {
                        "verification_case_counts": [
                            int(value) for value in spec["verification_case_counts"]
                        ]
                    }
                    if "verification_case_counts" in spec
                    else {}
                ),
                **(
                    {"verification_subset_mode": spec["verification_subset_mode"]}
                    if "verification_subset_mode" in spec
                    else {}
                ),
                **(
                    {
                        "verification_required_case_index": int(
                            spec["verification_required_case_index"]
                        )
                    }
                    if "verification_required_case_index" in spec
                    else {}
                ),
            }
        )
    if through + int(solver["target_tail_updates"]) != int(solver["matched_updates"]):
        raise RuntimeError("oracle stage and target-tail updates do not match fixed budget")
    stages.append(
        {
            "name": "official_prepend_target_tail",
            "kind": "target",
            "through_update": int(solver["matched_updates"]),
            "partial_weight": 0.0,
        }
    )
    oracle = {
        **common,
        "branch_id": f"{BRANCH_PREFIX}__oracle",
        "candidate_id": f"decisive_{VERSION_TAG}_oracle_prepend",
        "kind": "bridge_chain",
        "role": "manually specified causal PREPEND ladder",
        "proposal": {"components": cfg["oracle_ladder"]},
        "stages": stages,
        "learning_rate": float(solver["learning_rate"]),
        "warmup_updates": int(solver["warmup_updates"]),
        "kl_beta": float(solver["kl_beta_to_starting_solver"]),
        "reference_checkpoint": cfg["single_start"]["checkpoint"],
        "bootstrap_replay_from_branch": "mistral__root__basic__seed42_u20_rebuild",
        "bootstrap_replay_data_path": cfg["scope"]["split"],
        "success_replay_examples_per_update": int(solver["anchor_replay_examples_per_update"]),
        "success_replay_loss_weight": float(solver["anchor_replay_loss_weight"]),
        "success_replay_target_only": False,
        "stage_learnability_gate": cfg["stage_learnability_gate"],
        "online_scope_guard": cfg["online_scope_guard"],
    }
    atomic_json(
        RUNTIME_PATH,
        {
            "protocol_version": VERSION,
            "branch_ids": [direct["branch_id"], oracle["branch_id"]],
            "branches": [direct, oracle],
        },
    )


def submit(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "submission.json"
    if path.exists():
        return read_json(path)
    work_root = EXP_ROOT.parents[1]
    array = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={cfg.get('student_array_spec', '0-1%2')}",
            f"--export=ALL,SELF_GROK_DECISIVE_VERSION={VERSION}",
            str(EXP_ROOT / "scripts" / f"self_grok_decisive_{VERSION_TAG}_students.sbatch"),
        ],
        cwd=str(work_root),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().split(";", 1)[0]
    analysis = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{array}",
            str(EXP_ROOT / "scripts" / f"self_grok_decisive_{VERSION_TAG}_analyze.sbatch"),
        ],
        cwd=str(work_root),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().split(";", 1)[0]
    result = {"student_array_job_id": array, "analysis_job_id": analysis, "submitted_at": time.time()}
    atomic_json(path, result)
    return result


def main() -> None:
    cfg = read_json(CFG_PATH)
    ROOT.mkdir(parents=True, exist_ok=True)
    stage_paths = [materialize_stage(index, spec) for index, spec in enumerate(cfg["oracle_ladder"])]
    build_runtime(cfg, stage_paths)
    selection = [dict(row) for row in load_from_disk(str(EXP_ROOT / cfg["target"]["selection_split"]))]
    scope = [dict(row) for row in load_from_disk(str(EXP_ROOT / cfg["scope"]["split"])).select(range(cfg["scope"]["instances"]))]
    root_checkpoint = EXP_ROOT / cfg["single_start"]["checkpoint"]
    with VLLMServerPool(gpus=(0,), base_port=19430) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora(f"decisive_{VERSION_TAG}_start", root_checkpoint)
        target_rows = client.score_rows(
            selection,
            int(cfg["target"]["selection_rollouts_per_instance"]),
            stable_int(VERSION, "starting_target"),
            ROOT / "starting_target_endpoint.jsonl",
            sampling_seed_key=f"{VERSION}_endpoint_aligned",
        )
        profile = condition_profile(target_rows, selection)
        profile["bottleneck_index"] = bottleneck_index(profile, 0.1)
        scope_rows = client.score_rows(
            scope,
            int(cfg["scope"]["rollouts_per_instance"]),
            stable_int(VERSION, "starting_scope"),
            ROOT / "starting_scope_endpoint.jsonl",
            sampling_seed_key=f"{VERSION}_scope_aligned",
        )
        profile["scope"] = {
            **summarize_rollouts(scope_rows),
            "mean_case_fraction": sum(float(row["passed_cases"]) / max(1, int(row["total_cases"])) for row in scope_rows) / len(scope_rows),
        }
        atomic_json(ROOT / "starting_profile.json", profile)
        probes = []
        if not bool(cfg.get("skip_root_stage_probes", False)):
            gate = cfg["stage_learnability_gate"]
            for index, (spec, path) in enumerate(zip(cfg["oracle_ladder"], stage_paths)):
                dataset = load_from_disk(str(path))
                rows = [dict(dataset[i]) for i in range(int(gate["instances"]))]
                scored = client.score_rows(
                    rows,
                    int(gate["rollouts_per_instance"]),
                    stable_int(VERSION, "root_probe", index),
                    ROOT / "root_stage_probes" / f"stage_{index:02d}.jsonl",
                    verification_case_sets=verifier_case_sets(
                        spec,
                        rows,
                        int(gate["rollouts_per_instance"]),
                        index,
                        int(cfg["single_start"]["seed"]),
                    ),
                    sampling_seed_key=f"{VERSION}_root_probe_{index}",
                )
                summary = summarize_rollouts(scored)
                rate = float(summary["full_pass_rate"])
                groups = mixed_groups(scored)
                trainable = rate >= float(gate["minimum_success_rate"]) and rate <= float(gate["maximum_success_rate"]) and groups >= int(gate["minimum_mixed_groups"])
                mastered = rate >= float(gate["mastered_success_rate"])
                probes.append({"stage_index": index, "stage": spec, **summary, "mixed_groups": groups, "accepted_at_root": trainable or mastered})
        atomic_json(ROOT / "root_stage_probes.json", probes)
    result = submit(cfg)
    print(json.dumps({"starting_profile": profile, "root_stage_probes": probes, "submission": result}, indent=2))


if __name__ == "__main__":
    main()
