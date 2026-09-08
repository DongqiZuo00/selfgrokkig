from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from common import EXP_ROOT, apply_chat_template, atomic_json, read_json, stable_int
from generation import summarize_rollouts
from prepare_data import generate_distribution
from prepend_conditions import bottleneck_index, condition_profile
from vllm_runtime import VLLMRolloutClient, VLLMServerPool, http_json
from modeling import load_tokenizer


VERSION = os.environ.get(
    "SELF_GROK_DECISIVE_VERSION", "self_grok_decisive_mistral_v4_1"
)
if not re.fullmatch(r"self_grok_decisive_mistral_v[0-9_]+", VERSION):
    raise RuntimeError(f"invalid decisive protocol version: {VERSION}")
TAG = VERSION.removeprefix("self_grok_decisive_mistral_")
PROTOCOL_PATH = EXP_ROOT / "manifests" / f"{VERSION}.json"
BRANCHES_PATH = EXP_ROOT / "manifests" / f"{VERSION}_runtime_branches.json"
DEC_ROOT = EXP_ROOT / "raw_results" / VERSION
DATA_ROOT = EXP_ROOT / "data" / VERSION


def _selection_rows(cfg: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    path = DATA_ROOT / "selection"
    target_path = EXP_ROOT / cfg["target"]["training_split"]
    if "frozen_source" in cfg["target"] and (not path.exists() or not target_path.exists()):
        source = load_from_disk(str(EXP_ROOT / cfg["target"]["frozen_source"]))
        indices = list(range(len(source)))
        random.Random(int(cfg["target"]["split_seed"])).shuffle(indices)
        train_count = int(cfg["target"]["training_instances"])
        train = source.select(indices[:train_count])
        reference_path = cfg["target"].get("selection_reference")
        if not reference_path:
            raise RuntimeError("frozen target recovery requires an immutable selection reference")
        reference = load_from_disk(str(EXP_ROOT / reference_path))
        selection_count = int(cfg["target"]["selection_instances"])
        if len(reference) < selection_count:
            raise RuntimeError("frozen selection reference is too small")
        selection = reference.select(range(selection_count))
        train_ids = {str(train[index]["id"]) for index in range(len(train))}
        selection_ids = {str(selection[index]["id"]) for index in range(len(selection))}
        if train_ids & selection_ids:
            raise RuntimeError("recovered PREPEND train and frozen selection overlap")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        train.save_to_disk(str(target_path))
        Dataset.from_list(
            [dict(selection[index]) for index in range(len(selection))]
        ).save_to_disk(str(path))
    elif not path.exists():
        source = load_from_disk(str(EXP_ROOT / cfg["target"]["selection_source"]))
        count = int(cfg["target"]["selection_instances"])
        Dataset.from_list([dict(source[index]) for index in range(count)]).save_to_disk(str(path))
    return path, [dict(row) for row in load_from_disk(str(path))]


def _scope_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(EXP_ROOT / cfg["scope_guard"]["split"]))
    return [dict(dataset[index]) for index in range(int(cfg["scope_guard"]["instances"]))]


def _parse_spec(text: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    # The executable curriculum must be exactly what the Challenger emitted.
    # Do not scan for a convenient first object or silently discard prose,
    # prerequisites, alternative orders, or unknown fields.
    raw = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"components"}:
        return None
    components = parsed["components"]
    if not isinstance(components, list) or not 1 <= len(components) <= int(
        cfg["challenger"]["maximum_curriculum_stages"]
    ):
        return None
    allowed = set(cfg["challenger"]["allowed_families"])
    levels = set(cfg["challenger"]["levels"])
    normalized = []
    for component in components:
        required = {"family", "level", "weight"}
        if not isinstance(component, dict) or set(component) != required:
            return None
        family = str(component["family"])
        level = str(component["level"])
        if not isinstance(component["weight"], int) or isinstance(component["weight"], bool):
            return None
        weight = component["weight"]
        if family not in allowed or level not in levels or not 1 <= weight <= 4:
            return None
        normalized.append({"family": family, "level": level, "weight": weight})
    return {"components": normalized}


def _challenger_messages(cfg: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, str]]:
    families = ", ".join(cfg["challenger"]["allowed_families"])
    return [
        {
            "role": "system",
            "content": (
                "You are the Challenger in a one-round curriculum discovery experiment. "
                "Design an ordered bridge for a fixed Solver. Every stage is trained with its own strict "
                "binary verifier. Return only the requested JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Target: Manufactoria PREPEND_SEQUENCE with mutations. A correct program must parse, "
                "terminate on every test, prepend the requested color sequence, and preserve the exact "
                "registered mutated residual tape. The starting Solver endpoint condition rates are "
                f"{profile['rates']} for [parse, finish-all, prefix-all, residual-all, full-pass], and its "
                f"lowest unsaturated rung is {profile['bottleneck_index']}. Propose one complete ordered "
                "curriculum of one to three stages. Earlier stages should teach prerequisites and later "
                "stages should approach the target contract. Legal families are: "
                f"{families}. Each level is low, medium, or high; weight is an integer 1..4 and allocates "
                "the fixed curriculum-update budget. Output exactly one object such as "
                '{"components":[{"family":"starts_with","level":"low","weight":1},'
                '{"family":"append_sequence","level":"medium","weight":2}]}'
            ),
        },
    ]


def _allocate_updates(components: list[dict[str, Any]], total: int) -> list[int]:
    weights = [int(item["weight"]) for item in components]
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [max(1, int(value)) for value in raw]
    while sum(counts) < total:
        index = max(range(len(counts)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    while sum(counts) > total:
        choices = [i for i, value in enumerate(counts) if value > 1]
        index = min(choices, key=lambda i: raw[i] - counts[i])
        counts[index] -= 1
    return counts


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _materialize_stage(
    cfg: dict[str, Any],
    candidate_index: int,
    stage_index: int,
    component: dict[str, Any],
) -> Path:
    path = DATA_ROOT / f"candidate_{candidate_index:02d}" / f"stage_{stage_index:02d}"
    if path.exists():
        return path
    level = str(component["level"])
    rows = generate_distribution(
        str(component["family"]),
        dict(cfg["challenger"]["levels"][level]),
        256,
        stable_int("decisive_stage", 42, candidate_index, stage_index),
        f"decisive_c{candidate_index:02d}_s{stage_index:02d}",
    )
    Dataset.from_list(rows).save_to_disk(str(path))
    return path


def _submit(cfg: dict[str, Any]) -> dict[str, Any]:
    path = DEC_ROOT / "submission.json"
    if path.exists():
        return read_json(path)
    work_root = EXP_ROOT.parents[1]
    array = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array=0-3%{int(cfg['resources']['student_array_concurrency'])}",
            f"--export=ALL,SELF_GROK_DECISIVE_VERSION={VERSION}",
            str(EXP_ROOT / "scripts" / "self_grok_decisive_students.sbatch"),
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
            f"--export=ALL,SELF_GROK_DECISIVE_VERSION={VERSION}",
            str(EXP_ROOT / "scripts" / "self_grok_decisive_analyze.sbatch"),
        ],
        cwd=str(work_root),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().split(";", 1)[0]
    result = {
        "student_array_job_id": array,
        "analysis_job_id": analysis,
        "submitted_at_unix": time.time(),
    }
    atomic_json(path, result)
    return result


def main() -> None:
    cfg = read_json(PROTOCOL_PATH)
    DEC_ROOT.mkdir(parents=True, exist_ok=True)
    selection_path, selection = _selection_rows(cfg)
    scope = _scope_rows(cfg)
    root_checkpoint = EXP_ROOT / cfg["starting_solver"]["checkpoint"]
    root_profile_path = DEC_ROOT / "starting_solver_profile.json"
    generation_path = DEC_ROOT / "challenger_generation.json"

    with VLLMServerPool(gpus=(0,), base_port=17321) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora("decisive_starting_solver", root_checkpoint)
        target_rollouts = client.score_rows(
            selection,
            int(cfg["target"]["selection_rollouts_per_instance"]),
            stable_int("decisive_starting_target", 42),
            DEC_ROOT / "starting_solver_target_endpoint.jsonl",
            sampling_seed_key="decisive_endpoint_aligned",
        )
        profile = condition_profile(target_rollouts, selection)
        profile["bottleneck_index"] = bottleneck_index(profile, float(cfg["conditions"]["delta"]))
        scope_rollouts = client.score_rows(
            scope,
            int(cfg["scope_guard"]["rollouts_per_instance"]),
            stable_int("decisive_starting_scope", 42),
            DEC_ROOT / "starting_solver_scope_endpoint.jsonl",
            sampling_seed_key="decisive_scope_aligned",
        )
        profile["scope"] = {
            **summarize_rollouts(scope_rollouts),
            "mean_case_fraction": sum(
                float(item["passed_cases"]) / max(1, int(item["total_cases"]))
                for item in scope_rollouts
            )
            / len(scope_rollouts),
        }
        atomic_json(root_profile_path, profile)

        client.use_base()
        messages = _challenger_messages(cfg, profile)
        tokenizer = load_tokenizer("left")
        prompt = apply_chat_template(tokenizer, messages)
        diagnostics_path = DEC_ROOT / "challenger_generation_format_diagnostics.json"
        prior_raw = (
            read_json(diagnostics_path).get("raw_outputs", [])
            if diagnostics_path.exists()
            else []
        )
        if prior_raw:
            # Recover the already-sampled fixed request. Extra explanatory keys
            # are discarded by canonicalization; executable fields remain under
            # the unchanged strict registry/level/weight validator.
            choices = [
                {"index": index, "text": item["completion"]}
                for index, item in enumerate(prior_raw)
            ]
        else:
            response = http_json(
                urls[0] + "/v1/completions",
                {
                    "model": client.model_name,
                    "prompt": [prompt],
                    # One fixed request; oversample only to survive schema
                    # rejection without changing the three-curriculum cohort.
                    "n": 24,
                    "max_tokens": int(cfg["challenger"]["completion_tokens"]),
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": 42,
                },
                timeout=3600,
            )
            choices = response["choices"]
        proposals = []
        seen = set()
        raw = []
        for choice in sorted(choices, key=lambda item: int(item["index"])):
            completion = str(choice["text"])
            spec = _parse_spec(completion, cfg)
            raw.append({"completion": completion, "valid": spec is not None})
            if spec is None:
                continue
            key = json.dumps(spec, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(spec)
            if len(proposals) == int(cfg["challenger"]["proposal_group_size"]):
                break
        atomic_json(
            DEC_ROOT / "challenger_generation_format_diagnostics.json",
            {
                "requested_completions": 24,
                "valid_unique_curricula": len(proposals),
                "raw_outputs": raw,
            },
        )
        if len(proposals) != int(cfg["challenger"]["proposal_group_size"]):
            raise RuntimeError(f"single Challenger request produced only {len(proposals)} valid unique curricula")

        stage_probe_records = []
        client.load_lora("decisive_starting_solver_probe", root_checkpoint)
        probed = set()
        for candidate_index, proposal in enumerate(proposals):
            for stage_index, component in enumerate(proposal["components"]):
                stage_path = _materialize_stage(cfg, candidate_index, stage_index, component)
                key = (str(component["family"]), str(component["level"]))
                if key in probed:
                    continue
                probed.add(key)
                probe_rows = [dict(row) for row in load_from_disk(str(stage_path)).select(range(8))]
                probe_name = _safe_name("__".join(key))
                probe_rollouts = client.score_rows(
                    probe_rows,
                    4,
                    stable_int("decisive_stage_probe", *key),
                    DEC_ROOT / "stage_probes" / f"{probe_name}.jsonl",
                    sampling_seed_key=f"decisive_stage_probe_{probe_name}",
                )
                stage_probe_records.append(
                    {"family": key[0], "level": key[1], **summarize_rollouts(probe_rollouts)}
                )

    atomic_json(
        generation_path,
        {
            "messages": messages,
            "target_visible_to_challenger": True,
            "single_generation_request": True,
            "fixed_rng_seed": 42,
            "proposals": proposals,
            "raw_outputs": raw,
            "stage_probes": stage_probe_records,
        },
    )

    solver_cfg = cfg["solver_branches"]
    goal = int(solver_cfg["matched_updates"])
    common = {
        "seed": 42,
        "goal_updates": goal,
        "parent_checkpoint": cfg["starting_solver"]["checkpoint"],
        "parent_scientific_update": int(
            cfg["starting_solver"].get("scientific_update", 100)
        ),
        "inherit_optimizer_state": False,
        "prompts_per_update": int(solver_cfg["prompts_per_update"]),
        "rollouts_per_prompt": int(solver_cfg["rollouts_per_prompt"]),
        "partial_weight": 0.0,
        "evaluation_updates": [goal],
        "skip_development_evaluation": True,
        "grounded_reward_path": str(selection_path.relative_to(EXP_ROOT)),
        "grounded_reward_instances": int(cfg["target"]["selection_instances"]),
        "grounded_reward_rollouts_per_instance": int(cfg["target"]["selection_rollouts_per_instance"]),
        "scope_probe_path": cfg["scope_guard"]["split"],
        "scope_probe_instances": int(cfg["scope_guard"]["instances"]),
        "scope_probe_rollouts_per_instance": int(cfg["scope_guard"]["rollouts_per_instance"]),
        "evaluation_rng_key": "decisive_endpoint_aligned",
        "training_rng_key": "decisive_training_fixed",
        "target_training_path": cfg["target"]["training_split"],
    }
    direct_id = f"mistral_decisive_{TAG}__direct"
    branches = [
        {
            **common,
            "branch_id": direct_id,
            "kind": "direct",
            "role": "matched target-only null control",
        }
    ]
    curriculum_total = int(solver_cfg["curriculum_updates"])
    for candidate_index, proposal in enumerate(proposals):
        updates = _allocate_updates(proposal["components"], curriculum_total)
        through = 0
        stages = []
        for stage_index, (component, count) in enumerate(zip(proposal["components"], updates)):
            stage_path = _materialize_stage(cfg, candidate_index, stage_index, component)
            through += count
            stages.append(
                {
                    "name": f"{component['family']}__{component['level']}",
                    "kind": "bridge",
                    "training_data_path": str(stage_path.relative_to(EXP_ROOT)),
                    "through_update": through,
                    "target_mix_fraction": float(solver_cfg["target_mixture_fraction_during_curriculum"]),
                    "partial_weight": 0.0,
                }
            )
        stages.append(
            {
                "name": "prepend_target_tail",
                "kind": "target",
                "through_update": goal,
                "partial_weight": 0.0,
            }
        )
        branches.append(
            {
                **common,
                "branch_id": f"mistral_decisive_{TAG}__candidate_{candidate_index:02d}",
                "candidate_id": f"decisive_{TAG}_c{candidate_index:02d}",
                "kind": "bridge_chain",
                "role": "target-conditioned ordered curriculum",
                "stages": stages,
                "proposal": proposal,
            }
        )
    atomic_json(
        BRANCHES_PATH,
        {
            "protocol_version": cfg["protocol_version"],
            "backbone": cfg["backbone"],
            "branch_ids": [branch["branch_id"] for branch in branches],
            "branches": branches,
        },
    )
    submission = _submit(cfg)
    print(json.dumps({"profile": profile, "proposals": proposals, "submission": submission}, indent=2))


if __name__ == "__main__":
    main()
