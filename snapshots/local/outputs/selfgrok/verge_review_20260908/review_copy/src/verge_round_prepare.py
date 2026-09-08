from __future__ import annotations

import json
import os
import random
import subprocess
import time

from datasets import Dataset

from common import EXP_ROOT, atomic_json, read_json, apply_chat_template, stable_int
from modeling import load_tokenizer
from prepare_data import generate_distribution
from train_branch import build_model
from vllm_runtime import http_json
from verge_round_core import config, ROOT, DATA, VERSION, parse_proposal, generator_config, stage_tokens
from verge_round_runtime import rows, optimizer_for, commit, free_models, endpoint
from verge_round_runtime import VLLMServerPool, VLLMRolloutClient
from verge_target_data import prepare_target

TEACHER = VERSION + "_challenger"


def messages(cfg, initial):
    context = {
        "target": cfg["target_descriptor"],
        "condition_names": initial["target"]["condition_names"],
        "condition_rates": initial["target"]["rates"],
        "lowest_unsaturated_condition_zero_based": initial["target"]["bottleneck_index"],
        "stage_probe_rates": [],
        "archive": [],
        "initialization_note": "A fixed warm-start Solver; the curriculum archive begins empty.",
    }
    return [
        {"role": "system", "content": (
            "You are the Challenger. Propose a curriculum of executable Manufactoria tasks for a Solver. "
            "You receive target checks as feedback; the Solver receives only complete-task binary rewards. "
            "Your return is the final target-condition gain over matched direct target training. "
            "Choose every stage and its order yourself. Return exactly the requested JSON object."
        )},
        {"role": "user", "content": (
            json.dumps(context) + "\nLegal families: " + ", ".join(cfg["families"]) +
            "\nChoose 1 to 4 ordered stages. Each stage has exactly family, colors, length, mutation. "
            "colors is 2 or 4; length is an integer 1..6 controlling sequence length or numeric difficulty; "
            "mutation is none, simple, or pattern for prepend_sequence and must be none for other families. "
            "Numerical families require colors=2; regex_same_num requires colors=4. "
            "prepend_sequence is allowed at easier strata. Stages receive equal shares of 75% of the "
            "training-token budget, each mixing 25% target prompts. The last 25% is target-only. "
            'Schema: {"stages":[{"family":<legal family>,"colors":<2 or 4>,"length":<1..6>,'
            '"mutation":<legal mutation>}]} . No program solutions or prose.'
        )},
    ]


def initialize_teacher(cfg):
    path = EXP_ROOT / "checkpoints" / TEACHER / "resume_u0000"
    if (path / "verge_committed.json").exists():
        return path
    source = EXP_ROOT / cfg["challenger_start_checkpoint"] if cfg.get("book_suite") else None
    tokenizer, model = build_model(42, source)
    optimizer, scheduler = optimizer_for(model, cfg["challenger_learning_rate"])
    prior = 0
    if cfg.get("book_suite"):
        from verge_book_runtime import inherit_optimizer
        prior = inherit_optimizer(optimizer, scheduler, source)
    commit(model, optimizer, scheduler, TEACHER, 0, {"role": "target-conditioned Challenger",
           "teacher_steps_prior": prior, "source": str(source) if source else None})
    del tokenizer, model, optimizer, scheduler
    free_models()
    return path


def submit(branches):
    cfg = config()
    if cfg.get("repair_integrated") and not cfg.get("auto_submit_branches"):
        return {"branch_array": None, "finish_job": None, "note": "Explicit repair runner owns dependencies"}
    path = ROOT / "submission.json"
    prior = read_json(path) if path.exists() else {}
    if prior.get("finish_job"):
        return prior
    work_root = EXP_ROOT.parents[1]
    indices = [str(b["index"]) for b in branches]
    args = ["sbatch", "--parsable", f"--array={','.join(indices)}%2"]
    if os.environ.get("SLURM_JOB_ID"):
        args += [f"--dependency=afterok:{os.environ['SLURM_JOB_ID']}"]
    args += [str(EXP_ROOT / "scripts/verge_round_branches.sbatch")]
    student = prior.get("branch_array")
    if not student:
        student = subprocess.run(args, cwd=work_root, text=True, capture_output=True, check=True).stdout.strip().split(";")[0]
        atomic_json(path, {"branch_array": student, "array_indices": indices})
    final = subprocess.run(
        ["sbatch", "--parsable", f"--dependency=afterok:{student}",
         str(EXP_ROOT / "scripts/verge_round_finish.sbatch")],
        cwd=work_root, text=True, capture_output=True, check=True,
    ).stdout.strip().split(";")[0]
    result = {"branch_array": student, "finish_job": final, "array_indices": indices}
    atomic_json(path, result)
    return result


def main():
    cfg = config()
    if cfg.get("repair_integrated"):
        atomic_json(EXP_ROOT / "manifests" / f"{VERSION}.json", cfg)
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / "round_frozen.json"
    if frozen_path.exists():
        print(json.dumps(submit(read_json(frozen_path)["branches"])))
        return
    if cfg.get("book_suite"):
        from verge_book_runtime import initialize_roots
        initialize_roots(cfg)
    prepare_target()
    selection = rows(cfg["target_selection"])[:cfg.get("endpoint_instances", 64)]
    scope = rows(cfg["scope_dataset"])[:cfg["scope_instances"]]
    teacher = initialize_teacher(cfg)
    branches = [{"index": 0, "id": VERSION + "_direct", "stages": [
        {"path": cfg["target_train"], "tokens": cfg["train_tokens_per_branch"], "kind": "target"}
    ]}]
    port = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000
    with VLLMServerPool(gpus=(0,), base_port=port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora(VERSION + "_start", EXP_ROOT / cfg["solver_start"])
        initial = endpoint(client, ROOT / "initial", checkpoint=cfg["solver_start"],
                           selection=selection, scope=scope)
        generation_path = ROOT / "proposals.json"
        if cfg.get("repair_integrated"):
            from verge_repair_prepare import generate
            generation = generate(cfg, initial, client, teacher)
        elif generation_path.exists():
            generation = read_json(generation_path)
        else:
            client.load_lora(TEACHER, teacher)
            tokenizer = load_tokenizer()
            context = messages(cfg, initial)
            prompt = apply_chat_template(tokenizer, context)
            request = {
                "model": client.model_name, "prompt": [prompt], "n": 3,
                "max_tokens": cfg["proposal_max_tokens"], "temperature": 1.0,
                "top_p": .95, "top_k": 20, "seed": stable_int(VERSION, 42, "proposals"),
                "return_token_ids": True,
            }
            response_path = ROOT / "proposal_response.json"
            if response_path.exists():
                response = read_json(response_path)
            else:
                response = http_json(urls[0] + "/v1/completions", request, timeout=3600)
                atomic_json(response_path, response)
            choices = sorted(response["choices"], key=lambda c: c["index"])
            assert len(choices) == 3
            proposals = []
            for i, choice in enumerate(choices, start=1):
                text = choice["text"]
                token_ids = choice.get("token_ids")
                if not token_ids:
                    token_ids = tokenizer(text, add_special_tokens=False).input_ids
                proposals.append({"index": i, "text": text, "spec": parse_proposal(text, cfg),
                                  "completion_token_ids": token_ids})
            generation = {"messages": context, "candidates": proposals,
                          "usage": response.get("usage"), "one_generation_request": True}
            atomic_json(generation_path, generation)
            del tokenizer
        client.load_lora(VERSION + "_probe_solver", EXP_ROOT / cfg["solver_start"])
        probes = []
        for candidate in generation["candidates"]:
            spec = candidate["spec"]
            candidate["valid"] = spec is not None
            if spec is None:
                continue
            phase_paths = []
            try:
                for j, stage in enumerate(spec["stages"]):
                    phase = DATA / f"candidate_{candidate['index']}" / f"stage_{j}"
                    if not phase.exists():
                        if cfg.get("repair_integrated"):
                            from verge_repair_prepare import stage_rows
                            generated = stage_rows(stage, cfg["stage_train_instances"],
                                stable_int(cfg.get("random_stream_id", VERSION), 42, candidate["index"], j, "train"),
                                f"{VERSION}_c{candidate['index']}_s{j}_train", selection[0])
                        else:
                            generated = generate_distribution(stage["family"], generator_config(stage),
                                cfg["stage_train_instances"], stable_int(VERSION, 42, candidate["index"], j, "train"),
                                f"{VERSION}_c{candidate['index']}_s{j}_train")
                        Dataset.from_list(generated).save_to_disk(str(phase))
                    phase_paths.append(str(phase.relative_to(EXP_ROOT)))
            except (ValueError, KeyError, TypeError, RuntimeError, AttributeError) as exc:
                candidate["valid"] = False
                candidate["validation_error"] = str(exc)
                continue
            quotas = stage_tokens(cfg["train_tokens_per_branch"], len(phase_paths))
            branch = {"index": candidate["index"], "id": f"{VERSION}_candidate_{candidate['index']}",
                      "stages": [{"path": p, "tokens": n, "kind": "curriculum"}
                                 for p, n in zip(phase_paths, quotas[:-1])] + [
                                     {"path": cfg["target_train"], "tokens": quotas[-1], "kind": "target"}]}
            branches.append(branch)
            if cfg.get("repair_integrated") and cfg["stage_probe_instances"] == 0:
                continue
            for j, stage in enumerate(spec["stages"]):
                probe_path = DATA / f"candidate_{candidate['index']}" / f"probe_{j}"
                if not probe_path.exists():
                    generated = generate_distribution(stage["family"], generator_config(stage),
                        cfg["stage_probe_instances"], stable_int(VERSION, 42, candidate["index"], j, "probe"),
                        f"{VERSION}_c{candidate['index']}_s{j}_probe")
                    Dataset.from_list(generated).save_to_disk(str(probe_path))
                probe_rows = rows(str(probe_path.relative_to(EXP_ROOT)))
                scored = client.score_rows(probe_rows, cfg["stage_probe_samples"],
                    stable_int(VERSION, 42, candidate["index"], j, "probe_rollout"),
                    ROOT / "probes" / f"c{candidate['index']}_s{j}.jsonl",
                    sampling_seed_key=f"{VERSION}_probe_{candidate['index']}_{j}")
                probes.append({"candidate": candidate["index"], "stage": j,
                               "successes": sum(r["reward"] for r in scored), "rollouts": len(scored),
                               "purpose": "context for subsequent round; not a rejection gate"})
        client.use_base()
    frozen = {"config": cfg, "initial": initial, "generation": generation, "probes": probes,
              "branches": branches, "reward_condition": initial["target"]["bottleneck_index"],
              "selection_draw": random.Random(stable_int(cfg["random_stream_id"], "selector") if cfg.get("book_suite") else 42).random(), "frozen_at": time.time(),
              "teacher_checkpoint": str(teacher.relative_to(EXP_ROOT))}
    if cfg.get("repair_integrated") and len(branches) != 4:
        atomic_json(ROOT / "interface_failure.json", {"reason": "A typed task could not be rendered", "generation": generation})
        raise RuntimeError("All three legal curricula must render before a repaired round starts")
    atomic_json(frozen_path, frozen)
    print(json.dumps({"submission": submit(branches), "valid_candidates": len(branches) - 1,
                      "starting_condition_rates": initial["target"]["rates"]}, indent=2))


if __name__ == "__main__":
    main()
