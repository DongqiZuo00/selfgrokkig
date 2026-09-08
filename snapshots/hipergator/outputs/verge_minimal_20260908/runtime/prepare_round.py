"""Measure the untouched base, then sample a real independent Challenger proposal."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time

from hf_backend import (ROOT, WORK, MODEL, INITIAL_SOLVER, INITIAL_CHALLENGER,
                        Solver, sha256, write_json, trim_completion)
from execute_round import build_baseline_plan, execute_base, load_inputs
sys.path[:0] = [str(ROOT / "round"), str(ROOT / "decoding")]
from round_cli import build_plan
from round_evidence import aggregate_rollouts, RawRollout
from verge_protocol import frontier
from hf_grammar_bridge import GrammarService, HFGrammarLogitsProcessor, native_sampling_kwargs

VIEW = ROOT / "benchmarks/generated/pilot_view_v1"
CATALOGUE = ROOT / "benchmarks/generated/catalogue.json"
SCREEN = ROOT / "runs/stage_grammar_v1_41412533"
SEED = 20260908


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def proposal_schema(stages):
    item = {"oneOf": [object_schema({key: {"const": stage[key]} for key in ("stage_id", "kind", "spec")})
                      for stage in stages]}
    return object_schema({"base_checkpoint": {"const": str(INITIAL_SOLVER)},
            "curricula": object_schema({key: {"type": "array", "minItems": 3, "maxItems": 3, "items": item}
                                        for key in ("g1", "g2", "g3")})})


def base_context(fragment, baseline, target_rows, stages):
    records = [{k: r[k] for k in RawRollout.__dataclass_fields__} for r in fragment["evaluations"]["base"]["records"]]
    batch = aggregate_rollouts(records, checkpoint=str(INITIAL_SOLVER), training_seed=SEED,
                   manifest_id=baseline["selection"]["manifest_id"],
                   instance_ids=baseline["selection"]["instance_ids"], sample_ids=baseline["selection"]["sample_ids"])
    ev = batch.evaluation
    screen = read(SCREEN / "SUMMARY.json")
    # The index comes only from real sampled stage successes, never CPU oracles.
    library = []
    for summary in screen["summaries"]:
        raw = read(summary["raw_path"])
        for slot, reward in enumerate(raw["rewards"]):
            if reward == 1:
                library.append({"stage_id": summary["stage_id"], "variant": summary["variant"],
                                "evidence_path": summary["raw_path"], "rollout_slot": slot,
                                "verified_on": "complete frozen train-side stage suite",
                                "hint_regret_validated": False})
    return {"target": target_rows["train"][0]["messages"][0]["content"],
        "conditions": ["parse", "min-per-class exact fraction >= 1/36", ">= 0.25", ">= 0.5", ">= 0.75", "full pass"],
        "incumbents": [{"base_checkpoint": str(INITIAL_SOLVER), "b": frontier(ev),
                        "s": batch.candidate.structure, "rho": [ev.rate(j) for j in range(1, 7)],
                        "full_pass_count": ev.success_count, "samples": 256}],
        "archive": [], "per_stage_delta_history": [],
        "per_stage_delta_definition": "fixed-round-kappa difference of curriculum/direct gaps at matched token boundaries; include target-only tail and P32-minus-P8 correction",
        "verified_program_library_index": library,
        "stage_screen": screen["summaries"],
        "stage_catalogue": [{k: stage[k] for k in ("stage_id", "kind", "spec")} for stage in stages],
        "pilot_search_restriction": "This development round selects from nine frozen executable task specifications. It does not establish the value of open transform synthesis.",
        "stage_sources": {"registered": "available", "tape_transform": "bounded catalogue available; open expression interface retained",
                          "hinted_target": "retained, but not admissible this round because no 512-rollout hint-regret probe is complete"},
        "training": {"G": 3, "L": 3, "B": 65536, "alpha": .25, "eta": .25, "binary_reward_only": True}}


def sample_challenger(context, stages, output):
    solver = Solver(adapter=INITIAL_CHALLENGER)
    torch = solver.torch
    schema_path = output / "challenger_schema.json"
    write_json(schema_path, proposal_schema(stages))
    instruction = ("You are VERGE's Challenger. Choose the available incumbent base and propose three ordered curricula, "
        "each with exactly three executable stages, to improve the Solver on the fixed target. Read the measured "
        "base conditions and train-side stage screen below. Choose stage order using this evidence. "
        "All branches start from the same untouched Solver base; use no hint stage before its required probe. "
        "Return only a JSON object with base_checkpoint and curricula {g1:[...],g2:[...],g3:[...]}. "
        "Every stage object must contain stage_id, kind, and its complete exact spec from the frozen catalogue. "
        "The schema restricts syntax and the bounded pilot catalogue, not which curriculum you select.\n\n" +
        json.dumps(context, ensure_ascii=False, separators=(",", ":")))
    text = (solver.tokenizer.bos_token or "") + "[INST]" + instruction + "[/INST]"
    inputs = solver.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(solver.device)
    write_json(output / "challenger_context.json", {"context": context, "actual_prompt": text,
               "prompt_token_ids": inputs["input_ids"][0].tolist(), "schema_file_sha256": sha256(schema_path)})
    with GrammarService(WORK / "envs/vllm/bin/python", MODEL, json_schema_path=schema_path,
                        prefix="", log_name=f"q_{__import__('os').getpid()}.stderr.log") as grammar:
        for attempt in range(2):
            torch.manual_seed(SEED + 9000 + attempt)
            started = time.time()
            solver.model.eval()
            with torch.no_grad():
                generated = solver.model.generate(**inputs, **native_sampling_kwargs(),
                    logits_processor=[HFGrammarLogitsProcessor(grammar)], max_new_tokens=8192,
                    eos_token_id=sorted(solver.eos_ids), pad_token_id=solver.tokenizer.pad_token_id, use_cache=True)
            ids = trim_completion(generated[0, inputs["input_ids"].shape[1]:].tolist(), solver.eos_ids)
            raw_text = solver.tokenizer.decode(ids, skip_special_tokens=True)
            evidence = {"mode": "real_independent_challenger_generation", "checkpoint": str(INITIAL_CHALLENGER),
                "checkpoint_sha256": sha256(INITIAL_CHALLENGER / "adapter_model.safetensors"),
                "seed": SEED + 9000 + attempt, "completion_token_ids": ids, "raw_output": raw_text,
                "generated_tokens": len(ids), "max_new_tokens": 8192, "seconds": time.time() - started,
                "grammar_metadata": grammar.metadata, "solver_training_used": False,
                "context_path": str(output / "challenger_context.json")}
            path = output / f"challenger_sample_{attempt}.json"
            write_json(path, evidence)
            try:
                proposal = json.loads(raw_text)
            except json.JSONDecodeError:
                continue
            if not ids or ids[-1] not in solver.eos_ids:
                continue
            return proposal, {"checkpoint": str(INITIAL_CHALLENGER), "raw_output": raw_text,
                              "context": context, "evidence_id": str(path)}
    raise RuntimeError("Two bounded real Challenger attempts did not finish JSON; retain raw records for diagnosis")


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = read(VIEW / "target_view_metadata.json")
    stages = read(CATALOGUE)["stages"]
    baseline = build_baseline_plan(base_checkpoint=str(INITIAL_SOLVER), training_seed=SEED, selection=metadata["selection"])
    write_json(output / "baseline_plan.json", baseline)
    rows, _ = load_inputs(baseline, VIEW / "target_rows.json")
    fragment = execute_base(baseline, rows, output / "base")
    print(json.dumps({"base_evaluation_complete": str(output / "base/fragment.json")}), flush=True)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    context = base_context(fragment, baseline, rows, stages)
    write_json(output / "measured_context.json", context)
    proposal, challenger = sample_challenger(context, stages, output)
    request = {"round_id": "verge_minimal_round1_20260908", "training_seed": SEED,
        "base_checkpoint": str(INITIAL_SOLVER), "selection": metadata["selection"],
        "proposal": proposal, "challenger": challenger,
        "stage_validations": read(VIEW / "stage_validation_evidence.json")["stage_validations"],
        "B": 65536, "alpha": .25, "eta": .25, "q": 4, "n_min": 8, "bootstrap_resamples": 2000,
        "stage_source_status": context["stage_sources"]}
    write_json(output / "request.json", request)
    plan = build_plan(request)
    write_json(output / "plan.json", plan)
    write_json(output / "READY.json", {"status": "real_base_and_challenger_complete", "plan": str(output / "plan.json"),
        "base_fragment": str(output / "base/fragment.json"), "plan_sha256": plan["plan_sha256"],
        "curricula": {key: [stage["stage_id"] for stage in value] for key, value in proposal["curricula"].items()}})
    print(json.dumps(read(output / "READY.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
