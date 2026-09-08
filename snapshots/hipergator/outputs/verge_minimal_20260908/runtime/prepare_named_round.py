"""Prepare an independent names-only round from measured v2 evidence.

--dry-run never constructs a model and can report still-pending history/screen.
Real execution first measures the unchanged initial Solver, then samples Q.
Old-view measurements are diagnostic history, never current incumbents.
This new entry does not modify the v1 preparation or frozen runtime files.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

from hf_backend import ROOT, WORK, MODEL, INITIAL_SOLVER, INITIAL_CHALLENGER, sha256, write_json
from execute_round import build_baseline_plan, execute_base, load_inputs, require
sys.path.insert(0, str(ROOT / "round"))
from round_cli import build_plan, sha
from round_evidence import RawRollout, aggregate_rollouts
from verge_protocol import frontier

VIEW = ROOT / "benchmarks/generated/naming_round_v2"
SCREEN = ROOT / "runs/naming_v2_41415498"
HISTORY = ROOT / "runs/round1_result"
HISTORY_PLAN = ROOT / "runs/round_prepare_41413994/plan.json"
SEED = 20260908
HISTORY_FILES = ("summary.json", "round_score.json", "delta.json", "archive.json", "exchange.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source(path):
    return {"path": str(Path(path).resolve()), "file_sha256": sha256(path)}


def proposal_schema(stages):
    # Same menu and object shape as prepare_round; paths/context are explicit here.
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    item = {"oneOf": [obj({key: {"const": stage[key]} for key in ("stage_id", "kind", "spec")}) for stage in stages]}
    return obj({"base_checkpoint": {"const": str(INITIAL_SOLVER)},
                "curricula": obj({key: {"type": "array", "minItems": 3, "maxItems": 3, "items": item}
                                  for key in ("g1", "g2", "g3")})})


def load_view(view, *, seed=SEED):
    view = Path(view).resolve()
    metadata = read(view / "target_view_metadata.json")
    catalogue = read(view / "catalogue.json")
    validations = read(view / "stage_validation_evidence.json")
    for filename, field in (("target_rows.json", "target_rows_file_sha256"),
                            ("catalogue.json", "catalogue_file_sha256"),
                            ("stage_validation_evidence.json", "stage_validation_evidence_file_sha256")):
        require(sha256(view / filename) == metadata[field], f"frozen v2 {filename} hash mismatch")
    require(metadata["only_messages_changed"] and metadata["task_and_test_contracts_unchanged"],
            "names-only view must preserve the research task and every test")
    require(metadata["prompt_view_sha256"] != metadata["source_prompt_view_sha256"], "new view requires its own identity")
    require(catalogue["prompt_view_sha256"] == metadata["stage_prompt_view_sha256"] ==
            sha(metadata["stage_prompt_view_hash_payload"]), "stage prompt-view hash mismatch")
    baseline = build_baseline_plan(base_checkpoint=str(INITIAL_SOLVER), training_seed=seed, selection=metadata["selection"])
    rows, _ = load_inputs(baseline, view / "target_rows.json")
    stages = catalogue["stages"]
    require(len(stages) == 9 and len({s["stage_id"] for s in stages}) == 9, "retain the frozen nine-stage menu")
    for stage in stages:
        require(stage["kind"] in {"registered", "tape_transform"}, "no hint stage without a separate 512-rollout probe")
        ev = validations["stage_validations"][stage["stage_id"]]
        require(ev["accepted"] is True and ev["spec_sha256"] == sha(stage["spec"]), "accepted exact stage spec required")
        require(ev["checks"] == {"executable": True, "nonconstant": True, "no_leakage": True, "hint_regret": "NA"},
                "preserve exactly four validator checks; nonhint fourth check is NA")
    # Read every catalogue row/spec through the executor using a read-only contract.
    read_contract = {"selection": metadata["selection"], "proposal": {"curricula": {
        "g1": stages[:3], "g2": stages[3:6], "g3": stages[6:]}}}
    load_inputs(read_contract, view / "target_rows.json", view / "catalogue.json")
    return {"view": view, "metadata": metadata, "catalogue": catalogue, "stages": stages,
            "validations": validations["stage_validations"], "baseline": baseline, "rows": rows,
            "sources": {name: source(view / name) for name in
                        ("target_rows.json", "target_view_metadata.json", "catalogue.json", "stage_validation_evidence.json")}}


def load_history(history, history_plan, new_view_sha, *, _fixture=False):
    history, history_plan = Path(history), Path(history_plan)
    values = {name[:-5]: read(history / name) for name in HISTORY_FILES}
    summary, exchange = values["summary"], values["exchange"]
    mode = "unit_fixture" if _fixture else "real_model_round"
    require(summary["status"] == "complete" and summary["mode"] == mode and exchange["mode"] == mode,
            "history requires a completed actual model round")
    require(summary["exchange_sha256"] == sha(exchange), "history exchange changed after scoring")
    plan = read(history_plan)
    require(plan["plan_sha256"] == summary["plan_sha256"] == exchange["plan_sha256"], "history plan identity mismatch")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    require(sha(unsigned) == plan["plan_sha256"], "history plan content changed")
    old_views = {r["prompt_view_sha256"] for evaluation in exchange["evaluations"].values()
                 for r in evaluation.get("records", [])}
    require(old_views == {plan["selection"]["prompt_view_sha256"]} and new_view_sha not in old_views,
            "history must identify its distinct old prompt view")
    require(set(values["delta"]) == set(values["round_score"]["gains"]) == {"g1", "g2", "g3"},
            "all actual historical Delta and Gamma endpoints required")
    require(values["round_score"]["kappa"] == summary["kappa"], "historical reward condition mismatch")
    for branch in ("g1", "g2", "g3"):
        require(values["delta"][branch]["gamma"] == values["round_score"]["gains"][branch]["estimate"],
                "historical Delta and Gamma disagree")
    return {"role": "diagnostic_old_prompt_view_history_only", "eligible_as_current_incumbents": False,
            "round_id": summary["round_id"], "prompt_view_sha256": next(iter(old_views)),
            "summary": summary, "curricula": plan["proposal"]["curricula"],
            "round_score": values["round_score"], "per_stage_delta": values["delta"], "archive": values["archive"],
            "provenance": {**{name: source(history / name) for name in HISTORY_FILES}, "plan.json": source(history_plan)}}


def load_screen(screen, catalogue, *, initial_solver=INITIAL_SOLVER, _fixture=False):
    screen = Path(screen).resolve()
    summary, frozen = read(screen / "SUMMARY.json"), read(screen / "frozen_screen.json")
    mode = "SYNTHETIC_UNIT_FIXTURE" if _fixture else "real_model_naming_only_paired_stage_screen"
    require(summary["status"] == "completed" and summary["mode"] == mode, "completed v2 naming screen required")
    require(summary["source_adapter_unchanged"] is True, "screen changed the original adapter")
    require(frozen["original_adapter_sha256"] == sha256(Path(initial_solver) / "adapter_model.safetensors"),
            "screen did not use the unchanged initial Solver")
    taskfile = Path(frozen["task_file"])
    require(sha256(taskfile) == frozen["task_file_sha256"] and read(taskfile) == frozen["screen"], "screen source tasks changed")
    tasks = frozen["screen"]["tasks"]
    require(len(tasks) == len(summary["summaries"]) == 12 and summary["rollouts"] == 96,
            "all 12 paired v2 screen groups must finish")
    stages = {s["stage_id"]: s for s in catalogue["stages"]}
    library, verified_summaries, raw_sources = [], [], []
    for task, item in zip(tasks, summary["summaries"]):
        require((task["stage_id"], task["variant"]) == (item["stage_id"], item["variant"]), "screen task order changed")
        require(task["row"] in stages[task["stage_id"]]["rows_by_variant"][task["variant"]],
                "screen row does not match this v2 catalogue prompt and tests")
        raw_path = Path(item["raw_path"]).resolve()
        require(raw_path.is_relative_to(screen), "screen raw evidence must stay inside its run")
        raw = read(raw_path)
        require(raw["mode"] == ("SYNTHETIC_UNIT_FIXTURE" if _fixture else "real_model_generation") and
                raw["adapter"] == str(Path(initial_solver).resolve()) and raw["optimizer_steps_before"] == 0,
                "screen samples must come from the untouched original Solver")
        require(raw["instance_id"] == task["row"]["id"] and raw["seed"] == task["seed"] and
                raw["max_new_tokens"] == task["cap"] and raw["n"] == 8 and raw["decoding_policy"] == "dsl_grammar_v1",
                "screen raw sampling contract mismatch")
        require(all(message["content"] in raw["prompt"] for message in task["row"]["messages"]),
                "screen raw prompt is not the new naming view")
        rewards = raw["rewards"]
        require(len(rewards) == len(raw["verification"]) == len(raw["completion_token_ids"]) ==
                len(raw["verifier_completions"]) == 8 and all(type(r) in (int, float) and r in (0, 1) for r in rewards),
                "eight actual binary-reward rollouts required")
        require(sum(map(len, raw["completion_token_ids"])) == raw["generated_tokens"] == item["generated_tokens"],
                "screen native token count mismatch")
        for slot, (reward, verification) in enumerate(zip(rewards, raw["verification"])):
            require(verification["reward"] == reward and len(verification["per_test"]) == len(task["row"]["ground_truth"]),
                    "screen reward lacks its complete verifier suite")
            require([(t["test_id"], t["input"]) for t in verification["per_test"]] ==
                    [(t.get("test_id", str(index)), t["input"]) for index, t in enumerate(task["row"]["ground_truth"])],
                    "screen verifier test identity mismatch")
            require(bool(reward) == (verification["parse_valid"] and all(c["pass"] for c in verification["per_test"])),
                    "screen binary reward disagrees with full exact verification")
            if reward == 1:
                library.append({"stage_id": item["stage_id"], "variant": item["variant"],
                    "evidence_path": str(raw_path), "evidence_file_sha256": sha256(raw_path), "rollout_slot": slot,
                    "program_sha256": sha(raw["verifier_completions"][slot]),
                    "stage_suite_sha256": sha(task["row"]["ground_truth"]),
                    "verified_on": "complete frozen train-side stage suite", "hint_regret_validated": False})
        mixed = len(set(rewards)) == 2
        require(item["binary_successes"] == raw["binary_successes"] == sum(rewards) and
                item["mixed_group"] == raw["mixed_group"] == mixed and
                item["parse_valid"] == raw["parse_valid"] == sum(v["parse_valid"] for v in raw["verification"]),
                "screen summary disagrees with actual raw observations")
        verified_summaries.append(item)
        raw_sources.append(source(raw_path))
    require(summary["mixed_groups"] == sum(s["mixed_group"] for s in verified_summaries) and
            summary["binary_successes"] == sum(s["binary_successes"] for s in verified_summaries),
            "screen aggregate differs from actual groups")
    return {"screen_path": str(screen), "summaries": verified_summaries, "library": library,
            "run_preparation_diagnostic": {"mixed_groups": summary["mixed_groups"],
                "actual_optimizer_steps": summary["actual_optimizer_steps"],
                "is_stage_validator_gate": False, "positive_gamma_required": False,
                "purpose": "bounded development resource decision only; launcher decides whether to run"},
            "provenance": {"summary": source(screen / "SUMMARY.json"), "frozen": source(screen / "frozen_screen.json"),
                           "task_file": source(taskfile), "raw_evidence": raw_sources}}


def collect_inputs(view, screen, history, history_plan, *, seed=SEED, allow_pending=False):
    loaded = load_view(view, seed=seed)
    dependencies, pending = {}, []
    for key, operation in (("history", lambda: load_history(history, history_plan, loaded["rows"]["prompt_view_sha256"])),
                           ("screen", lambda: load_screen(screen, loaded["catalogue"]))):
        try:
            dependencies[key] = operation()
        except FileNotFoundError as error:
            if not allow_pending:
                raise
            pending.append({"dependency": key, "missing_path": str(error.filename)})
    return loaded, dependencies, pending


def measured_context(fragment, loaded, dependencies):
    baseline = loaded["baseline"]
    records = [{k: r[k] for k in RawRollout.__dataclass_fields__} for r in fragment["evaluations"]["base"]["records"]]
    require(all(r["prompt_view_sha256"] == loaded["rows"]["prompt_view_sha256"] and r["hint"] is None
                for r in fragment["evaluations"]["base"]["records"]), "new incumbent must use the new unhinted view")
    require(fragment["baseline_contract_sha256"] == baseline["baseline_contract_sha256"], "new base contract mismatch")
    batch = aggregate_rollouts(records, checkpoint=str(INITIAL_SOLVER), training_seed=baseline["training_seed"],
        manifest_id=baseline["selection"]["manifest_id"], instance_ids=baseline["selection"]["instance_ids"],
        sample_ids=baseline["selection"]["sample_ids"])
    evaluation = batch.evaluation
    history, screen = dependencies["history"], dependencies["screen"]
    return {"target": loaded["rows"]["train"][0]["messages"][0]["content"],
        "current_prompt_view_sha256": loaded["rows"]["prompt_view_sha256"],
        "conditions": ["parse", "min-per-class exact fraction >= 1/36", ">= 0.25", ">= 0.5", ">= 0.75", "full pass"],
        "incumbents": [{"base_checkpoint": str(INITIAL_SOLVER), "b": frontier(evaluation), "s": batch.candidate.structure,
                        "rho": [evaluation.rate(j) for j in range(1, 7)], "full_pass_count": evaluation.success_count,
                        "samples": 256, "prompt_view_sha256": loaded["rows"]["prompt_view_sha256"]}],
        "archive": [], "per_stage_delta_history": [{"round_id": history["round_id"],
            "prompt_view_sha256": history["prompt_view_sha256"], "role": history["role"],
            "values": history["per_stage_delta"], "source": history["provenance"]["delta.json"]}],
        "old_prompt_view_history": history,
        "history_use": "Old rho/archive/Gamma/Delta describe a different prompt view. Use only as diagnostic experience; never substitute them for the newly measured incumbent.",
        "per_stage_delta_definition": "fixed-round-kappa difference of curriculum/direct gaps at matched token boundaries; include target-only tail and P32-minus-P8 correction",
        "verified_program_library_index": screen["library"], "stage_screen": screen["summaries"],
        "stage_screen_provenance": screen["provenance"], "run_preparation_diagnostic": screen["run_preparation_diagnostic"],
        "stage_catalogue": [{k: s[k] for k in ("stage_id", "kind", "spec")} for s in loaded["stages"]],
        "pilot_search_restriction": "Nine frozen executable tasks; this development round does not establish open transform synthesis or the formal 64-instance predictions.",
        "stage_sources": {"registered": "available", "tape_transform": "bounded catalogue available; open expression interface retained",
                          "hinted_target": "retained but inadmissible here: v2 successful library entries have no 512-rollout hint-regret validation"},
        "training": {"G": 3, "L": 3, "B": 65536, "alpha": .25, "eta": .25, "n_min": 8,
                     "binary_reward_only": True, "rollouts_per_group": 8, "max_new_tokens": 2048}}


def sample_challenger(context, stages, output, *, seed=SEED):
    from hf_backend import Solver, trim_completion
    sys.path.insert(0, str(ROOT / "decoding"))
    from hf_grammar_bridge import GrammarService, HFGrammarLogitsProcessor, native_sampling_kwargs
    solver = Solver(adapter=INITIAL_CHALLENGER)
    try:
        schema_path = output / "challenger_schema.json"
        write_json(schema_path, proposal_schema(stages))
        instruction = ("You are VERGE's Challenger. Choose the available incumbent base and propose three ordered curricula, "
            "each with exactly three executable stages, to improve the Solver on the fixed target. Use the newly measured "
            "current-view base conditions and v2 train-side stage screen. The separately labelled old-view history is "
            "diagnostic experience only; its rho, archive and checkpoints are not current incumbent measurements. "
            "The Solver updates only on groups containing both binary successes and failures; all-zero and all-one "
            "groups are skipped exactly. Use the measured mixed-group evidence when choosing and ordering stages. "
            "All branches start from the same untouched initial Solver; no hint stage is admissible before its required probe. "
            "Return only JSON with base_checkpoint and curricula {g1:[...],g2:[...],g3:[...]}. Every stage contains stage_id, "
            "kind and its complete exact spec from the frozen catalogue. The schema fixes syntax and the bounded menu, "
            "not your curriculum choices.\n\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":")))
        text = (solver.tokenizer.bos_token or "") + "[INST]" + instruction + "[/INST]"
        inputs = solver.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(solver.device)
        write_json(output / "challenger_context.json", {"context": context, "actual_prompt": text,
            "prompt_token_ids": inputs["input_ids"][0].tolist(), "schema_file_sha256": sha256(schema_path)})
        with GrammarService(WORK / "envs/vllm/bin/python", MODEL, json_schema_path=schema_path, prefix="",
                            log_name=f"q_named_{__import__('os').getpid()}.stderr.log") as grammar:
            for attempt in range(2):
                solver.torch.manual_seed(seed + 9000 + attempt)
                started = time.time()
                solver.model.eval()
                with solver.torch.no_grad():
                    generated = solver.model.generate(**inputs, **native_sampling_kwargs(),
                        logits_processor=[HFGrammarLogitsProcessor(grammar)], max_new_tokens=8192,
                        eos_token_id=sorted(solver.eos_ids), pad_token_id=solver.tokenizer.pad_token_id, use_cache=True)
                ids = trim_completion(generated[0, inputs["input_ids"].shape[1]:].tolist(), solver.eos_ids)
                raw = solver.tokenizer.decode(ids, skip_special_tokens=True)
                path = output / f"challenger_sample_{attempt}.json"
                write_json(path, {"mode": "real_independent_challenger_generation", "checkpoint": str(INITIAL_CHALLENGER),
                    "checkpoint_sha256": sha256(INITIAL_CHALLENGER / "adapter_model.safetensors"),
                    "seed": seed + 9000 + attempt, "completion_token_ids": ids, "raw_output": raw,
                    "generated_tokens": len(ids), "max_new_tokens": 8192, "seconds": time.time() - started,
                    "grammar_metadata": grammar.metadata, "solver_training_used": False,
                    "context_path": str(output / "challenger_context.json")})
                try:
                    proposal = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ids and ids[-1] in solver.eos_ids:
                    return proposal, {"checkpoint": str(INITIAL_CHALLENGER), "raw_output": raw,
                                      "context": context, "evidence_id": str(path)}
        raise RuntimeError("Two real Challenger attempts did not finish JSON; raw evidence preserved")
    finally:
        solver.close()


def main(output, *, view=VIEW, screen=SCREEN, history=HISTORY, history_plan=HISTORY_PLAN,
         seed=SEED, dry_run=False):
    output = Path(output).resolve()
    if not dry_run:
        require(output.is_relative_to(WORK.resolve()), "real preparation output must remain inside selfgrok")
    output.mkdir(parents=True, exist_ok=False)
    try:
        loaded, dependencies, pending = collect_inputs(view, screen, history, history_plan,
                                                       seed=seed, allow_pending=dry_run)
        report = {"mode": "cpu_input_dry_run" if dry_run else "real_named_round_preparation",
            "status": "pending_external_evidence" if pending else "inputs_verified", "pending_dependencies": pending,
            "source_files": loaded["sources"], "screen_path": str(Path(screen).resolve()),
            "history_path": str(Path(history).resolve()), "history_plan": str(Path(history_plan).resolve()),
            "verified_dependency_sources": {key: value["provenance"] for key, value in dependencies.items()},
            "baseline_contract_sha256": loaded["baseline"]["baseline_contract_sha256"],
            "current_prompt_view_sha256": loaded["rows"]["prompt_view_sha256"],
            "stage_validation_checks": ["executable", "nonconstant", "no_leakage", "hint_regret"],
            "models_loaded": False, "gpu_work_submitted": False}
        write_json(output / "INPUTS.json", report)
        for key, value in dependencies.items():
            write_json(output / f"verified_{key}.json", value)
        write_json(output / "baseline_plan.json", loaded["baseline"])
        write_json(output / "challenger_schema.json", proposal_schema(loaded["stages"]))
        if dry_run:
            print(json.dumps(report), flush=True)
            return report
        fragment = execute_base(loaded["baseline"], loaded["rows"], output / "base")
        print(json.dumps({"new_view_base_complete": str(output / "base/fragment.json")}), flush=True)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        context = measured_context(fragment, loaded, dependencies)
        write_json(output / "measured_context.json", context)
        proposal, challenger = sample_challenger(context, loaded["stages"], output, seed=seed)
        request = {"round_id": "verge_minimal_naming_round2_20260908", "training_seed": seed,
            "base_checkpoint": str(INITIAL_SOLVER), "selection": loaded["metadata"]["selection"],
            "proposal": proposal, "challenger": challenger, "stage_validations": loaded["validations"],
            "B": 65536, "alpha": .25, "eta": .25, "q": 4, "n_min": 8, "bootstrap_resamples": 2000,
            "stage_source_status": context["stage_sources"]}
        write_json(output / "request.json", request)
        plan = build_plan(request)
        load_inputs(plan, loaded["view"] / "target_rows.json", loaded["view"] / "catalogue.json")
        write_json(output / "plan.json", plan)
        ready = {"status": "real_base_and_challenger_complete", "plan": str(output / "plan.json"),
            "base_fragment": str(output / "base/fragment.json"), "plan_sha256": plan["plan_sha256"],
            "baseline_contract_sha256": loaded["baseline"]["baseline_contract_sha256"],
            "target_rows": str(loaded["view"] / "target_rows.json"),
            "target_rows_sha256": loaded["sources"]["target_rows.json"]["file_sha256"],
            "catalogue": str(loaded["view"] / "catalogue.json"),
            "catalogue_sha256": loaded["sources"]["catalogue.json"]["file_sha256"],
            "current_prompt_view_sha256": loaded["rows"]["prompt_view_sha256"],
            "source_files": loaded["sources"], "entry_source": source(Path(__file__)),
            "source_adapter": source(INITIAL_SOLVER / "adapter_model.safetensors"),
            "source_challenger": source(INITIAL_CHALLENGER / "adapter_model.safetensors"),
            "screen_path": dependencies["screen"]["screen_path"], "screen_sources": dependencies["screen"]["provenance"],
            "history_sources": dependencies["history"]["provenance"],
            "history_prompt_view_sha256": dependencies["history"]["prompt_view_sha256"],
            "curricula": {key: [s["stage_id"] for s in value] for key, value in proposal["curricula"].items()}}
        write_json(output / "READY.json", ready)
        print(json.dumps(ready), flush=True)
        return ready
    except Exception as error:
        write_json(output / "FAILED.json", {"status": "failed", "exception": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", type=Path, default=VIEW)
    parser.add_argument("--screen", type=Path, default=SCREEN)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--history-plan", type=Path, default=HISTORY_PLAN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    main(**vars(parser.parse_args()))
