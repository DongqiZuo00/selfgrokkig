"""Conditional development round with a declared first-stage proposal prior.

Only g1[0] is chosen by the real Challenger from stages with an observed mixed
binary screen group. The other eight slots retain the existing nine-stage menu.
This is a human-set proposal prior, not a fifth stage validator. No job is
submitted here. The completed round2 result is required before production use.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import prepare_named_round as named
from execute_round import checked_plan, validate_base_fragment, baseline_contract, require
from hf_backend import ROOT, WORK, MODEL, INITIAL_SOLVER, INITIAL_CHALLENGER, write_json, sha256
from round_cli import build_plan, sha
from round_evidence import RawRollout, aggregate_rollouts

PREPARATION = ROOT / "runs/round_prepare_named_41417194"
RESULT = ROOT / "runs/round2_result"
POLICY_VERSION = "verge_pilot_g1_first_stage_observed_mixed_prior_v1"


def observed_starts(stages, screen):
    """Screen observations affect one proposal position, never stage validity."""
    by_id = {s["stage_id"]: s for s in stages}
    choices, observations = [], []
    for item in screen["summaries"]:
        # Training currently uses the catalogue's concise rows. An official-only
        # success cannot justify this prior on a different prompt variant.
        if item["variant"] == "concise" and item["mixed_group"]:
            require(0 < item["binary_successes"] < 8, "mixed observation must contain both binary rewards")
            stage = by_id[item["stage_id"]]
            require(stage["rows"] == stage["rows_by_variant"]["concise"], "screen/training stage prompts differ")
            choices.append(stage["stage_id"])
            observations.append({"stage_id": stage["stage_id"], "variant": "concise", "successes": item["binary_successes"],
                                 "n": 8, "source": item["raw_path"]})
    require(set(choices) == {"identity", "append_R"} and len(choices) == 2,
            "this reviewed policy is bound to the two actual v2 mixed stages; changed evidence requires a new policy")
    return choices, observations


def ignition_schema(stages, eligible, *, base_checkpoint=str(INITIAL_SOLVER)):
    schema = named.proposal_schema(stages)
    schema["properties"]["base_checkpoint"]["const"] = base_checkpoint
    curricula = schema["properties"]["curricula"]["properties"]
    all_items = copy.deepcopy(curricula["g1"]["items"])
    starts = {"oneOf": [copy.deepcopy(item) for item in all_items["oneOf"]
                        if item["properties"]["stage_id"]["const"] in eligible]}
    require(len(starts["oneOf"]) == len(set(eligible)) == 2, "exactly two observed starting stages required")
    curricula["g1"] = {"type": "array", "prefixItems": [starts, all_items, copy.deepcopy(all_items)],
                        "items": False, "minItems": 3, "maxItems": 3}
    return schema


def assert_proposal(proposal, stages, eligible, base_checkpoint):
    require(set(proposal) == {"base_checkpoint", "curricula"} and proposal["base_checkpoint"] == base_checkpoint,
            "Challenger must choose the retained initial base")
    require(set(proposal["curricula"]) == {"g1", "g2", "g3"}, "exactly three real curricula required")
    menu = [{k: s[k] for k in ("stage_id", "kind", "spec")} for s in stages]
    for course in proposal["curricula"].values():
        require(len(course) == 3 and all(s in menu for s in course), "every actual stage must equal its frozen menu entry")
    require(proposal["curricula"]["g1"][0]["stage_id"] in eligible, "actual Q output violates the declared first-stage prior")


def completed_history(result, plan, *, _fixture=False):
    result = Path(result)
    data = {name[:-5]: named.read(result / name) for name in named.HISTORY_FILES}
    summary, exchange = data["summary"], data["exchange"]
    expected_mode = "unit_fixture" if _fixture else "real_model_round"
    require(summary["status"] == "complete" and summary["mode"] == exchange["mode"] == expected_mode,
            "complete actual round2 result required; no partial or predicted history")
    require(summary["exchange_sha256"] == sha(exchange) and
            summary["plan_sha256"] == exchange["plan_sha256"] == plan["plan_sha256"], "round2 history identity mismatch")
    require(summary["round_id"] == plan["round_id"], "completed history names a different round")
    views = {r["prompt_view_sha256"] for item in exchange["evaluations"].values() for r in item.get("records", [])}
    require(views == {plan["selection"]["prompt_view_sha256"]}, "round2 history must use the current naming view")
    require(set(data["delta"]) == set(data["round_score"]["gains"]) == {"g1", "g2", "g3"},
            "all completed curriculum Gamma and Delta endpoints required")
    for key in ("g1", "g2", "g3"):
        require(data["delta"][key]["gamma"] == data["round_score"]["gains"][key]["estimate"], "round2 Delta/Gamma disagreement")
    require(data["round_score"]["kappa"] == summary["kappa"], "round2 reward-condition mismatch")
    measured_updates = sum(phase["optimizer_steps"] for branch in ("g1", "g2", "g3")
                           for phase in exchange["branches"][branch]["phases"])
    require(measured_updates == summary["real_curriculum_mixed_binary_optimizer_steps"],
            "history update summary disagrees with its actual branch ledgers")
    # This is a condition on whether to launch this development intervention,
    # not a stage-validation condition and not a positive-Gamma gate.
    require(summary["real_curriculum_mixed_binary_optimizer_steps"] == 0,
            "round2 already has a real curriculum update; this conditional ignition intervention is not needed")
    return {"role": "completed_current_prompt_view_diagnostic_history", "eligible_as_current_incumbents": False,
            "prompt_view_sha256": next(iter(views)), "summary": summary, "curricula": plan["proposal"]["curricula"],
            "round_score": data["round_score"], "per_stage_delta": data["delta"], "archive": data["archive"],
            "provenance": {name: named.source(result / name) for name in named.HISTORY_FILES}}


def reuse_base(preparation, baseline, *, _fixture=False):
    """Return a copied fragment with unchanged source records and zero new work."""
    preparation = Path(preparation).resolve()
    ready = named.read(preparation / "READY.json")
    require(ready["status"] == "real_base_and_challenger_complete", "source preparation is not complete")
    plan = checked_plan(preparation / "plan.json")
    require(ready["plan_sha256"] == plan["plan_sha256"], "source READY and plan differ")
    require(sha(baseline_contract(plan)) == baseline["baseline_contract_sha256"], "cannot reuse a different baseline contract")
    fragment_path = preparation / "base/fragment.json"
    fragment = named.read(fragment_path)
    validate_base_fragment(plan, fragment)
    require(fragment["baseline_contract_sha256"] == ready["baseline_contract_sha256"] and
            fragment["execution"]["no_training"] is True, "source baseline was not an unchanged measurement")
    records = fragment["evaluations"]["base"]["records"]
    selection = baseline["selection"]
    require(len(records) == 256 and {(r["instance_id"], r["sample_id"]) for r in records} ==
            {(i, s) for i in selection["instance_ids"] for s in selection["sample_ids"]}, "complete source P32 required")
    sources = {}
    for record in records:
        require(record["checkpoint"] == plan["base_checkpoint"] and record["hint"] is None and
                record["prompt_view_sha256"] == selection["prompt_view_sha256"], "base source view/checkpoint/hint mismatch")
        rawname, slot_text = record["generation_evidence_id"].rsplit("#sample", 1)
        rawpath, slot = Path(rawname).resolve(), int(slot_text)
        require(rawpath.is_relative_to(preparation / "base/raw_evaluation"), "baseline raw source escaped its actual original run")
        if str(rawpath) not in sources:
            require(sha256(rawpath) == record["raw_generation_sha256"], "original baseline raw file changed")
            sources[str(rawpath)] = {"file_sha256": sha256(rawpath), "raw": named.read(rawpath)}
        item = sources[str(rawpath)]
        require(item["file_sha256"] == record["raw_generation_sha256"] and 0 <= slot < 8, "inconsistent baseline raw origin")
        raw = item["raw"]
        require(raw["mode"] == ("SYNTHETIC_UNIT_FIXTURE" if _fixture else "real_model_generation") and
                raw["decoding_policy"] == "dsl_grammar_v1", "baseline source must be actual grammar generation")
        if not _fixture:
            require(raw["n"] == 8 and raw["adapter"] == plan["base_checkpoint"] and raw["optimizer_steps_before"] == 0 and
                    raw["seed"] == record["chunk_seed"] and raw["instance_id"] == record["instance_id"],
                    "baseline source checkpoint or actual batch seed changed")
        require(raw["completion_token_ids"][slot] == record["completion_token_ids"] and
                raw["verifier_completions"][slot] == record["raw_completion"], "base record differs from actual original sample")
    require(len(sources) == 32, "baseline must preserve its 32 actual eight-sample chunks")
    report = {"evaluation_cache_hit": True, "cache_scope": "unchanged_checkpoint_and_complete_baseline_contract_across_preparations",
        "source_preparation": str(preparation), "source_fragment": named.source(fragment_path),
        "source_plan": named.source(preparation / "plan.json"), "source_ready": named.source(preparation / "READY.json"),
        "baseline_contract_sha256": baseline["baseline_contract_sha256"], "reused_rollout_slots": 256,
        "new_evaluation_rollouts": 0, "new_evaluation_generated_tokens": 0,
        "original_evaluation_generated_tokens": fragment["execution"]["actual_generation_tokens"],
        "raw_sources": [{"path": path, "file_sha256": item["file_sha256"]} for path, item in sources.items()],
        "records_copied_without_any_change": True,
        "scorer_accounting": "Frozen scorer counts referenced baseline evidence; subtract this cross-round reuse only when reporting newly generated work."}
    copied = copy.deepcopy(fragment)
    copied["baseline_reuse"] = report
    copied["original_execution"] = copy.deepcopy(fragment["execution"])
    copied["execution"] = {**fragment["execution"], "actual_generation_tokens": 0,
                            "new_evaluation_rollouts": 0, "reused_rollout_slots": 256, "evaluation_cache_hit": True}
    require(copied["evaluations"] == fragment["evaluations"] and copied["checkpoints"] == fragment["checkpoints"],
            "cross-preparation reuse must not rewrite records or physical checkpoints")
    return plan, copied, report


def build_context(source_plan, fragment, history, screen, eligible, observations):
    context = copy.deepcopy(source_plan["challenger"]["context"])
    records = [{k: r[k] for k in RawRollout.__dataclass_fields__} for r in fragment["evaluations"]["base"]["records"]]
    selection = source_plan["selection"]
    batch = aggregate_rollouts(records, checkpoint=source_plan["base_checkpoint"], training_seed=source_plan["training_seed"],
        manifest_id=selection["manifest_id"], instance_ids=selection["instance_ids"], sample_ids=selection["sample_ids"])
    require(len(context["incumbents"]) == 1 and context["incumbents"][0]["base_checkpoint"] == source_plan["base_checkpoint"] and
            context["incumbents"][0]["rho"] == [batch.evaluation.rate(j) for j in range(1, 7)], "source incumbent differs from reused actual base")
    require(context["current_prompt_view_sha256"] == selection["prompt_view_sha256"], "source context changed prompt view")
    context["completed_current_view_round_history"] = history
    context["per_stage_delta_history"].append({"round_id": history["summary"]["round_id"],
        "prompt_view_sha256": history["prompt_view_sha256"], "role": history["role"],
        "values": history["per_stage_delta"], "source": history["provenance"]["delta.json"]})
    context["stage_screen"] = screen["summaries"]
    context["stage_screen_provenance"] = screen["provenance"]
    context["verified_program_library_index"] = screen["library"]
    context["baseline_reuse"] = fragment["baseline_reuse"]
    context["proposal_policy"] = {"version": POLICY_VERSION, "human_set_development_prior": True,
        "constraint": "Only curricula.g1[0] must be selected by the Challenger from eligible_first_stage_ids; all other eight positions retain the full nine-stage menu.",
        "eligible_first_stage_ids": eligible, "observations": observations,
        "stage_validator_changed": False, "stage_validator_count": 4,
        "unconstrained_proposal_advantage_established": False, "ignition_or_positive_gamma_guaranteed": False,
        "gradient_fact": "Only groups with both binary successes and failures can update the Solver. All-zero and all-one groups are skipped exactly.",
        "scope": "One declared bounded development intervention; research target, all tests/classes, stage-source interfaces and formal prediction scope remain unchanged."}
    return context


def sample_challenger(context, schema, stages, eligible, output, *, seed):
    from hf_backend import Solver, trim_completion
    sys.path.insert(0, str(ROOT / "decoding"))
    from hf_grammar_bridge import GrammarService, HFGrammarLogitsProcessor, native_sampling_kwargs
    solver = Solver(adapter=INITIAL_CHALLENGER)
    try:
        schema_path = output / "challenger_schema.json"
        write_json(schema_path, schema)
        instruction = ("You are VERGE's Challenger. Generate three ordered curricula of three executable stages for the fixed target. "
            "This development run explicitly applies a human-set first-stage proposal prior: choose g1's first stage yourself "
            "from the observed mixed-reward eligible_first_stage_ids. Every other position remains your choice from all nine stages. "
            "This prior does not reject stages or change the four validators. The Solver learns only from mixed binary groups; "
            "same-reward groups skip updates. A prior observed mixed group does not guarantee another update or transfer. "
            "Use the completed measured history. The current initial-base rho is reused from the identical prompt-view/checkpoint "
            "and sampling contract, with zero new baseline samples. Do not treat old-view history as current measurements. "
            "Return only JSON with base_checkpoint and curricula g1/g2/g3, each containing three exact stage_id/kind/spec objects.\n\n" +
            json.dumps(context, ensure_ascii=False, separators=(",", ":")))
        text = (solver.tokenizer.bos_token or "") + "[INST]" + instruction + "[/INST]"
        inputs = solver.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(solver.device)
        write_json(output / "challenger_context.json", {"context": context, "actual_prompt": text,
            "prompt_token_ids": inputs["input_ids"][0].tolist(), "schema_file_sha256": sha256(schema_path)})
        with GrammarService(WORK / "envs/vllm/bin/python", MODEL, json_schema_path=schema_path, prefix="",
                            log_name=f"q_ignition_{__import__('os').getpid()}.stderr.log") as grammar:
            for attempt in range(2):
                actual_seed = seed + 9000 + attempt
                solver.torch.manual_seed(actual_seed)
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
                    "checkpoint_sha256": sha256(INITIAL_CHALLENGER / "adapter_model.safetensors"), "seed": actual_seed,
                    "completion_token_ids": ids, "raw_output": raw, "generated_tokens": len(ids), "max_new_tokens": 8192,
                    "seconds": time.time() - started, "grammar_metadata": grammar.metadata, "solver_training_used": False,
                    "context_path": str(output / "challenger_context.json"), "proposal_policy": POLICY_VERSION})
                try:
                    proposal = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if ids and ids[-1] in solver.eos_ids:
                    assert_proposal(proposal, stages, eligible, context["incumbents"][0]["base_checkpoint"])
                    return proposal, {"checkpoint": str(INITIAL_CHALLENGER), "raw_output": raw,
                                      "context": context, "evidence_id": str(path)}
        raise RuntimeError("Two real Challenger attempts did not complete; retain all raw evidence")
    finally:
        solver.close()


def main(output, *, preparation=PREPARATION, result=RESULT, view=named.VIEW, screen=named.SCREEN, dry_run=False):
    output = Path(output).resolve()
    if not dry_run:
        require(output.is_relative_to(WORK.resolve()), "real output must remain inside selfgrok")
    output.mkdir(parents=True, exist_ok=False)
    try:
        loaded = named.load_view(view)
        pending, dependencies = [], {}
        for key, operation in (("baseline", lambda: reuse_base(preparation, loaded["baseline"])),
                               ("screen", lambda: named.load_screen(screen, loaded["catalogue"]))):
            try:
                dependencies[key] = operation()
            except FileNotFoundError as error:
                if not dry_run:
                    raise
                pending.append({"dependency": key, "missing_path": str(error.filename)})
        history = None
        try:
            plan = dependencies["baseline"][0] if "baseline" in dependencies else checked_plan(Path(preparation) / "plan.json")
            history = completed_history(result, plan)
        except FileNotFoundError as error:
            if not dry_run:
                raise
            pending.append({"dependency": "completed_round2_result", "missing_path": str(error.filename)})
        report = {"status": "pending_external_evidence" if pending else "inputs_verified",
            "mode": "cpu_input_dry_run" if dry_run else "real_ignition_preparation", "pending_dependencies": pending,
            "policy_version": POLICY_VERSION, "human_set_development_prior": True,
            "new_baseline_evaluation_rollouts": 0, "new_baseline_evaluation_generated_tokens": 0,
            "source_preparation": str(Path(preparation).resolve()), "required_completed_result": str(Path(result).resolve()),
            "source_files": loaded["sources"], "models_loaded_at_input_check": False, "gpu_jobs_submitted": False}
        write_json(output / "INPUTS.json", report)
        if "screen" in dependencies:
            eligible, observations = observed_starts(loaded["stages"], dependencies["screen"])
            schema = ignition_schema(loaded["stages"], eligible)
            write_json(output / "challenger_schema.json", schema)
        if dry_run:
            if history is not None:
                write_json(output / "verified_round2_history.json", history)
            print(json.dumps(report), flush=True)
            return report
        source_plan, fragment, reuse = dependencies["baseline"]
        write_json(output / "baseline_plan.json", loaded["baseline"])
        write_json(output / "base/fragment.json", fragment)
        write_json(output / "baseline_reuse.json", reuse)
        write_json(output / "verified_round2_history.json", history)
        write_json(output / "verified_screen.json", dependencies["screen"])
        context = build_context(source_plan, fragment, history, dependencies["screen"], eligible, observations)
        write_json(output / "measured_context.json", context)
        proposal, challenger = sample_challenger(context, schema, loaded["stages"], eligible, output,
                                                 seed=loaded["baseline"]["training_seed"])
        request = {"round_id": "verge_minimal_ignition_round3_20260908", "training_seed": loaded["baseline"]["training_seed"],
            "base_checkpoint": str(INITIAL_SOLVER), "selection": loaded["metadata"]["selection"], "proposal": proposal,
            "challenger": challenger, "stage_validations": loaded["validations"], "B": 65536, "alpha": .25, "eta": .25,
            "q": 4, "n_min": 8, "bootstrap_resamples": 2000, "stage_source_status": context["stage_sources"],
            "proposal_policy": context["proposal_policy"]}
        write_json(output / "request.json", request)
        plan = build_plan(request)
        named.load_inputs(plan, loaded["view"] / "target_rows.json", loaded["view"] / "catalogue.json")
        validate_base_fragment(plan, fragment)
        write_json(output / "plan.json", plan)
        ready = {"status": "real_base_and_challenger_complete", "plan": str(output / "plan.json"),
            "plan_sha256": plan["plan_sha256"], "base_fragment": str(output / "base/fragment.json"),
            "baseline_contract_sha256": loaded["baseline"]["baseline_contract_sha256"], "baseline_reuse": reuse,
            "target_rows": str(loaded["view"] / "target_rows.json"), "target_rows_sha256": loaded["sources"]["target_rows.json"]["file_sha256"],
            "catalogue": str(loaded["view"] / "catalogue.json"), "catalogue_sha256": loaded["sources"]["catalogue.json"]["file_sha256"],
            "current_prompt_view_sha256": loaded["rows"]["prompt_view_sha256"], "source_files": loaded["sources"],
            "entry_source": named.source(Path(__file__)), "history_sources": history["provenance"],
            "screen_path": dependencies["screen"]["screen_path"], "screen_sources": dependencies["screen"]["provenance"],
            "proposal_policy": context["proposal_policy"],
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
    parser.add_argument("--preparation", type=Path, default=PREPARATION)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--view", type=Path, default=named.VIEW)
    parser.add_argument("--screen", type=Path, default=named.SCREEN)
    parser.add_argument("--dry-run", action="store_true")
    main(**vars(parser.parse_args()))
