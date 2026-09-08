"""One actual CPU execution of VERGE component integration with synthetic data.

No model is loaded, sampled or optimized. Token IDs, rewards, condition records,
hint-regret arms and mutable callback state are explicit engineering fixtures.
The pinned real Manufactoria parser separately verifies the reference witness
on actual frozen target cases, and its class aggregation must match TestClasses.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
for subdir in ("benchmark", "protocol", "runtime", "stages"):
    sys.path.insert(0, str(ROOT / subdir))

import benchmark as b
from generated_budget import JsonlJournal, PhaseLedger, Rollout, run_phase, verify_journal
from operational_stages import (HintProbe, RegisteredFamily, RejectionSamplingBuffer, VerifiedProgramLibrary,
    generate_cases, instance_tuple_from_manifest, render_target_prompt, validate_stage)
from round_evidence import RawRollout, aggregate_rollouts
from verge_protocol import CheckpointArchive, StageSpec, TestClasses, decompose_stages, plan_round, score_round

MODE = "synthetic_CPU_integration_not_model_training"
SEED = 20260908
SAMPLE_IDS = tuple(f"synthetic_slot_{i:02d}" for i in range(32))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_rows(split):
    return [json.loads(line) for line in (ROOT / "benchmark/generated" / f"target_{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def source_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(parser_path, run_path):
    if run_path.exists():
        raise FileExistsError("integration runs are append-only: choose a fresh run directory")
    run_path.mkdir(parents=True)
    parser = b.load_python(parser_path, "integration_real_vendor_parser")
    manifest = json.loads((ROOT / "benchmark/generated/manifest.json").read_text(encoding="utf-8"))
    rows = {split: read_rows(split) for split in b.SPLIT_COUNTS}
    target_definition = {"benchmark_version": b.VERSION, "parameters": b.PARAMETERS,
                         "input_alphabet": b.INPUT_ALPHABET, "max_input_length": b.MAX_LENGTH}
    protected = [instance_tuple_from_manifest(target_definition, [c["input"] for c in row["ground_truth"]])
                 for split in ("selection", "heldout", "test") for row in rows[split]]
    train_identity = instance_tuple_from_manifest(target_definition, [c["input"] for c in rows["train"][0]["ground_truth"]])
    assert train_identity not in protected and all(len(x) == 64 for x in train_identity)
    save(run_path / "provenance.json", {"mode": MODE, "model_loaded": False, "model_rollouts": 0,
        "model_optimizer_steps": 0, "GPU_used": False, "training_seed_label_is_fixture": SEED,
        "benchmark_manifest_sha256": source_digest(ROOT / "benchmark/generated/manifest.json"),
        "parser_path": str(parser_path), "parser_sha256": source_digest(parser_path),
        "target_definition": target_definition, "train_instance_tuple": train_identity,
        "protected_instance_tuples": protected,
        "sources": {str(p.relative_to(ROOT)): source_digest(p) for p in (
            ROOT / "integration_round.py", ROOT / "benchmark/benchmark.py", ROOT / "protocol/verge_protocol.py",
            ROOT / "protocol/round_evidence.py", ROOT / "runtime/generated_budget.py", ROOT / "stages/operational_stages.py")}})

    # Real benchmark -> original interpreter -> both verifier aggregators.
    reference_evidence = []
    for row in rows["selection"][:2]:
        cases = row["ground_truth"]
        actual = b.evaluate_program(b.reference_program(), cases, parser.create_robot_factory)
        class_adapter = TestClasses(tuple(c["test_id"] for c in cases),
            {c["test_id"]: c["test_class"] for c in cases}, expected_classes=b.CLASS_IDS)
        outcomes = {c["test_id"]: bool(result["pass"]) for c, result in zip(cases, actual["per_test"])}
        aggregated = class_adapter.evaluate(actual["parse_valid"], outcomes)
        assert aggregated.class_rates == actual["class_rates"]
        assert aggregated.weakest_class_rate == actual["f"] == 1
        assert aggregated.binary_reward == actual["reward"] == 1
        assert len(cases) == 36 and all(aggregated.conditions)
        reference_evidence.append({"instance": row["id"], "kind": "handwritten_CPU_oracle_not_model_completion",
                                   "benchmark": actual, "protocol": asdict(aggregated)})
    save(run_path / "actual_benchmark_bridge.json", reference_evidence)
    full_conditions = tuple(reference_evidence[0]["protocol"]["conditions"])

    # Library/probe below are synthetic integration fixtures only. The program's
    # exact verification is real; the invented hint-success rates are not evidence
    # of transfer, and no live training library or target prompt is modified.
    small_inputs = [x for x in b.input_support() if b.split_of(x) == "train" and len(x) <= 4][:4]
    factory = parser.create_robot_factory(b.reference_program())
    small_passes = [int((r := factory.process_robot(x)).finished and r.final_tape == b.expected_output(x)) for x in small_inputs]
    assert len(small_inputs) >= 2 and all(small_passes)
    hint_identity = instance_tuple_from_manifest(target_definition, small_inputs)
    library = VerifiedProgramLibrary()
    hint = library.add(entry_id="SYNTHETIC_ORACLE_LIBRARY_FIXTURE", program=b.reference_program(),
        family="fixed_prepend_target", instance_size=max(map(len, small_inputs)), source_instance_tuple=hint_identity,
        source_split="train", verification_id="CPU-small-oracle-check", verifier_protocol_id=b.VERSION,
        verification_test_ids=[b.sha({"input": x}) for x in small_inputs], exact_pass=small_passes)
    distribution = {"alphabet": "RB", "min_length": 0, "max_length": 64, "corner_cases": small_inputs}
    base_spec = {"distribution": distribution}
    registered_definition = {"family": "prepend_identity", "prefix": "R", "alphabet": "RB", "tier": 0, "mutation_tier": 0}
    expression = {"op": "reverse", "arg": {"op": "input"}}
    transform_definition = {"family": "tape_transform", "expression": expression, "alphabet": "RB"}
    stage_cases = generate_cases(distribution)
    registered_identity = instance_tuple_from_manifest(registered_definition, stage_cases)
    transform_identity = instance_tuple_from_manifest(transform_definition, stage_cases)
    assert len({train_identity[0], registered_identity[0], transform_identity[0]}) == 3
    registered = {**base_spec, "instance_tuples": [list(registered_identity)],
                  "kind": "registered", "family": "prepend_identity", "tier": 0, "mutation_tier": 0}
    transform = {**base_spec, "instance_tuples": [list(transform_identity)], "kind": "tape_transform", "expression": expression}
    hinted = {**base_spec, "instance_tuples": [list(train_identity)], "kind": "hinted_target", "family": "fixed_prepend_target", "tier": 0, "mutation_tier": 0,
              "target_size": 64, "hint_id": hint.entry_id, "hint_relation": "smaller_train"}
    registry = {"prepend_identity": RegisteredFamily(lambda x, _: "R" + x, "synthetic-registry-prepend-identity-v1"),
                "fixed_prepend_target": RegisteredFamily(lambda x, _: b.expected_output(x), b.VERSION)}
    hint_probe = HintProbe(with_hint=[[0, 1] * 4 for _ in range(32)], without_hint=[[0] * 8 for _ in range(32)],
        instance_ids=[r["id"] for r in rows["train"][:32]], rollout_seed_ids=[list(range(i*8, i*8+8)) for i in range(32)],
        condition_id="SYNTHETIC_HINT_BINARY_OUTCOMES_NO_MODEL_SAMPLING")
    specs = [registered, transform, hinted]
    validations = []
    for spec in specs:
        extra = {"library": library, "hint_probe": hint_probe} if spec["kind"] == "hinted_target" else {}
        value = validate_stage(spec, registry=registry, protected_tuples=protected, **extra)
        assert value.accepted, value.details
        validations.append(value.to_dict())
    target_prompt = rows["selection"][0]["messages"][0]["content"]
    assert render_target_prompt(target_prompt, final_evaluation=True) == target_prompt
    hinted_fixture = render_target_prompt(target_prompt, hint=hint)
    assert hinted_fixture != target_prompt
    save(run_path / "stage_validation.json", {"mode": MODE, "specifications": specs, "validations": validations,
        "identity_definitions": {"registered": registered_definition, "tape_transform": transform_definition,
                                 "hinted_target": target_definition},
        "hint_probe_outcomes_are_synthetic": True, "hint_probe": asdict(hint_probe),
        "small_oracle_inputs": small_inputs, "small_oracle_exact_pass": small_passes,
        "library_is_synthetic_fixture_only": True, "final_target_has_hint": False})

    ordered = {
        "g1": specs,
        "g2": [transform, registered, hinted],
        "g3": [hinted, transform, registered],
    }
    plan = plan_round("SYNTHETIC_SHARED_BASE", {name: [StageSpec(s["kind"], s) for s in curriculum]
                      for name, curriculum in ordered.items()}, B=131, eta=.25)
    save(run_path / "branch_plan.json", {"mode": MODE, "plan": asdict(plan), "alpha": .25})
    ledgers, update_counts = {}, {}
    for branch in (*plan.curricula, plan.direct):
        journal_path = run_path / "budget_journals" / f"{branch.branch_id}.jsonl"
        journal = JsonlJournal(journal_path, run_path.name + "_" + branch.branch_id)
        # Four independent fresh synthetic callback states, all at same base.
        state = {"value": 0, "updates": 0, "mode": MODE, "base": plan.base_checkpoint}
        rng = random.Random(f"{SEED}|{branch.branch_id}")
        quotas = [b-a for a, b in zip(branch.milestones, branch.milestones[1:])]
        branch_ledgers = []
        for phase_index, quota in enumerate(quotas):
            tail = phase_index == len(quotas) - 1
            direct = branch.branch_id == "direct"
            def generate(n, cap, drain):
                kind = "target" if direct or tail or rng.random() < .25 else "stage"
                # Synthetic complete tokens are counted exactly by production
                # budget accounting. No real tokenizer or generation is implied.
                return [Rollout(prompt_id=f"SYNTHETIC_{branch.branch_id}_{phase_index}_{kind}", task_kind=kind,
                    completion_token_ids=tuple(range(cap)), reward=0 if kind == "target" else i % 2,
                    raw_completion=f"SYNTHETIC fixture completion {i}", verifier_digest=b.VERSION + "_SYNTHETIC_binary")
                    for i in range(n)]
            def update(outputs, advantages):
                assert len(outputs) == 8 and any(advantages)
                state["updates"] += 1
                state["value"] += 1  # Callback counter only; not a model/optimizer.
            ledger = run_phase(PhaseLedger(f"segment_{phase_index+1}" + ("_target_tail" if tail else ""), quota),
                generate=generate, update=update, journal=journal, max_new_tokens=4)
            branch_ledgers.append(asdict(ledger))
            save(run_path / "synthetic_states" / f"{branch.branch_id}_tokens_{branch.milestones[phase_index+1]}.json", state)
        assert sum(x["generated_tokens"] for x in branch_ledgers) == 131
        assert sum(x["optimizer_steps"] for x in branch_ledgers) == state["updates"]
        assert verify_journal(journal_path) > 0
        ledgers[branch.branch_id] = branch_ledgers
        update_counts[branch.branch_id] = state["updates"]
        if direct:
            assert state["value"] == state["updates"] == 0
    save(run_path / "budget_summary.json", {"mode": MODE, "generated_tokens_are_synthetic": True,
        "branch_ledgers": ledgers, "callback_update_counts_not_model_steps": update_counts,
        "total_training_fixture_tokens": 4 * 131, "actual_model_generated_tokens": 0})

    instances = tuple(r["id"] for r in rows["selection"][:2])
    manifest_id = "SYNTHETIC_CONDITIONS_USING_FROZEN_INSTANCE_IDS|" + manifest["split_rows"]["selection"]["sha256"]
    def batch(checkpoint, successes, *, cycle=False, n=32):
        slots = SAMPLE_IDS[:n]
        raw = [RawRollout(checkpoint, SEED, manifest_id, instance, sample,
            full_conditions if i in successes else (True, False, False, False, False, False), cycle)
            for instance in instances for i, sample in enumerate(slots)]
        result = aggregate_rollouts(raw, checkpoint=checkpoint, training_seed=SEED, manifest_id=manifest_id,
                                    instance_ids=instances, sample_ids=slots)
        save(run_path / "raw_synthetic_condition_records" / f"{checkpoint}_{n}.json", {"mode": MODE,
             "conditions_are_invented_fixture_not_model_verification": True, "records": [asdict(r) for r in result.raw_records]})
        return result
    base = batch(plan.base_checkpoint, set())
    final = {"direct": batch("SYNTHETIC_direct_final", set()),
             "g1": batch("SYNTHETIC_g1_final", set(range(2, 10)), cycle=True),
             "g2": batch("SYNTHETIC_g2_final", set(range(4)), cycle=False),
             "g3": batch("SYNTHETIC_g3_final", set(), cycle=True)}
    scored = score_round(base.evaluation, final["direct"].evaluation,
        {k: v.evaluation for k, v in final.items() if k != "direct"}, q=4, resamples=1000, random_seed=SEED)
    assert scored.reward_switched and scored.kappa == 6 and set(scored.j_plus) == {"g1", "g2"}
    archive = CheckpointArchive()
    archive.consider(base.candidate)
    changes = {name: archive.consider(value.candidate, resamples=1000, random_seed=SEED) for name, value in final.items()}
    assert len(archive.cells) == 2
    save(run_path / "archive.json", {"mode": MODE, "changes": changes,
        "cells": [{"key": key, "candidate": asdict(value)} for key, value in archive.cells.items()],
        "lead": asdict(archive.lead), "saturated": archive.saturated})
    delta_rows = {}
    for branch in plan.curricula:
        name = branch.branch_id
        course = [base.first_eight()]
        direct_path = [base.first_eight()]
        stage_success_counts = {"g1": (2, 6, 7), "g2": (1, 2, 5), "g3": (1, 3, 2)}[name]
        for idx, count in enumerate(stage_success_counts):
            course.append(batch(f"SYNTHETIC_{name}_stage{idx+1}", set(range(count)), cycle=name != "g2", n=8))
            direct_path.append(batch(f"SYNTHETIC_direct_stage{idx+1}", set(), n=8))
        final_probe, direct_probe = final[name].first_eight(), final["direct"].first_eight()
        assert all(any(record is source for source in final[name].raw_records) for record in final_probe.raw_records)
        course.append(final_probe)
        direct_path.append(direct_probe)
        result = decompose_stages([x.evaluation for x in course], [x.evaluation for x in direct_path],
            final[name].evaluation, final["direct"].evaluation, branch.milestones, scored.kappa)
        assert len(result.deltas) == 4 and abs(result.telescoping_residual) < 1e-12
        assert result.gamma == scored.gains[name].estimate
        delta_rows[name] = {**asdict(result), "telescoping_residual": result.telescoping_residual,
            "segments": ["stage_1", "stage_2", "stage_3", "target_only_tail"],
            "final_first8_derived_from_identical_raw_objects": True,
            "first8_records": [asdict(x) for x in final_probe.raw_records]}
    assert delta_rows["g1"]["deltas"][-1] < 0  # Tail loss is retained, never zero-filled.
    save(run_path / "round_score_and_delta.json", {"mode": MODE, "score": asdict(scored), "delta": delta_rows})
    buffer = RejectionSamplingBuffer(n_min=2)
    candidates = [{"branch_id": name, "gamma": gain.estimate, "paired_ci": [gain.ci_low, gain.ci_high],
        "curriculum": ordered[name], "evidence_id": f"{run_path.name}/round_score_and_delta.json#{name}"}
        for name, gain in scored.gains.items()]
    buffer.add_round(round_id=run_path.name, seed=SEED,
        context={"mode": MODE, "delta": delta_rows, "kappa": scored.kappa, "incumbents": [list(k) for k in archive.cells]},
        base={"checkpoint_id": plan.base_checkpoint, "cell": list(base.candidate.key)},
        candidates=candidates, j_plus=list(scored.j_plus), provenance={"run_id": run_path.name,
            "condition_id": f"SYNTHETIC_kappa_{scored.kappa}", "evidence_id": str(run_path / "round_score_and_delta.json")})
    rft_plan = buffer.plan_update()
    assert rft_plan is not None and rft_plan["epochs"] == 1 and len(rft_plan["cumulative_examples"]) == 2
    assert buffer.new_count == 2 and not buffer.completed_updates and not buffer.trained_ids
    save(run_path / "challenger_buffer.json", {"mode": MODE, "rft_training_executed": False, "buffer": buffer.to_dict()})
    save(run_path / "rft_plan_only.json", {"mode": MODE, "status": "PLAN_ONLY_NO_TRAINER_CALLED", "plan": rft_plan})
    summary = {"mode": MODE, "status": "PASS", "actual_model_rollouts": 0, "actual_model_optimizer_steps": 0,
        "actual_GPU_used": False, "actual_reference_verifier_cases": 2 * 36 + len(small_inputs),
        "all_three_stage_validators_accepted_synthetic_fixture": True,
        "branches": 4, "synthetic_generated_tokens_per_branch": 131,
        "synthetic_generated_tokens_total": 524, "shared_base": plan.base_checkpoint,
        "direct_synthetic_callback_state_unchanged": update_counts["direct"] == 0,
        "score_kappa": scored.kappa, "j_plus": list(scored.j_plus), "archive_cells": len(archive.cells),
        "delta_includes_target_only_tail": True, "raw_final_first8_identity_checked": True,
        "all_telescoping_residuals_zero": all(abs(x["telescoping_residual"]) < 1e-12 for x in delta_rows.values()),
        "challenger_RFT": "plan_only_not_executed", "run_directory": str(run_path)}
    save(run_path / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--parser", type=Path, default=Path("/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py"))
    cli.add_argument("--output", type=Path, default=ROOT / "runs" / ("cpu_round_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")))
    args = cli.parse_args()
    main(args.parser, args.output)
