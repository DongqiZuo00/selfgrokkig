"""Real-evidence exchange and scoring for one small VERGE round.

No model imports, generation, invented outcomes, GPU use or training launch.
Tests alone call the private fixture mode; the CLI accepts real_model_round.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
OPERATIONAL = HERE.parents[1] / "verge_operational_20260908"
for directory in ("protocol", "runtime", "stages"):
    sys.path.insert(0, str(OPERATIONAL / directory))

from generated_budget import Rollout, binary_advantages, verify_journal
from operational_stages import RejectionSamplingBuffer
from round_evidence import RawRollout, aggregate_rollouts
from verge_protocol import (CheckpointArchive, StageSpec, TestClasses, decompose_stages,
                            frontier, plan_round, score_round)

SCHEMA = "verge-mini-round-exchange-v1"
BRANCHES = ("g1", "g2", "g3", "direct")
CHECK_NAMES = {"executable", "nonconstant", "no_leakage", "hint_regret"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def generation_seed_for_sample(plan, instance_id, sample_id):
    """Common random seed slots across checkpoints; never a checkpoint hash."""
    return int(sha([plan["training_seed"], plan["selection"]["manifest_id"], instance_id, sample_id])[:15], 16)


def _stage_validation(stage, validation, fixture):
    _require(set(stage) == {"stage_id", "kind", "spec"}, "proposal stage needs exactly stage_id/kind/spec")
    _require(stage["kind"] in ("registered", "tape_transform", "hinted_target"), "unknown stage source")
    _require(stage["stage_id"] and stage["spec"], "stage identity and full specification required")
    _require(validation.get("accepted") is True, f"stage {stage['stage_id']} has not passed validator")
    checks = validation.get("checks", {})
    _require(set(checks) == CHECK_NAMES, "exactly the four design validator checks must be recorded")
    _require(all(checks[k] is True for k in CHECK_NAMES - {"hint_regret"}), "one of the first three stage gates failed")
    _require(validation.get("spec_sha256") == sha(stage["spec"]), "stage validator must bind this exact spec")
    _require(bool(validation.get("evidence_id")), "stage validation provenance required")
    if stage["kind"] == "hinted_target":
        probe = validation.get("details", {}).get("hint_probe", {})
        _require(checks["hint_regret"] is True and probe.get("accepted") is True,
                 "hinted target requires measured passing hint-regret evidence")
        _require(probe.get("total_rollouts") == 512 and probe.get("p_with_hint", -1) > probe.get("p_without_hint", 1)
                 and probe.get("p_without_hint", 1) < .9, "hint probe must satisfy the 512-rollout operational contract")
        _require(validation.get("hint_probe_evidence_id") and validation.get("details", {}).get("hint_provenance"),
                 "hint needs paired rollout provenance and a verified train-program library source")
    else:
        _require(checks["hint_regret"] == "NA", "non-hint stage records the fourth check explicitly as NA")
    if not fixture:
        _require(validation.get("synthetic_fixture", False) is False, "fixture stage outcomes cannot authorize a live stage")


def build_plan(request, *, _fixture=False):
    """Parse the actual Challenger JSON and freeze the four-branch exchange."""
    _require(request.get("round_id") and request.get("base_checkpoint"), "round and shared base required")
    _require(type(request.get("training_seed")) is int, "one training seed required")
    proposal = request["proposal"]
    challenger = request["challenger"]
    _require(challenger.get("checkpoint") and challenger.get("evidence_id") and "context" in challenger,
             "actual Challenger checkpoint, context and generation evidence are required")
    _require(json.loads(challenger["raw_output"]) == proposal, "parsed proposal must equal the actual Challenger JSON output")
    _require(proposal.get("base_checkpoint") == request["base_checkpoint"], "Challenger must select the declared shared base")
    _require(set(proposal["curricula"]) == {"g1", "g2", "g3"}, "exactly G=3 curricula required")
    stage_ids = {}
    typed = {}
    for branch_id, stages in proposal["curricula"].items():
        _require(len(stages) == 3, "each curriculum has exactly three ordered stages")
        for stage in stages:
            sid = stage["stage_id"]
            if sid in stage_ids:
                _require(stage_ids[sid] == stage, "stage_id cannot alias differing stage specifications")
            stage_ids[sid] = stage
            _stage_validation(stage, request["stage_validations"][sid], _fixture)
        typed[branch_id] = [StageSpec(s["kind"], s["spec"]) for s in stages]
    B = request.get("B", 65536)
    _require((type(B) is int and B >= 16) if _fixture else B in (16384, 32768, 65536),
             "mini B must be 16384, 32768 or 65536 generated tokens per branch")
    eta, alpha = request.get("eta", .25), request.get("alpha", .25)
    _require(0 < alpha < 1, "alpha must preserve both target and stage training")
    _require(request.get("beta", 0) == 0 and request.get("weight_decay", 0) == 0, "Solver beta and weight decay are zero")
    selection = request["selection"]
    instances, samples = selection["instance_ids"], selection["sample_ids"]
    _require(len(instances) == 8 and len(set(instances)) == 8, "development mini uses exactly eight frozen selection instances")
    _require(len(samples) == 32 and len(set(samples)) == 32, "freeze exactly 32 sample slots with first eight as P8")
    _require(selection.get("manifest_id"), "frozen mini selection manifest identity required")
    if not _fixture:
        _require(selection.get("prompt_view_sha256"), "freeze the common unhinted target prompt view before the real round")
        _require("test_contracts" in selection, "real scoring requires the complete frozen per-instance test-class contract")
    if "test_contracts" in selection:
        _require(set(selection["test_contracts"]) == set(instances), "test contracts must cover all mini selection instances")
        for contract in selection["test_contracts"].values():
            TestClasses(tuple(contract["test_ids"]), contract["class_by_test"], tuple(contract["expected_classes"]))
    q, n_min = request.get("q", 4), request.get("n_min", 2)
    _require(type(q) is int and q > 0 and type(n_min) is int and n_min > 0, "q and Nmin must be positive integers")
    core = plan_round(request["base_checkpoint"], typed, B=B, eta=eta)
    branches = {}
    evals = [{"alias": "base", "checkpoint_alias": "base", "branch": "base", "milestone": 0,
              "tokens": 0, "n": 32, "hint": None, "sample_ids": samples}]
    checkpoints = [{"alias": "base", "path": request["base_checkpoint"], "branch": "base", "tokens": 0}]
    for branch in (*core.curricula, core.direct):
        stages = proposal["curricula"].get(branch.branch_id, [])
        phases = []
        for index, (before, after) in enumerate(zip(branch.milestones, branch.milestones[1:]), 1):
            stage = stages[index - 1] if branch.branch_id != "direct" and index <= 3 else None
            checkpoint_alias = f"{branch.branch_id}_m{index}"
            phase = {"phase_id": f"{branch.branch_id}_segment_{index}", "segment": index,
                     "kind": "curriculum_stage" if stage else "target_only",
                     "stage_id": stage["stage_id"] if stage else None,
                     "quota": after - before, "tokens_before": before, "tokens_after": after,
                     "target_probability": alpha if stage else 1.0, "checkpoint_alias": checkpoint_alias}
            phases.append(phase)
            checkpoints.append({"alias": checkpoint_alias, "branch": branch.branch_id, "tokens": after,
                                "parent_alias": "base" if index == 1 else f"{branch.branch_id}_m{index-1}"})
            final = index == 4
            evals.append({"alias": f"{branch.branch_id}_final" if final else checkpoint_alias,
                          "checkpoint_alias": checkpoint_alias, "branch": branch.branch_id,
                          "milestone": index, "tokens": after, "n": 32 if final else 8,
                          "hint": None, "sample_ids": samples if final else samples[:8]})
        branches[branch.branch_id] = {"base_checkpoint": branch.base_checkpoint,
            "fresh_optimizer": True, "beta": 0, "weight_decay": 0, "solver_reward": "binary_full_pass",
            "generated_token_budget": B, "milestones": list(branch.milestones), "phases": phases}
    plan = {"schema_version": SCHEMA, "mode": "unit_fixture" if _fixture else "real_model_round",
            "round_id": request["round_id"], "training_seed": request["training_seed"],
            "base_checkpoint": request["base_checkpoint"], "B": B, "eta": eta, "alpha": alpha,
            "q": q, "n_min": n_min, "selection": selection, "proposal": proposal,
            "challenger": challenger, "stage_validations": request["stage_validations"],
            "stage_source_status": request.get("stage_source_status", {}),
            "branches": branches, "checkpoint_requests": checkpoints, "evaluation_requests": evals,
            "total_branch_training_generated_tokens": 4 * B,
            "planned_evaluation_rollouts": sum(x["n"] * len(instances) for x in evals),
            "scope": {"development_mini": True, "selection_instances": 8, "formal_selection_instances": 64,
                      "all_target_test_classes_retained": True, "formal_predictions_deleted": False,
                      "regime_or_effectiveness_claim_from_one_round": False,
                      "solver_reward": "binary_full_pass", "no_hint_in_any_target_evaluation": True},
            "bootstrap": {"confidence": .95, "resamples": request.get("bootstrap_resamples", 2000),
                          "random_seed": request["training_seed"]}}
    plan["plan_sha256"] = sha(plan)
    return plan


def _read_journal(branch_data, branch_plan, *, fixture):
    path = Path(branch_data["journal_path"])
    _require(file_sha(path) == branch_data["journal_sha256"], "training journal content digest mismatch")
    verify_journal(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(not any(r["kind"] == "rejected_generation" for r in rows), "incomplete/rejected generation prevents complete budget claim")
    expected_phases = branch_plan["phases"]
    phase_names = {p["phase_id"] for p in expected_phases}
    _require(all(r["payload"]["phase"] in phase_names for r in rows if r["kind"] == "rollouts"),
             "journal contains generated work outside the four planned training phases")
    reported = branch_data["phases"]
    _require(len(reported) == len(expected_phases) == 4, "four completed training segments required")
    result = []
    for expected, ledger in zip(expected_phases, reported):
        name = expected["phase_id"]
        _require(ledger["name"] == name and ledger["quota"] == expected["quota"], "phase ledger does not match its planned segment")
        starts = [r for r in rows if r["kind"] == "phase_start" and r["payload"]["name"] == name]
        ends = [r for r in rows if r["kind"] == "phase_complete" and r["payload"]["name"] == name]
        _require(len(starts) == len(ends) == 1 and ends[0]["payload"] == ledger, "phase must have one matching committed completion")
        _require(starts[0]["payload"]["generated_tokens"] == 0, "phase starts from an empty ledger")
        groups = [r["payload"] for r in rows if r["kind"] == "rollouts" and r["payload"]["phase"] == name]
        totals = {"generated_tokens": 0, "rollouts": 0, "optimizer_steps": 0, "constant_groups": 0,
                  "boundary_drain_tokens": 0, "loss_eligible_tokens": 0, "nonzero_advantage_tokens": 0}
        task_tokens = {"target": 0, "stage": 0}
        full_pass_count = {"target": 0, "stage": 0}
        first_target_success = None
        for group_index, group in enumerate(groups):
            records = group["records"]
            _require(len(records) == (1 if group["drain"] else 8), "training reward groups must be complete")
            parsed = []
            for source in records:
                item = Rollout(**{k: source[k] for k in Rollout.__dataclass_fields__})
                item.validate(group["cap"])
                parsed.append(item)
                if not fixture:
                    _require(source.get("synthetic_fixture", False) is False,
                             "synthetic training fixtures cannot be scored as live model evidence")
            _require(len({(r.prompt_id, r.task_kind, r.verifier_digest) for r in parsed}) == 1,
                     "training group mixes prompts or verifier identities")
            if expected["kind"] == "target_only":
                _require(all(r.task_kind == "target" for r in parsed), "direct and target-only tail must use only target prompts")
            tokens = sum(len(r.completion_token_ids) for r in parsed)
            totals["generated_tokens"] += tokens
            totals["rollouts"] += len(parsed)
            for index, r in enumerate(parsed):
                task_tokens[r.task_kind] += len(r.completion_token_ids)
                full_pass_count[r.task_kind] += r.reward
                if first_target_success is None and r.task_kind == "target" and r.reward:
                    first_target_success = {"phase": name, "group_index": group_index, "rollout_in_group": index,
                                            "prompt_id": r.prompt_id}
            if group["drain"]:
                totals["boundary_drain_tokens"] += tokens
            else:
                totals["loss_eligible_tokens"] += tokens
                if any(binary_advantages([r.reward for r in parsed])):
                    totals["optimizer_steps"] += 1
                    totals["nonzero_advantage_tokens"] += tokens
                else:
                    totals["constant_groups"] += 1
        _require(totals["generated_tokens"] == expected["quota"], "actual generated training tokens must exactly equal planned quota")
        _require(all(ledger[k] == v for k, v in totals.items()), "ledger differs from raw completion tokens or binary update decisions")
        result.append({"phase_id": name, **totals, "tokens_by_task": task_tokens,
                       "full_pass_by_task": full_pass_count, "first_observed_target_success": first_target_success})
    _require(sum(p["generated_tokens"] for p in result) == branch_plan["generated_token_budget"], "branch budget mismatch")
    return result


def _condition_record(source, plan, eval_request):
    keys = tuple(RawRollout.__dataclass_fields__)
    record = {k: source[k] for k in keys}
    _require(source.get("hint") is None, "all target probes and final evaluations must be unhinted")
    _require(isinstance(source.get("raw_completion"), str), "retain the actual raw model completion, including empty failures")
    tokens = source.get("completion_token_ids")
    _require(isinstance(tokens, list) and tokens and all(type(v) is int and v >= 0 for v in tokens),
             "every evaluated rollout needs actual generated token IDs")
    _require(source.get("generation_evidence_id"), "model generation evidence identifier required")
    if "prompt_view_sha256" in plan["selection"]:
        _require(source.get("prompt_view_sha256") == plan["selection"]["prompt_view_sha256"],
                 "all base/direct/curriculum target evaluations must use the identical frozen prompt view")
    contracts = plan["selection"].get("test_contracts")
    if contracts is not None:
        contract = contracts[record["instance_id"]]
        classes = TestClasses(tuple(contract["test_ids"]), contract["class_by_test"], tuple(contract["expected_classes"]))
        outcomes = source.get("test_outcomes")
        _require(isinstance(outcomes, dict), "frozen test contracts require every per-test exact-pass result")
        verified = classes.evaluate(record["conditions"][0], outcomes)
        _require(list(verified.conditions) == list(record["conditions"]), "conditions disagree with exact per-class aggregation")
    return record


def _rate_interval(batch, condition, seed):
    values = batch.evaluation.instance_rates(condition)
    rng = random.Random(seed)
    draws = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(2000))
    return {"estimate": sum(values) / len(values), "ci_low": draws[49], "ci_high": draws[1949],
            "confidence": .95, "unit": "selection_instances", "n_instances": len(values)}


def score_exchange(plan, exchange, *, _fixture=False):
    """Validate complete real evidence before producing any round result."""
    original = dict(plan)
    expected_hash = original.pop("plan_sha256")
    _require(sha(original) == expected_hash == exchange.get("plan_sha256"), "exchange must bind the intact frozen plan")
    _require(plan["schema_version"] == exchange.get("schema_version") == SCHEMA, "exchange schema version mismatch")
    mode = "unit_fixture" if _fixture else "real_model_round"
    _require(plan["mode"] == exchange.get("mode") == mode, "CLI scores real model evidence only; fixtures remain in unit tests")
    _require(set(exchange["branches"]) == set(BRANCHES), "all four branches required")
    checkpoint_requests = plan["checkpoint_requests"]
    _require(set(exchange["checkpoints"]) == {c["alias"] for c in checkpoint_requests}, "all planned checkpoint endpoints must be present")
    checkpoints = exchange["checkpoints"]
    _require(checkpoints["base"]["path"] == plan["base_checkpoint"], "base checkpoint source mismatch")
    checked_weights = {}
    for checkpoint in checkpoints.values():
        _require(checkpoint.get("path") and checkpoint.get("weights_sha256"), "checkpoint path and weights digest required")
        if not _fixture:
            weight_file = Path(checkpoint.get("weights_file", str(Path(checkpoint["path"]) / "adapter_model.safetensors")))
            if str(weight_file) not in checked_weights:
                checked_weights[str(weight_file)] = file_sha(weight_file)
            _require(checked_weights[str(weight_file)] == checkpoint["weights_sha256"], "checkpoint weights do not match their recorded digest")
    budget = {}
    for branch_id, branch_plan in plan["branches"].items():
        source = exchange["branches"][branch_id]
        _require(source.get("base_checkpoint") == plan["base_checkpoint"], "every branch must share the same base checkpoint")
        _require(source.get("fresh_optimizer") is True and source.get("optimizer_initial_state_entries") == 0,
                 "actual branch optimizer must start with empty state")
        _require(source.get("beta") == 0 and source.get("weight_decay") == 0, "binary Solver beta/wd restrictions violated")
        budget[branch_id] = _read_journal(source, branch_plan, fixture=_fixture)
        previous = checkpoints["base"]["weights_sha256"]
        for phase, audited in zip(branch_plan["phases"], budget[branch_id]):
            after = checkpoints[phase["checkpoint_alias"]]["weights_sha256"]
            audited["checkpoint_weights_changed"] = after != previous
            if not _fixture and audited["optimizer_steps"] == 0:
                _require(after == previous, "a phase with no binary update must retain exactly the same Solver weights")
            previous = after
    requests = {e["alias"]: e for e in plan["evaluation_requests"]}
    _require(set(exchange["evaluations"]) == set(requests), "all planned unhinted target evaluations must be complete")
    batches, raw_sizes = {}, {}
    sample_indices, evaluation_sources, resolving = {}, {}, set()

    def resolve_sources(alias):
        if alias in evaluation_sources:
            return evaluation_sources[alias]
        _require(alias not in resolving, "evaluation cache dependency contains a cycle")
        resolving.add(alias)
        req, supplied = requests[alias], exchange["evaluations"][alias]
        checkpoint = checkpoints[req["checkpoint_alias"]]["path"]
        _require(supplied["checkpoint"] == checkpoint, "evaluation checkpoint does not match the planned endpoint")
        cache_hit = supplied.get("evaluation_cache_hit", False)
        _require(type(cache_hit) is bool, "evaluation_cache_hit must be an explicit boolean when supplied")
        origin = supplied.get("source_evidence_id")
        if cache_hit or origin is not None:
            _require(origin in requests, "cached evaluation must reference an existing evaluation alias")
            source_checkpoint, original_rows, original_origins = resolve_sources(origin)
            _require(source_checkpoint == checkpoint, "cached evaluation must retain the exact actually evaluated checkpoint path")
            slots = set(req["sample_ids"])
            shared = {(row["instance_id"], row["sample_id"]): row for row in original_rows if row["sample_id"] in slots}
            if cache_hit:
                sources = list(shared.values())
                if "records" in supplied:
                    actual = {(row["instance_id"], row["sample_id"]): row for row in supplied["records"]}
                    _require(len(actual) == len(supplied["records"]) and set(actual) == set(shared)
                             and all(sha(actual[key]) == sha(shared[key]) for key in shared),
                             "cached records differ from their original measured source")
            else:
                sources = supplied["records"]
                actual = {(row["instance_id"], row["sample_id"]): row for row in sources}
                _require(all(key in actual and sha(actual[key]) == sha(row) for key, row in shared.items()),
                         "partial P8-to-P32 reuse differs from its original measured records")
            origins = {(row["instance_id"], row["sample_id"]): original_origins[(row["instance_id"], row["sample_id"])]
                       if (row["instance_id"], row["sample_id"]) in shared else alias for row in sources}
        else:
            sources = supplied["records"]
            origins = {(row["instance_id"], row["sample_id"]): alias for row in sources}
        resolving.remove(alias)
        evaluation_sources[alias] = checkpoint, sources, origins
        return evaluation_sources[alias]

    for alias, req in requests.items():
        supplied = exchange["evaluations"][alias]
        cache_hit = supplied.get("evaluation_cache_hit", False)
        checkpoint, sources, origins = resolve_sources(alias)
        if not _fixture:
            _require(all(r.get("synthetic_fixture", False) is False for r in sources), "synthetic condition records cannot become model results")
        raw = [_condition_record(r, plan, req) for r in sources]
        batches[alias] = aggregate_rollouts(raw, checkpoint=checkpoint, training_seed=plan["training_seed"],
            manifest_id=plan["selection"]["manifest_id"], instance_ids=plan["selection"]["instance_ids"], sample_ids=req["sample_ids"])
        new_sources = [r for r in sources if origins[(r["instance_id"], r["sample_id"])] == alias]
        raw_sizes[alias] = {"rollouts": len(new_sources),
                           "generated_tokens": sum(len(r["completion_token_ids"]) for r in new_sources),
                           "logical_rollout_slots": len(raw), "evaluation_cache_hit": cache_hit,
                           "reused_rollout_slots": len(raw) - len(new_sources),
                           "source_evidence_id": supplied.get("source_evidence_id"), "records_sha256": sha(sources)}
        for r in sources:
            if "global_rollout_index" in r:
                _require(type(r["global_rollout_index"]) is int and r["global_rollout_index"] >= 0, "global rollout index must be nonnegative integer")
                index = r["global_rollout_index"]
                actual_origin = origins[(r["instance_id"], r["sample_id"])]
                if index in sample_indices:
                    _require(sample_indices[index][1] == actual_origin and sha(sample_indices[index][2]) == sha(r),
                             "reused global rollout index must refer to an identical cached generation record")
                else:
                    sample_indices[index] = (index, actual_origin, r)
    base = batches["base"]
    finals = {b: batches[f"{b}_final"] for b in BRANCHES}
    scored = score_round(base.evaluation, finals["direct"].evaluation,
                         {b: finals[b].evaluation for b in BRANCHES if b != "direct"},
                         q=plan["q"], **plan["bootstrap"])
    breakdown = {}
    for branch_id in ("g1", "g2", "g3"):
        course = [base.first_eight()] + [batches[f"{branch_id}_m{i}"] for i in range(1, 4)] + [finals[branch_id].first_eight()]
        direct = [base.first_eight()] + [batches[f"direct_m{i}"] for i in range(1, 4)] + [finals["direct"].first_eight()]
        item = decompose_stages([b.evaluation for b in course], [b.evaluation for b in direct],
                               finals[branch_id].evaluation, finals["direct"].evaluation,
                               plan["branches"][branch_id]["milestones"], scored.kappa)
        _require(math.isclose(item.gamma, scored.gains[branch_id].estimate, abs_tol=1e-12), "Delta endpoint differs from complete-course Gamma")
        breakdown[branch_id] = {**asdict(item), "telescoping_residual": item.telescoping_residual,
            "segments": ["stage_1", "stage_2", "stage_3", "target_only_tail"],
            "final_probe_is_identical_raw_first8": True,
            "checkpoint_endpoints": {"course": [b.evaluation.checkpoint for b in course],
                                     "direct": [b.evaluation.checkpoint for b in direct]}}
    archive = CheckpointArchive()
    archive.consider(base.candidate)
    lead_before = archive.lead
    changes, occupied = {}, [{"key": list(base.candidate.key), "first_occupied_round": 0, "checkpoint": base.evaluation.checkpoint}]
    for branch_id in ("direct", "g1", "g2", "g3"):
        candidate = finals[branch_id].candidate
        action = archive.consider(candidate, **plan["bootstrap"])
        changes[branch_id] = action
        if action == "inserted":
            occupied.append({"key": list(candidate.key), "first_occupied_round": plan["round_id"], "checkpoint": candidate.evaluation.checkpoint})
    lead = archive.lead
    raw_first = next(((idx, alias, r) for idx, alias, r in sorted(sample_indices.values(), key=lambda x: x[0]) if r["conditions"][-1]), None)
    first_success = None
    if raw_first:
        idx, alias, record = raw_first
        first_success = {"scope": "provided_unhinted_selection_evaluation_stream", "global_rollout_index": idx,
                         "evaluation_alias": alias, "checkpoint": record["checkpoint"],
                         "instance_id": record["instance_id"], "sample_id": record["sample_id"],
                         "target_rate": _rate_interval(batches[alias], 6, plan["training_seed"])}
    events = {"first_success_in_ordered_evaluation_stream": first_success,
              "first_success_order_observable": bool(sample_indices),
              "training_target_successes_by_branch": {b: [p["first_observed_target_success"] for p in rows if p["first_observed_target_success"]] for b, rows in budget.items()},
              "reward_switch": {"occurred_this_round": scored.reward_switched, "kappa": scored.kappa, "q": plan["q"]},
              "learning_onset": {"observed_on_lead": lead.evaluation.rate(6) >= .05,
                                  "first_crossing_in_this_round": lead_before.evaluation.rate(6) < .05 <= lead.evaluation.rate(6),
                                  "lead_target_rate": _rate_interval(next(v for v in [base, *finals.values()] if v.evaluation.checkpoint == lead.evaluation.checkpoint), 6, plan["training_seed"])},
              "first_cell_occupations": occupied,
              "detour": {"occurred_this_round": base.evaluation.checkpoint != lead_before.evaluation.checkpoint,
                         "meaning": "this first-round archive starts from the sole initial base"}}
    buffer = RejectionSamplingBuffer(plan["n_min"])
    candidates = [{"branch_id": b, "gamma": gain.estimate, "paired_ci": [gain.ci_low, gain.ci_high],
                   "curriculum": plan["proposal"]["curricula"][b], "per_stage_delta": breakdown[b],
                   "first_success_observed": finals[b].evaluation.success_count > 0,
                   "evidence_id": f"{plan['round_id']}/round_score.json#{b}"}
                  for b, gain in scored.gains.items()]
    buffer.add_round(round_id=plan["round_id"], seed=plan["training_seed"], context=plan["challenger"]["context"],
                     base={"checkpoint_id": plan["base_checkpoint"], "cell": list(base.candidate.key)},
                     candidates=candidates, j_plus=list(scored.j_plus), provenance={"run_id": plan["round_id"],
                         "condition_id": f"frozen_kappa_{scored.kappa}", "evidence_id": f"{plan['round_id']}/round_score.json"})
    rft = buffer.plan_update()
    lineage = [{"alias": c["alias"], "checkpoint": checkpoints[c["alias"]], "parent_alias": c.get("parent_alias"),
                "branch": c["branch"], "generated_tokens_from_base": c["tokens"]} for c in checkpoint_requests]
    curriculum_updates = sum(p["optimizer_steps"] for b in ("g1", "g2", "g3") for p in budget[b])
    changed_curriculum_phases = sum(p["optimizer_steps"] > 0 and p["checkpoint_weights_changed"]
                                   for b in ("g1", "g2", "g3") for p in budget[b])
    summary = {"mode": mode, "status": "complete", "round_id": plan["round_id"],
               "scope": plan["scope"], "plan_sha256": expected_hash, "exchange_sha256": sha(exchange),
               "B_per_branch": plan["B"], "total_training_generated_tokens": sum(p["generated_tokens"] for rows in budget.values() for p in rows),
               "evaluation_rollouts": sum(x["rollouts"] for x in raw_sizes.values()),
               "logical_evaluation_rollout_slots": sum(x["logical_rollout_slots"] for x in raw_sizes.values()),
               "cached_evaluation_aliases": sum(x["evaluation_cache_hit"] for x in raw_sizes.values()),
               "evaluation_generated_tokens": sum(x["generated_tokens"] for x in raw_sizes.values()),
               "real_curriculum_mixed_binary_optimizer_steps": curriculum_updates,
               "curriculum_phases_with_binary_updates_and_changed_weights": changed_curriculum_phases,
               "at_least_one_curriculum_mixed_update": curriculum_updates > 0 and changed_curriculum_phases > 0,
               "kappa": scored.kappa, "j_plus": list(scored.j_plus), "lead_checkpoint": lead.evaluation.checkpoint,
               "lead_target_rate": lead.evaluation.rate(6), "archive_cells": len(archive.cells),
               "rft_status": "eligible_plan_not_executed" if rft else "not_eligible_no_new_update",
               "rft_completed_updates": 0, "research_predictions_established": False}
    return {"summary": summary, "round_score": asdict(scored), "delta": breakdown, "events": events,
            "archive": {"changes": changes, "cells": [{"key": list(key), "candidate": asdict(value)} for key, value in archive.cells.items()],
                        "lead": asdict(lead), "saturated": archive.saturated},
            "budget_audit": budget, "evaluation_provenance": raw_sizes, "lineage": lineage,
            "challenger_buffer": buffer.to_dict(), "rft_plan": rft}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--request", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--plan", type=Path, required=True)
    scorer.add_argument("--exchange", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "plan":
        result = build_plan(json.loads(args.request.read_text(encoding="utf-8-sig")))
        write_json(args.output, result)
        print(json.dumps({"status": "planned", "plan_sha256": result["plan_sha256"], "evaluation_requests": len(result["evaluation_requests"]),
                          "evaluation_rollouts": result["planned_evaluation_rollouts"]}))
    else:
        _require(not args.output.exists(), "score outputs are append-only; choose an independent result directory")
        result = score_exchange(json.loads(args.plan.read_text(encoding="utf-8-sig")),
                                json.loads(args.exchange.read_text(encoding="utf-8-sig")))
        for name, value in result.items():
            write_json(args.output / f"{name}.json", value)
        write_json(args.output / "input_provenance.json", {"plan_path": str(args.plan), "plan_file_sha256": file_sha(args.plan),
                                                          "exchange_path": str(args.exchange), "exchange_file_sha256": file_sha(args.exchange)})
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
