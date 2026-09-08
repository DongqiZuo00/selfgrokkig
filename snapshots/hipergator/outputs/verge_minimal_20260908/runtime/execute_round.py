"""Execute frozen VERGE base/branch jobs and merge their real evidence.

CLI:
  base --plan plan.json --target-rows target_rows.json --output base_dir
  branch --branch g1 --plan plan.json --target-rows target_rows.json
         --catalogue catalogue.json --base-evidence base_dir/fragment.json --output g1_dir
  merge --plan plan.json --base-evidence base_dir/fragment.json
        --branch-evidence g1/fragment.json g2/fragment.json g3/fragment.json direct/fragment.json
        --output merged_dir

Sampling seeds identify an actual batch of eight per instance/chunk. They are
not advertised as independently assigned per-sample seeds. No reference/oracle
program is loaded, and saved checkpoints are created only after weights change.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "verge_operational_20260908"
sys.path[:0] = [str(ROOT / "round"), str(PRIOR / "runtime"), str(PRIOR / "protocol"), str(PRIOR / "benchmark")]
from round_cli import SCHEMA, BRANCHES, file_sha, score_exchange, sha, write_json
from generated_budget import JsonlJournal, PhaseLedger, Rollout, assert_fresh_optimizer, run_phase
from verge_protocol import TestClasses
from benchmark import extract_program

WORK = Path("/blue/du.j/jinjiaguo/self grok")
CAP = 2048
BASELINE_SCHEMA = "verge-mini-preproposal-baseline-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def baseline_contract(plan):
    return {"base_checkpoint": plan["base_checkpoint"], "training_seed": plan["training_seed"],
            "selection": plan["selection"], "samples_per_instance": 32, "actual_sampling_chunk_size": 8,
            "max_new_tokens": CAP, "chunk_seed_policy": "actual_batch8_chunk_seed_v1",
            "decoding_policy": "dsl_grammar_v1", "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "target_hint": None}


def build_baseline_plan(*, base_checkpoint, training_seed, selection):
    """Before Challenger context/proposal exists: freeze only actual base P32."""
    require(base_checkpoint and type(training_seed) is int, "audited base checkpoint and integer seed required")
    require(len(selection["instance_ids"]) == 8 and len(set(selection["instance_ids"])) == 8,
            "freeze eight selection instances before base sampling")
    require(len(selection["sample_ids"]) == 32 and len(set(selection["sample_ids"])) == 32,
            "freeze 32 actual sample slots before base sampling")
    require(selection.get("manifest_id") and selection.get("prompt_view_sha256") and selection.get("test_contracts"),
            "base contract needs frozen target view and full test-class membership")
    plan = {"schema_version": BASELINE_SCHEMA, "mode": "real_model_base_preproposal",
            "base_checkpoint": str(base_checkpoint), "training_seed": training_seed, "selection": selection,
            "evaluation_requests": [{"alias": "base", "checkpoint_alias": "base", "branch": "base", "milestone": 0,
                                     "tokens": 0, "n": 32, "hint": None, "sample_ids": selection["sample_ids"]}]}
    plan["baseline_contract_sha256"] = sha(baseline_contract(plan))
    return plan


def checked_plan(path_or_value, *, allow_baseline=False):
    plan = load_json(path_or_value) if isinstance(path_or_value, (str, Path)) else copy.deepcopy(path_or_value)
    if allow_baseline and plan.get("schema_version") == BASELINE_SCHEMA:
        require(plan.get("baseline_contract_sha256") == sha(baseline_contract(plan)), "preproposal base contract changed")
        require(plan["evaluation_requests"] == build_baseline_plan(base_checkpoint=plan["base_checkpoint"],
                training_seed=plan["training_seed"], selection=plan["selection"])["evaluation_requests"],
                "preproposal plan may request only the frozen unhinted base P32")
        return plan
    unsigned = dict(plan)
    stored = unsigned.pop("plan_sha256")
    require(sha(unsigned) == stored and plan["schema_version"] == SCHEMA, "plan changed after freezing")
    return plan


def validate_base_fragment(plan, fragment):
    require(fragment["fragment_role"] == "base", "actual base evaluation fragment required")
    require(fragment.get("baseline_contract_sha256") == sha(baseline_contract(plan)),
            "complete plan changed the base sampling contract used before Challenger proposal")
    if not fragment.get("preproposal_base", False):
        require(fragment.get("plan_sha256") == plan.get("plan_sha256"), "base evaluation belongs to another frozen round")
    require(fragment["checkpoints"]["base"] == checkpoint_record(plan["base_checkpoint"]),
            "base checkpoint changed since its actual evaluation")


def load_inputs(plan, target_rows_path, catalogue_path=None, *, _fixture=False):
    rows = load_json(target_rows_path)
    require(rows["manifest_id"] == plan["selection"]["manifest_id"], "target view manifest does not match round plan")
    require(rows["prompt_view_sha256"] == plan["selection"]["prompt_view_sha256"], "target prompt view does not match round plan")
    selected = {r["id"]: r for r in rows["selection"]}
    require(list(selected) == plan["selection"]["instance_ids"] and len(selected) == 8,
            "exact frozen eight selection instances required in their original order")
    for row in [*rows["train"], *rows["selection"]]:
        require(row.get("hint") is None, "all target training/probe rows must be unhinted")
        require(row["messages"] == rows["train"][0]["messages"], "all target rows must use the same frozen prompt view")
        require(row.get("ground_truth"), "target row has no executable tests")
    if not _fixture:
        metadata = load_json(Path(target_rows_path).with_name("target_view_metadata.json"))
        require(metadata["target_rows_file_sha256"] == file_sha(target_rows_path), "target rows differ from their frozen prompt-view file")
        require(sha(metadata["prompt_view_hash_payload"]) == rows["prompt_view_sha256"], "target prompt-view identity is invalid")
        payload = metadata["prompt_view_hash_payload"]
        require(payload["messages"] == rows["train"][0]["messages"], "target message text differs from the frozen view")
        require(all(sha(row["ground_truth"]) == payload["ground_truth_sha256_by_instance"][row["id"]]
                    for row in [*rows["train"], *rows["selection"]]), "target ground truth differs from the frozen view")
        original_path = PRIOR / "benchmark/generated/target_train.jsonl"
        require(file_sha(original_path) == rows["source_train_manifest_sha256"], "original train manifest hash changed")
        original = {r["id"]: r for r in (json.loads(line) for line in original_path.read_text().splitlines())}
        require(len(rows["train"]) == len(original), "all original train rows must remain available")
        for row in rows["train"]:
            require({k: v for k, v in row.items() if k != "messages"} ==
                    {k: v for k, v in original[row["id"]].items() if k != "messages"},
                    "train input differs from its frozen source beyond the approved prompt view")
        for row in rows["selection"]:
            contract = plan["selection"]["test_contracts"][row["id"]]
            require(len(row["ground_truth"]) == 36 and [c["test_id"] for c in row["ground_truth"]] == contract["test_ids"],
                    "all 36 frozen target tests must be retained")
            require(all(contract["class_by_test"][c["test_id"]] == c["test_class"] for c in row["ground_truth"]),
                    "target class membership changed")
    stages = {}
    if catalogue_path:
        catalogue = load_json(catalogue_path)
        for stage in catalogue["stages"]:
            require(stage["stage_id"] not in stages, "duplicate catalogue stage id")
            stages[stage["stage_id"]] = stage
        for curriculum in plan["proposal"]["curricula"].values():
            for chosen in curriculum:
                require(chosen["stage_id"] in stages, "Challenger stage is missing from the frozen executable catalogue")
                stage = stages[chosen["stage_id"]]
                require(stage["kind"] == chosen["kind"] and stage["spec"] == chosen["spec"], "executed catalogue spec differs from the Challenger plan")
                require(stage["rows"], "catalogue stage has no actual training rows")
    return rows, stages


def parameter_fingerprint(solver):
    """Hash actual trainable tensor bytes, avoiding checkpoint serialization noise."""
    h = hashlib.sha256()
    for parameter in solver.parameters:
        value = parameter.detach().cpu().contiguous()
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(solver.torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint_record(path):
    path = Path(path).resolve()
    weights = path / "adapter_model.safetensors"
    return {"path": str(path), "weights_file": str(weights), "weights_sha256": file_sha(weights)}


def graph_has_cycle(factory):
    """Whole parsed directed node graph, including unreachable components."""
    nodes = factory.nodes
    successors = {name: {route.target for route in node.routes if route.target in nodes} for name, node in nodes.items()}
    indegree = dict.fromkeys(nodes, 0)
    for targets in successors.values():
        for target in targets:
            indegree[target] += 1
    ready = [name for name, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(nodes)


def chunk_seed(plan, instance_id, chunk_index):
    return int(sha(["actual_batch8_chunk_seed_v1", plan["training_seed"], plan["selection"]["manifest_id"],
                    plan["selection"]["prompt_view_sha256"], instance_id, chunk_index])[:15], 16)


def _resolve_records(alias, evaluations, active=None):
    active = set() if active is None else active
    require(alias not in active, "cached evaluation dependency cycle")
    item = evaluations[alias]
    if "records" in item:
        return item["records"]
    active.add(alias)
    records = _resolve_records(item["source_evidence_id"], evaluations, active)
    active.remove(alias)
    samples = set(item["sample_ids"])
    return [r for r in records if r["sample_id"] in samples]


def evaluate_request(solver, plan, request, checkpoint, selected_rows, output, known, *, _fixture=False):
    """Measure missing actual eight-sample chunks; reuse already measured slots."""
    alias = request["alias"]
    required_samples = request["sample_ids"]
    candidates = []
    for source_alias, item in known.items():
        if item["checkpoint"] == checkpoint["path"] and item.get("weights_sha256") == checkpoint["weights_sha256"]:
            source = _resolve_records(source_alias, known)
            coverage = len({r["sample_id"] for r in source} & set(required_samples))
            candidates.append((coverage, source_alias, source))
    candidates.sort(key=lambda item: item[0], reverse=True)
    origin = candidates[0][1] if candidates and candidates[0][0] else None
    source_records = candidates[0][2] if origin else []
    existing = {(r["instance_id"], r["sample_id"]): r for r in source_records if r["sample_id"] in required_samples}
    expected = {(i, s) for i in plan["selection"]["instance_ids"] for s in required_samples}
    if set(existing) == expected:
        result = {"checkpoint": checkpoint["path"], "weights_sha256": checkpoint["weights_sha256"],
                  "evaluation_cache_hit": True, "source_evidence_id": origin, "sample_ids": required_samples}
        write_json(output / "evaluations" / f"{alias}.json", result)
        known[alias] = result
        return result
    new_records = []
    for instance_id in plan["selection"]["instance_ids"]:
        row = selected_rows[instance_id]
        contract = plan["selection"]["test_contracts"][instance_id]
        verifier = TestClasses(tuple(contract["test_ids"]), contract["class_by_test"], tuple(contract["expected_classes"]))
        for start in range(0, len(required_samples), 8):
            sample_ids = required_samples[start:start + 8]
            require(len(sample_ids) == 8, "evaluation uses complete native eight-sample chunks")
            cached = [(instance_id, sample) in existing for sample in sample_ids]
            if all(cached):
                continue
            require(not any(cached), "partially cached eight-sample chunk cannot be resampled under a different batch")
            index = start // 8
            seed = chunk_seed(plan, instance_id, index)
            source_path = output / "raw_evaluation" / f"{alias}__{sha(instance_id)[:12]}__chunk{index}.json"
            sampled_at = time.time()
            raw = solver.generate(row, 8, CAP, seed, output=source_path, target=True)
            source_sha256 = file_sha(source_path)
            require(len(raw["completion_token_ids"]) == len(raw["verification"]) == len(raw["verifier_completions"]) == 8,
                    "backend returned an incomplete evaluation chunk")
            if not _fixture:
                require(raw.get("decoding_policy") == "dsl_grammar_v1", "real pilot must use the frozen grammar policy")
            for slot, (sample_id, tokens, text, verification) in enumerate(zip(sample_ids, raw["completion_token_ids"],
                                                                              raw["verifier_completions"], raw["verification"])):
                per_test = verification["per_test"]
                require(len(per_test) == len(row["ground_truth"]), "verifier dropped a frozen target test")
                outcomes = {}
                for case, outcome in zip(row["ground_truth"], per_test):
                    require(outcome["input"] == case["input"], "verifier test ordering changed")
                    require(type(outcome["pass"]) is int and outcome["pass"] in (0, 1), "test result is not binary")
                    outcomes[case["test_id"]] = bool(outcome["pass"])
                conditions = verifier.evaluate(bool(verification["parse_valid"]), outcomes).conditions
                require(int(conditions[-1]) == verification["reward"], "target binary reward differs from frozen full-pass aggregation")
                cycle = graph_has_cycle(solver.create_factory(extract_program(text))) if conditions[0] else None
                record = {"checkpoint": checkpoint["path"], "training_seed": plan["training_seed"],
                    "manifest_id": plan["selection"]["manifest_id"], "instance_id": instance_id, "sample_id": sample_id,
                    "conditions": list(conditions), "has_cycle": cycle, "raw_completion": text,
                    "completion_token_ids": list(tokens), "hint": None,
                    "generation_evidence_id": f"{source_path}#sample{slot}", "raw_generation_sha256": source_sha256,
                    "prompt_view_sha256": plan["selection"]["prompt_view_sha256"], "test_outcomes": outcomes,
                    "chunk_seed": seed, "chunk_index": index, "sample_index_in_chunk": slot,
                    "sampling_seed_scope": "one_actual_eight_sample_batch_per_instance_chunk",
                    "chunk_started_at_unix": sampled_at, "synthetic_fixture": _fixture}
                new_records.append(record)
    indexed = {**existing, **{(r["instance_id"], r["sample_id"]): r for r in new_records}}
    require(set(indexed) == expected, "evaluation missing an actual frozen instance/sample")
    records = [indexed[(i, s)] for i in plan["selection"]["instance_ids"] for s in required_samples]
    result = {"checkpoint": checkpoint["path"], "weights_sha256": checkpoint["weights_sha256"],
              "evaluation_cache_hit": False, "records": records, "sample_ids": required_samples}
    if origin:
        result["source_evidence_id"] = origin
    write_json(output / "evaluations" / f"{alias}.json", result)
    known[alias] = result
    return result


def _start_output(output, *, fixture):
    output = Path(output).resolve()
    require(not output.exists(), "execution outputs are immutable: choose a new job directory")
    if not fixture:
        require(output.is_relative_to(WORK.resolve()), "all experiment output must remain inside selfgrok")
    output.mkdir(parents=True)
    return output


def _solver_factory(adapter, device):
    from hf_backend import Solver
    return Solver(adapter=adapter, device=device, grammar_enabled=True)


def execute_base(plan, target_rows, output, *, device=0, solver_factory=_solver_factory,
                 fingerprint=parameter_fingerprint, _fixture=False):
    plan = checked_plan(plan, allow_baseline=True)
    output = _start_output(output, fixture=_fixture)
    checkpoint = checkpoint_record(plan["base_checkpoint"])
    selected = {r["id"]: r for r in target_rows["selection"]}
    solver = solver_factory(plan["base_checkpoint"], device)
    try:
        assert_fresh_optimizer(solver.optimizer, beta=0, weight_decay=0)
        before = fingerprint(solver)
        known = {}
        request = next(r for r in plan["evaluation_requests"] if r["alias"] == "base")
        evaluate_request(solver, plan, request, checkpoint, selected, output, known, _fixture=_fixture)
        require(fingerprint(solver) == before and solver.optimizer_steps == 0, "base evaluation changed the Solver")
        result = {"schema_version": SCHEMA, "mode": plan["mode"], "plan_sha256": plan.get("plan_sha256"),
                  "baseline_contract_sha256": sha(baseline_contract(plan)),
                  "preproposal_base": plan["schema_version"] == BASELINE_SCHEMA,
                  "fragment_role": "base", "checkpoints": {"base": checkpoint}, "evaluations": known, "branches": {},
                  "execution": {"grammar_enabled": True, "no_training": True, "parameter_fingerprint": before,
                                "actual_generation_tokens": solver.generated_tokens,
                                "sampling_seed_policy": "batch8_chunk_v1", "global_cross_job_rollout_order_claimed": False}}
        write_json(output / "fragment.json", result)
        return result
    finally:
        if hasattr(solver, "close"):
            solver.close()


def execute_branch(plan, target_rows, catalogue, base_fragment, branch_id, output, *, device=0,
                   solver_factory=_solver_factory, fingerprint=parameter_fingerprint, _fixture=False):
    plan = checked_plan(plan)
    require(branch_id in BRANCHES, "choose one of g1/g2/g3/direct")
    validate_base_fragment(plan, base_fragment)
    output = _start_output(output, fixture=_fixture)
    branch = plan["branches"][branch_id]
    train_rows = target_rows["train"]
    require(train_rows, "target train manifest is empty")
    selected = {r["id"]: r for r in target_rows["selection"]}
    stage_by_id = catalogue if isinstance(catalogue, dict) else {s["stage_id"]: s for s in catalogue}
    solver = solver_factory(plan["base_checkpoint"], device)
    journal_path = output / "training_journal.jsonl"
    journal = JsonlJournal(journal_path, f"{plan['round_id']}:{branch_id}")
    known = copy.deepcopy(base_fragment["evaluations"])
    checkpoints = {"base": copy.deepcopy(base_fragment["checkpoints"]["base"])}
    actual_updates = []
    try:
        assert_fresh_optimizer(solver.optimizer, beta=0, weight_decay=0)
        initial_state_entries = len(solver.optimizer.state)
        initial_fingerprint = fingerprint(solver)
        require(initial_fingerprint == base_fragment["execution"]["parameter_fingerprint"], "fresh branch weights differ from actually evaluated base")
        current_fingerprint, current = initial_fingerprint, checkpoints["base"]
        ledgers = []
        rng = random.Random(int(sha([plan["training_seed"], branch_id, "prompt_mixture_v1"])[:15], 16))
        generation_counter = 0
        for phase in branch["phases"]:
            pending = {}
            phase_start_steps = solver.optimizer_steps
            def generate(n, cap, drain):
                nonlocal generation_counter
                target = phase["kind"] == "target_only" or rng.random() < phase["target_probability"]
                pool = train_rows if target else stage_by_id[phase["stage_id"]]["rows"]
                row = pool[rng.randrange(len(pool))]
                seed = int(sha([plan["training_seed"], branch_id, phase["phase_id"], generation_counter, "training_batch_seed_v1"])[:15], 16)
                raw_path = output / "raw_training" / f"{generation_counter:06d}_{phase['phase_id']}.json"
                generation_counter += 1
                before_steps = solver.optimizer_steps
                raw = solver.generate(row, n, cap, seed, output=raw_path, target=target)
                require(solver.optimizer_steps == before_steps, "generation changed optimizer step count")
                require(len(raw["rewards"]) == len(raw["completion_token_ids"]) == len(raw["verifier_completions"]) == n,
                        "backend returned an incomplete training group")
                if not _fixture:
                    require(raw.get("decoding_policy") == "dsl_grammar_v1", "training must use the frozen grammar policy")
                pending.clear()
                pending.update(raw=raw, path=str(raw_path))
                journal.append("generation_source", {"phase": phase["phase_id"], "raw_path": str(raw_path),
                    "raw_sha256": file_sha(raw_path), "actual_batch_seed": seed, "sample_count": n,
                    "optimizer_steps_before": before_steps, "target": target,
                    "native_tokens_include_eos_and_exclude_post_eos_padding": True})
                verifier_id = sha({"ground_truth": row["ground_truth"], "task_kind": "target" if target else "stage"})
                return [Rollout(row["id"], "target" if target else "stage", tuple(ids), int(reward), text, verifier_id)
                        for ids, reward, text in zip(raw["completion_token_ids"], raw["rewards"], raw["verifier_completions"])]

            def update(outputs, advantages):
                metric = solver.update(pending["raw"])
                require(metric["optimizer_step"] is True and metric["parameters_changed"] is True,
                        "mixed binary group did not produce a real Solver update")
                event = {"phase": phase["phase_id"], "raw_path": pending["path"], "metric": metric}
                actual_updates.append(event)
                journal.append("actual_model_update", event)

            ledger = run_phase(PhaseLedger(phase["phase_id"], phase["quota"]), generate=generate,
                               update=update, journal=journal, max_new_tokens=CAP)
            require(solver.optimizer_steps - phase_start_steps == ledger.optimizer_steps, "actual optimizer counter differs from binary group ledger")
            after = fingerprint(solver)
            if after != current_fingerprint:
                saved = solver.save(output / "checkpoints" / phase["checkpoint_alias"])
                current = checkpoint_record(saved)
                current_fingerprint = after
            else:
                require(ledger.optimizer_steps == 0 or any(x["phase"] == phase["phase_id"] for x in actual_updates),
                        "claimed optimizer steps have no actual update evidence")
            checkpoints[phase["checkpoint_alias"]] = copy.deepcopy(current)
            journal.append("checkpoint_boundary", {"phase": phase["phase_id"], "checkpoint_alias": phase["checkpoint_alias"],
                "checkpoint": current, "parameter_fingerprint": current_fingerprint, "optimizer_steps": solver.optimizer_steps})
            ledgers.append(asdict(ledger))
            evaluation_request = next(r for r in plan["evaluation_requests"] if r["checkpoint_alias"] == phase["checkpoint_alias"])
            evaluate_request(solver, plan, evaluation_request, current, selected, output, known, _fixture=_fixture)
            require(fingerprint(solver) == current_fingerprint, "target evaluation changed trained Solver weights")
            write_json(output / f"progress_{phase['segment']}.json", {"phase": phase, "checkpoint": current,
                "training_ledgers": ledgers, "actual_optimizer_steps": solver.optimizer_steps,
                "backend_tokens_all_generation_calls": solver.generated_tokens})
        require(checkpoint_record(plan["base_checkpoint"]) == checkpoints["base"], "original base changed during branch execution")
        own_aliases = {r["alias"] for r in plan["evaluation_requests"] if r["branch"] == branch_id}
        result = {"schema_version": SCHEMA, "mode": plan["mode"], "plan_sha256": plan["plan_sha256"],
            "fragment_role": branch_id, "checkpoints": checkpoints,
            "evaluations": {k: v for k, v in known.items() if k in own_aliases},
            "branches": {branch_id: {"base_checkpoint": plan["base_checkpoint"], "fresh_optimizer": True,
                "optimizer_initial_state_entries": initial_state_entries, "beta": 0, "weight_decay": 0,
                "phases": ledgers, "journal_path": str(journal_path), "journal_sha256": file_sha(journal_path),
                "actual_model_updates": actual_updates}},
            "execution": {"grammar_enabled": True, "actual_optimizer_steps": solver.optimizer_steps,
                "training_generated_tokens": sum(x["generated_tokens"] for x in ledgers),
                "backend_tokens_all_generation_calls": solver.generated_tokens,
                "initial_parameter_fingerprint": initial_fingerprint, "final_parameter_fingerprint": current_fingerprint,
                "sampling_seed_policy": "batch8_chunk_v1", "global_cross_job_rollout_order_claimed": False}}
        write_json(output / "fragment.json", result)
        return result
    finally:
        if hasattr(solver, "close"):
            solver.close()


def merge_fragments(plan, base_fragment, branch_fragments, output, *, _fixture=False):
    plan = checked_plan(plan)
    validate_base_fragment(plan, base_fragment)
    require({f["fragment_role"] for f in branch_fragments} == set(BRANCHES) and len(branch_fragments) == 4,
            "exactly one completed fragment for each branch required")
    exchange = {"schema_version": SCHEMA, "mode": plan["mode"], "plan_sha256": plan["plan_sha256"],
                "checkpoints": {}, "evaluations": {}, "branches": {}}
    for fragment in [base_fragment, *branch_fragments]:
        if fragment["fragment_role"] != "base":
            require(fragment["plan_sha256"] == plan["plan_sha256"] and fragment["mode"] == plan["mode"], "fragment belongs to another frozen round")
        for collection in ("checkpoints", "evaluations", "branches"):
            for name, value in fragment[collection].items():
                require(name not in exchange[collection] or exchange[collection][name] == value, "conflicting shared fragment evidence")
                exchange[collection][name] = value
    result = score_exchange(plan, exchange, _fixture=_fixture)
    output = _start_output(output, fixture=_fixture)
    write_json(output / "exchange.json", exchange)
    for name, value in result.items():
        write_json(output / f"{name}.json", value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("base", "branch", "merge"):
        sub = commands.add_parser(command)
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if command != "merge":
            sub.add_argument("--target-rows", type=Path, required=True)
            sub.add_argument("--device", type=int, default=0)
        if command == "branch":
            sub.add_argument("--catalogue", type=Path, required=True)
            sub.add_argument("--branch", choices=BRANCHES, required=True)
        if command in ("branch", "merge"):
            sub.add_argument("--base-evidence", type=Path, required=True)
        if command == "merge":
            sub.add_argument("--branch-evidence", type=Path, nargs=4, required=True)
    args = parser.parse_args()
    plan = checked_plan(args.plan, allow_baseline=args.command == "base")
    require(plan["mode"] == "real_model_round" or (args.command == "base" and plan["mode"] == "real_model_base_preproposal"),
            "production executor does not run unit fixtures")
    if args.command == "merge":
        result = merge_fragments(plan, load_json(args.base_evidence), [load_json(p) for p in args.branch_evidence], args.output)
        print(json.dumps(result["summary"], indent=2))
    else:
        rows, catalogue = load_inputs(plan, args.target_rows, getattr(args, "catalogue", None))
        if args.command == "base":
            result = execute_base(plan, rows, args.output, device=args.device)
        else:
            result = execute_branch(plan, rows, catalogue, load_json(args.base_evidence), args.branch, args.output, device=args.device)
        print(json.dumps({"status": "completed", "fragment_role": result["fragment_role"], "execution": result["execution"]}, indent=2))


if __name__ == "__main__":
    main()
