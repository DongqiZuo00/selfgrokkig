"""Only unit-test data; never imported by the real plan/score CLI."""
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from round_cli import SCHEMA, build_plan, canonical, file_sha, sha
from generated_budget import JsonlJournal, PhaseLedger, Rollout, run_phase


def request():
    kinds = ("registered", "tape_transform", "registered")
    stages = [{"stage_id": f"s{i+1}", "kind": kind, "spec": {"kind": kind, "unit_fixture": i}}
              for i, kind in enumerate(kinds)]
    proposal = {"base_checkpoint": "SYNTHETIC_UNIT_BASE", "curricula": {
        "g1": stages, "g2": stages[::-1], "g3": [stages[1], stages[2], stages[0]]}}
    contracts = {f"instance-{i}": {"test_ids": ["t1", "t2", "t3", "t4"],
                 "class_by_test": {"t1": "a", "t2": "a", "t3": "b", "t4": "b"},
                 "expected_classes": ["a", "b"]} for i in range(8)}
    return {"round_id": "unit-round", "training_seed": 7, "base_checkpoint": "SYNTHETIC_UNIT_BASE",
            "proposal": proposal, "challenger": {"checkpoint": "SYNTHETIC_UNIT_Q", "context": {"unit_fixture": True},
                "evidence_id": "SYNTHETIC_UNIT_CHALLENGER_LOG", "raw_output": canonical(proposal)},
            "selection": {"manifest_id": "SYNTHETIC_UNIT_SELECTION", "instance_ids": list(contracts),
                          "sample_ids": [f"sample-{i}" for i in range(32)], "test_contracts": contracts},
            "stage_validations": {s["stage_id"]: {"accepted": True,
                "checks": {"executable": True, "nonconstant": True, "no_leakage": True, "hint_regret": "NA"},
                "spec_sha256": sha(s["spec"]), "evidence_id": "SYNTHETIC_UNIT_VALIDATION", "synthetic_fixture": True}
                for s in stages}, "B": 32, "eta": .25, "alpha": .25, "q": 4, "n_min": 2,
            "bootstrap_resamples": 100}


def exchange(directory, plan):
    directory = Path(directory)
    data = {"schema_version": SCHEMA, "mode": "unit_fixture", "plan_sha256": plan["plan_sha256"],
            "checkpoints": {}, "evaluations": {}, "branches": {}}
    for checkpoint in plan["checkpoint_requests"]:
        alias = checkpoint["alias"]
        data["checkpoints"][alias] = {"path": plan["base_checkpoint"] if alias == "base" else f"SYNTHETIC_UNIT_{alias}",
                                       "weights_sha256": sha(alias)}
    for branch_id, branch in plan["branches"].items():
        journal_path = directory / f"{branch_id}.jsonl"
        journal = JsonlJournal(journal_path, f"SYNTHETIC_UNIT_{branch_id}")
        phases = []
        for phase in branch["phases"]:
            def generate(n, cap, drain):
                kind = "stage" if phase["kind"] == "curriculum_stage" else "target"
                return [Rollout("SYNTHETIC_UNIT_PROMPT", kind, (7,) * cap,
                                i % 2 if kind == "stage" else 0,
                                "SYNTHETIC_UNIT_COMPLETION", "SYNTHETIC_UNIT_VERIFIER") for i in range(n)]
            ledger = run_phase(PhaseLedger(phase["phase_id"], phase["quota"]), generate=generate,
                               update=lambda outputs, adv: None, journal=journal, max_new_tokens=1)
            phases.append(asdict(ledger))
        data["branches"][branch_id] = {"base_checkpoint": plan["base_checkpoint"], "fresh_optimizer": True,
            "optimizer_initial_state_entries": 0, "beta": 0, "weight_decay": 0,
            "phases": phases, "journal_path": str(journal_path), "journal_sha256": file_sha(journal_path)}
    global_index = 0
    for evaluation in plan["evaluation_requests"]:
        alias, branch = evaluation["alias"], evaluation["branch"]
        checkpoint = data["checkpoints"][evaluation["checkpoint_alias"]]["path"]
        counts = {"base": 0, "direct": 0, "g1": 16, "g2": 8, "g3": 0}
        success_count = counts[branch] if evaluation["n"] == 32 else min(evaluation["milestone"], counts[branch])
        records = []
        for instance in plan["selection"]["instance_ids"]:
            for index, sample in enumerate(evaluation["sample_ids"]):
                success = index < success_count
                record = {"checkpoint": checkpoint, "training_seed": plan["training_seed"],
                    "manifest_id": plan["selection"]["manifest_id"], "instance_id": instance, "sample_id": sample,
                    "conditions": [True] + [success] * 5, "has_cycle": branch in ("g1", "g3"),
                    "raw_completion": "SYNTHETIC_UNIT_COMPLETION", "completion_token_ids": [7, 8], "hint": None,
                    "generation_evidence_id": f"SYNTHETIC_UNIT_EVENT_{global_index}", "global_rollout_index": global_index,
                    "test_outcomes": dict.fromkeys(["t1", "t2", "t3", "t4"], success), "synthetic_fixture": True}
                global_index += 1
                records.append(record)
        data["evaluations"][alias] = {"checkpoint": checkpoint, "records": records}
    return data
