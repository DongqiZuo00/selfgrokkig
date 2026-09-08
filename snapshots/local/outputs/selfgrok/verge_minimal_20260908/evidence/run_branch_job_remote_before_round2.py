"""Slurm-compatible entry for exactly one READY, frozen VERGE branch.

RUN_PILOT passes only --output. Set VERGE_PLAN_ROOT to the explicit completed
round_prepare_<job> directory and VERGE_BRANCH to g1/g2/g3/direct, or provide
--plan-root and --branch. This file does not submit or cancel any Slurm job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from execute_round import (ROOT, WORK, BRANCHES, checked_plan, execute_branch, file_sha,
                           load_inputs, load_json, require, validate_base_fragment, write_json)


def verify_runtime_contract(path, preparation_root, plan):
    path = Path(path).resolve()
    contract = load_json(path)
    require(contract.get("version") and contract.get("files"), "runtime contract must identify its version and frozen files")
    require(Path(contract["prepare_root"]).resolve() == Path(preparation_root).resolve(), "runtime contract names a different real preparation")
    required = {"runtime/hf_backend.py", "runtime/execute_round.py", "round/round_cli.py",
                "decoding/dsl_grammar.py", "decoding/grammar_service.py", "decoding/hf_grammar_bridge.py",
                "benchmarks/generated/catalogue.json", "benchmarks/generated/pilot_view_v1/target_rows.json",
                "benchmarks/generated/pilot_view_v1/target_view_metadata.json",
                "benchmarks/generated/pilot_view_v1/stage_validation_evidence.json"}
    require(required <= set(contract["files"]), "runtime contract omits a required backend/decoder/protocol/input file")
    checks = {}
    for relative, expected in contract["files"].items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT.resolve()), "runtime contract file escapes the independent release")
        actual = file_sha(source)
        require(actual == expected, f"frozen runtime file changed: {relative}; require an explicit new reviewed contract version")
        checks[relative] = actual
    budget = contract["training_generated_token_budget"]
    require(budget["per_branch"] == plan["B"] and budget["four_branch_total"] == 4 * plan["B"],
            "runtime token budget differs from frozen round plan")
    resources = contract["resources"]
    require(resources["per_job"] == {"b200": 1, "memory_gib": 64, "time_limit_minutes": 60}
            and resources["maximum_concurrent_new_jobs"] == 2,
            "this wrapper is scoped to one B200/64GiB/1h per job and at most two newly scheduled jobs")
    require(resources["user_ceiling_b200"] <= 4 and resources["user_ceiling_memory_gib"] <= 192,
            "runtime resource contract exceeds the user ceiling")
    return contract, {"path": str(path), "sha256": file_sha(path), "version": contract["version"], "verified_files": checks}


def resolve_ready(plan_root, *, _fixture=False):
    directory = Path(plan_root).resolve()
    if not _fixture:
        require(directory.is_relative_to((ROOT / "runs").resolve()) and directory.name.startswith("round_prepare_"),
                "explicit VERGE_PLAN_ROOT must name this release's completed runs/round_prepare_<job> directory")
    ready_path = directory / "READY.json"
    require(ready_path.is_file(), "preparation has no READY.json; do not start a branch")
    ready = load_json(ready_path)
    require(ready.get("status") == "real_base_and_challenger_complete", "preparation is not READY with real base and Challenger evidence")
    plan_path = Path(ready["plan"]).resolve()
    base_path = Path(ready["base_fragment"]).resolve()
    require(plan_path == directory / "plan.json" and base_path == directory / "base/fragment.json",
            "READY paths must point to this exact preparation directory")
    plan = checked_plan(plan_path)
    if not _fixture:
        require(plan["mode"] == "real_model_round", "a real branch cannot use a fixture plan")
    require(ready["plan_sha256"] == plan["plan_sha256"], "READY plan digest differs from the frozen plan")
    expected_curricula = {key: [stage["stage_id"] for stage in stages] for key, stages in plan["proposal"]["curricula"].items()}
    require(ready.get("curricula") == expected_curricula, "READY curricula summary differs from the actual Challenger proposal")
    base = load_json(base_path)
    validate_base_fragment(plan, base)
    if ready.get("baseline_contract_sha256"):
        require(ready["baseline_contract_sha256"] == base["baseline_contract_sha256"], "READY base sampling contract differs")
    source = Path(plan["challenger"]["evidence_id"]).resolve()
    require(source.is_relative_to(directory) and source.is_file(), "Challenger raw evidence must belong to the completed preparation")
    evidence = load_json(source)
    require(evidence.get("raw_output") == plan["challenger"]["raw_output"] and evidence.get("checkpoint") == plan["challenger"]["checkpoint"],
            "frozen plan no longer matches its actual Challenger generation")
    if not _fixture:
        require(evidence.get("mode") == "real_independent_challenger_generation", "Challenger source is not a real generation record")
    return {"root": directory, "ready_path": ready_path, "ready": ready, "plan_path": plan_path,
            "base_path": base_path, "plan": plan, "base": base, "challenger_source": source}


def run_job(output, *, plan_root=None, branch_id=None, device=0, target_rows=None, catalogue=None,
            runtime_contract=None, _fixture=False, executor=execute_branch):
    plan_root = plan_root or os.environ.get("VERGE_PLAN_ROOT")
    branch_id = branch_id or os.environ.get("VERGE_BRANCH")
    require(plan_root, "set explicit VERGE_PLAN_ROOT or --plan-root; never infer the latest preparation")
    require(branch_id in BRANCHES, "set VERGE_BRANCH or --branch to g1/g2/g3/direct")
    context = resolve_ready(plan_root, _fixture=_fixture)
    ready = context["ready"]
    if runtime_contract is not None or not _fixture:
        contract, contract_report = verify_runtime_contract(runtime_contract or ROOT / "configs/round1_runtime_contract.json",
                                                           context["root"], context["plan"])
    else:
        contract, contract_report = None, {"mode": "explicit_unit_fixture_no_runtime_contract"}
    target_path = Path(target_rows or ready.get("target_rows") or ROOT / "benchmarks/generated/pilot_view_v1/target_rows.json").resolve()
    catalogue_path = Path(catalogue or ready.get("catalogue") or ROOT / "benchmarks/generated/catalogue.json").resolve()
    if not _fixture:
        require(target_path.is_relative_to(ROOT.resolve()) and catalogue_path.is_relative_to(ROOT.resolve()),
                "target/catalogue inputs must remain inside this independent release")
    if ready.get("target_rows_sha256"):
        require(file_sha(target_path) == ready["target_rows_sha256"], "target view changed after real Challenger preparation")
    if ready.get("catalogue_sha256"):
        require(file_sha(catalogue_path) == ready["catalogue_sha256"], "catalogue wording/rows changed after real Challenger preparation")
    rows, stage_map = load_inputs(context["plan"], target_path, catalogue_path, _fixture=_fixture)
    output = Path(output).resolve()
    require(not output.exists(), "branch job output must be a new independent directory")
    if not _fixture:
        require(output.is_relative_to((ROOT / "runs").resolve()), "branch job output must stay under this release runs directory")
    output.mkdir(parents=True)
    launch = {"status": "validated_ready_before_model_load", "branch": branch_id,
              "preparation_root": str(context["root"]), "plan_sha256": context["plan"]["plan_sha256"],
              "plan_file_sha256": file_sha(context["plan_path"]), "ready_file_sha256": file_sha(context["ready_path"]),
              "base_fragment": str(context["base_path"]), "base_fragment_sha256": file_sha(context["base_path"]),
              "challenger_evidence": str(context["challenger_source"]), "challenger_evidence_sha256": file_sha(context["challenger_source"]),
              "target_rows": str(target_path), "target_rows_sha256": file_sha(target_path),
              "catalogue": str(catalogue_path), "catalogue_sha256": file_sha(catalogue_path),
              "executor_sha256": file_sha(Path(__file__).with_name("execute_round.py")),
              "runtime_contract": contract_report,
              "branch_generated_token_budget": context["plan"]["branches"][branch_id]["generated_token_budget"],
              "device_index_within_allocation": device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "started_unix": time.time(), "synthetic_fixture": _fixture}
    write_json(output / "LAUNCH.json", launch)
    if contract is not None:
        write_json(output / "runtime_contract.json", contract)
    try:
        fragment = executor(context["plan"], rows, stage_map, context["base"], branch_id,
                            output / "execution", device=device, _fixture=_fixture)
        write_json(output / "fragment.json", fragment)
        write_json(output / "COMPLETE.json", {"status": "complete", "branch": branch_id,
            "plan_sha256": context["plan"]["plan_sha256"], "fragment": str(output / "fragment.json"),
            "fragment_sha256": file_sha(output / "fragment.json"), "elapsed_seconds": time.time() - launch["started_unix"],
            "execution": fragment["execution"]})
        return fragment
    except Exception as error:
        write_json(output / "FAILED.json", {"status": "failed", "branch": branch_id, "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.time() - launch["started_unix"], "partial_evidence_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-rows", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--runtime-contract", type=Path)
    args = parser.parse_args()
    result = run_job(args.output, plan_root=args.plan_root, branch_id=args.branch, device=args.device,
                     target_rows=args.target_rows, catalogue=args.catalogue, runtime_contract=args.runtime_contract)
    print(json.dumps({"status": "complete", "branch": result["fragment_role"], "execution": result["execution"]}, indent=2))


if __name__ == "__main__":
    main()
