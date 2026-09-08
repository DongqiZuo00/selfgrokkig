"""Freeze and validate one actual ignition preparation. No GPU work or submission."""
import argparse
import json
from pathlib import Path
from run_branch_job import ROOT, resolve_ready, verify_runtime_contract, load_inputs
from hf_backend import sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.preparation.resolve()
    context = resolve_ready(prepared)
    ready = context["ready"]
    if ready.get("proposal_policy", {}).get("version") != "verge_pilot_g1_first_stage_observed_mixed_prior_v1":
        raise ValueError("This launch is only for the reviewed actual ignition policy")
    if ready["curricula"]["g1"][0] not in {"identity", "append_R"}:
        raise ValueError("Real Challenger output is inconsistent with its declared policy")
    prior_path = ROOT / "configs/round2_runtime_contract.json"
    contract = json.loads(prior_path.read_text(encoding="utf-8"))
    for name, expected in contract["files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Previously reviewed source changed: {name}")
    contract.update(version="verge_minimal_round3_ignition_source_contract_v1", prepare_root=str(prepared),
                    source_bundle="work/selfgrok/verge_ignition_round_bundle.tar.gz",
                    proposal_policy=ready["proposal_policy"], prior_contract_sha256=sha256(prior_path),
                    baseline_reuse=ready["baseline_reuse"])
    for name in ("runtime/prepare_ignition_round.py", "runtime/preflight_ignition.py",
                 "RUN_IGNITION_BRANCH_ARRAY.sbatch", "SUBMIT_ROUND3.sh"):
        contract["files"][name] = sha256(ROOT / name)
    contract["changes"] = ["g1 first stage constrained to two observed mixed stages; actual Q chooses all positions",
                           "same-view base P32 reused with original raw paths and zero new evaluation tokens",
                           "completed real second-round history in Challenger context"]
    path = ROOT / "configs/round3_runtime_contract.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    _, report = verify_runtime_contract(path, prepared, context["plan"])
    rows, stages = load_inputs(context["plan"], Path(ready["target_rows"]), Path(ready["catalogue"]))
    result = {"status": "passed", "preparation": str(prepared), "plan_sha256": context["plan"]["plan_sha256"],
              "contract": report, "actual_challenger_source": str(context["challenger_source"]),
              "train_rows": len(rows["train"]), "selection_rows": len(rows["selection"]),
              "catalogue_stages": len(stages), "new_baseline_tokens": 0, "B_per_branch": context["plan"]["B"],
              "model_loaded": False}
    with (ROOT / "runs/round3_cpu_preflight.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
