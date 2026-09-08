"""Read-only live plan/source/data validation before allocating branch GPUs."""
from run_branch_job import ROOT, resolve_ready, verify_runtime_contract, load_inputs, write_json

if __name__ == "__main__":
    root = ROOT / "runs/round_prepare_41413994"
    context = resolve_ready(root)
    contract, report = verify_runtime_contract(ROOT / "configs/round1_runtime_contract.json", root, context["plan"])
    rows, stages = load_inputs(context["plan"], ROOT / "benchmarks/generated/pilot_view_v1/target_rows.json",
                              ROOT / "benchmarks/generated/catalogue.json")
    result = {"status": "passed", "plan_sha256": context["plan"]["plan_sha256"],
        "source_contract": report, "train_rows": len(rows["train"]), "selection_rows": len(rows["selection"]),
        "validated_catalogue_stages": len(stages), "generated_tokens_per_branch": context["plan"]["B"],
        "real_challenger_source": str(context["challenger_source"]), "model_loaded": False}
    write_json(ROOT / "runs/round1_cpu_preflight.json", result)
    print("PASS: live plan, real Challenger, shared base, all ten source hashes, stage specs and target tests")
