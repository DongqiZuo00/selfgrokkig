"""Validate the actual names-only plan and all sixteen source/input hashes."""
from run_branch_job import ROOT, resolve_ready, verify_runtime_contract, load_inputs, write_json

if __name__ == "__main__":
    root = ROOT / "runs/round_prepare_named_41417194"
    context = resolve_ready(root)
    contract, report = verify_runtime_contract(ROOT / "configs/round2_runtime_contract.json", root, context["plan"])
    view = ROOT / "benchmarks/generated/naming_round_v2"
    rows, stages = load_inputs(context["plan"], view / "target_rows.json", view / "catalogue.json")
    result = {"status": "passed", "plan_sha256": context["plan"]["plan_sha256"], "source_contract": report,
        "train_rows": len(rows["train"]), "selection_rows": len(rows["selection"]),
        "validated_catalogue_stages": len(stages), "generated_tokens_per_branch": context["plan"]["B"],
        "real_challenger_source": str(context["challenger_source"]), "model_loaded": False}
    write_json(ROOT / "runs/round2_cpu_preflight.json", result)
    print("PASS: actual names-only plan, Challenger, base, sixteen source hashes, all stage specs and target tests")
