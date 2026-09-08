"""New reviewed contract for the names-only round; the first contract is retained."""
import json
from hf_backend import ROOT, sha256

if __name__ == "__main__":
    prior = json.loads((ROOT / "configs/round1_runtime_contract.json").read_text(encoding="utf-8"))
    for name, expected in prior["files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"First-round source was changed: {name}")
    data = dict(prior)
    data["version"] = "verge_minimal_round2_names_only_source_contract_v1"
    data["prepare_root"] = "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/runs/round_prepare_named_41417194"
    data["source_bundle"] = "work/selfgrok/verge_named_prepare_bundle.tar.gz"
    data["files"] = dict(prior["files"])
    new_files = ["runtime/prepare_named_round.py", "runtime/run_branch_job.py"] + [
        f"benchmarks/generated/naming_round_v2/{name}" for name in
        ("target_rows.json", "target_view_metadata.json", "catalogue.json", "stage_validation_evidence.json")]
    data["files"].update({name: sha256(ROOT / name) for name in new_files})
    data["decoding_policy_unchanged"] = "dsl_grammar_v1"
    data["numeric_name_grammar_adopted"] = False
    data["changes"] = ["generic canonical node naming added to all Solver prompts",
                       "actual prior failed round supplied as explicitly old-view diagnostic Challenger context",
                       "Challenger instruction states the existing constant-reward skip rule"]
    target = ROOT / "configs/round2_runtime_contract.json"
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise ValueError("Do not overwrite the frozen second-round source contract")
    if not target.exists():
        target.write_text(encoded, encoding="utf-8")
    print(target)
