"""Record the source/data contract before any matched branch training begins."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "runtime/hf_backend.py", "runtime/execute_round.py",
    "decoding/dsl_grammar.py", "decoding/grammar_service.py", "decoding/hf_grammar_bridge.py",
    "round/round_cli.py", "benchmarks/generated/catalogue.json",
    "benchmarks/generated/pilot_view_v1/target_rows.json",
    "benchmarks/generated/pilot_view_v1/target_view_metadata.json",
    "benchmarks/generated/pilot_view_v1/stage_validation_evidence.json",
)


def freeze():
    data = {"version": "verge_minimal_round1_source_contract_v1", "files": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
        "resources": {"per_job": {"b200": 1, "memory_gib": 64, "time_limit_minutes": 60},
                      "maximum_concurrent_new_jobs": 2, "user_ceiling_b200": 4, "user_ceiling_memory_gib": 192},
        "prepare_root": "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/runs/round_prepare_41413994",
        "preserve_original_checkpoint_and_other_jobs": True,
        "training_generated_token_budget": {"per_branch": 65536, "four_branch_total": 262144},
        "source_bundle": "work/selfgrok/verge_minimal_round_bundle.tar.gz"}
    target = ROOT / "configs/round1_runtime_contract.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise ValueError("Frozen round runtime changed; do not overwrite its evidence")
    if not target.exists():
        target.write_text(encoded, encoding="utf-8")
    print(str(target))


if __name__ == "__main__":
    freeze()
