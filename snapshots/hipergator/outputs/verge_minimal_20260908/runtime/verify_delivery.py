"""Verify the final report/evidence/checkpoint mirror without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["METHOD_REVIEW.md", "IMPLEMENTATION_AUDIT.md", "NEXT_ACTION.md", "EXPERIMENT_RESULT.json",
         "evidence/COMMANDS.md", "evidence/FINAL_ACCEPTANCE.log", "evidence/COST_THREE_ROUNDS.json",
         "evidence/COST_THREE_ROUNDS.md", "evidence/ROUND3_UPDATE_REPLAY.md",
         "evidence/MINIMAL_COMPLETION_CRITERIA.md",
         *[f"evidence/round{i}_independent_score_audit.json" for i in (1, 2, 3)],
         *[f"configs/round{i}_runtime_contract.json" for i in (1, 2, 3)],
         "runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/adapter_model.safetensors"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "DELIVERY_MANIFEST.json"
    if args.freeze:
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump({"scope": "final reports, evidence, contracts and trained checkpoint mirror",
                       "files": {name: digest(ROOT / name) for name in FILES}}, handle, indent=2)
            handle.write("\n")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for name, expected in data["files"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or digest(source) != expected:
            raise ValueError(f"Delivery mismatch: {name}")
    print(f"PASS: {len(data['files'])} final files, including trained LoRA weights, exactly match the delivery manifest")
