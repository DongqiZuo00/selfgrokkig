"""Explicit full-round submission; dry-run by default, no automatic budget escalation."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    exp = Path(__file__).resolve().parents[1]
    acceptance = exp / "raw_results/verge_mistral_repair_acceptance/ENGINEERING_ACCEPTANCE.json"
    if not acceptance.exists() or not json.loads(acceptance.read_text()).get("complete"):
        raise RuntimeError("Full round requires completed engineering acceptance")
    if not args.submit:
        print(json.dumps({"submit": False, "version": "verge_mistral_repair_v2",
            "branch_tokens": 524288, "branches": 4, "max_concurrent": 2,
            "memory_gb_per_worker": 32, "gpu_per_worker": "1 B200", "scope_tolerance": 0.0}))
        return
    from common import atomic_json, read_json
    root = exp / "raw_results/verge_mistral_repair_v2"
    submission = root / "submission.json"
    saved = read_json(submission) if submission.exists() else {}
    script = exp / "scripts/verge_repair_stage.sbatch"
    def enqueue(stage, options):
        response = subprocess.run(["sbatch", "--parsable"] + options + [str(script), stage],
                                  cwd=exp.parents[1], text=True, capture_output=True, check=True)
        return response.stdout.strip().split(";")[0]
    if "prepare_job" not in saved:
        saved["prepare_job"] = enqueue("prepare", ["--job-name=verge-v2-prepare"])
        atomic_json(submission, saved)
    if "branch_array" not in saved:
        saved["branch_array"] = enqueue("train", ["--job-name=verge-v2-branch", "--array=0-3%2",
            "--kill-on-invalid-dep=yes", f"--dependency=afterok:{saved['prepare_job']}"])
        atomic_json(submission, saved)
    if "finish_job" not in saved:
        saved["finish_job"] = enqueue("finish", ["--job-name=verge-v2-finish", "--kill-on-invalid-dep=yes",
            f"--dependency=afterok:{saved['branch_array']}"])
        atomic_json(submission, saved)
    print(json.dumps(saved))


if __name__ == "__main__":
    main()
