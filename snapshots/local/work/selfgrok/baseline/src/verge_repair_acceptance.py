"""One GPU, reduced complete loop. Never launches the scientific-budget round."""
import json
import os
import subprocess
import sys
from pathlib import Path

if os.environ.get("VERGE_VERSION") != "verge_mistral_repair_acceptance":
    raise RuntimeError("Acceptance must use its isolated namespace")

from common import EXP_ROOT, atomic_json, read_json
from verge_round_core import ROOT, VERSION, config


def run(script, index=None):
    env = os.environ.copy()
    if index is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(index)
    subprocess.run([sys.executable, str(EXP_ROOT / "src" / script)], env=env, cwd=EXP_ROOT, check=True)


def main():
    run("verge_round_prepare.py")
    frozen = read_json(ROOT / "round_frozen.json")
    for branch in frozen["branches"]:
        run("verge_round_train.py", branch["index"])
    run("verge_round_finish.py")
    run("verify_verge_repair.py")
    if not read_json(ROOT / "DISPOSABLE_GRADIENT_TEST.json")["passed"]:
        raise RuntimeError("Disposable real-model gradient validation is missing")
    atomic_json(ROOT / "ENGINEERING_ACCEPTANCE.json", {
        "complete": True, "scientific_result": False, "version": VERSION,
        "checks": ["three autonomous typed proposals", "four exact-budget Solver branches",
                   "native-token grouped learner", "schema-matched teacher probabilities",
                   "fresh endpoints and scope-safe selection", "checkpoint and ledger artifacts"],
        "note": "No automatic full-budget experiment submission"})
    print(json.dumps(read_json(ROOT / "ENGINEERING_ACCEPTANCE.json")))


if __name__ == "__main__":
    main()
