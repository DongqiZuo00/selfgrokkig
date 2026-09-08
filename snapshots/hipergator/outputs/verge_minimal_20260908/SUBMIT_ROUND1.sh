#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
test -f "$ROOT/runs/round_prepare_41413994/READY.json"
test -f "$ROOT/configs/round1_runtime_contract.json"
test -f "$ROOT/runtime/run_branch_job.py"
test ! -f "$ROOT/runs/round1_array_submission.txt"
cd "$ROOT"
sbatch --parsable --chdir="$ROOT" --array=0-3%2 \
  --output="$ROOT/runs/round1_%A_%a.out" --error="$ROOT/runs/round1_%A_%a.err" \
  "$ROOT/RUN_BRANCH_ARRAY.sbatch" | tee "$ROOT/runs/round1_array_submission.txt"
