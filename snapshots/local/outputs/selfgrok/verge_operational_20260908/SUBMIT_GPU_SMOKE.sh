#!/bin/bash
# Single bounded pipeline smoke. This is not a Gate 0-3 suite launcher.
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_operational_20260908"
cd "$ROOT"
mkdir -p runs
if test -f runs/gpu_smoke_submission.txt; then
  echo "A smoke submission is already recorded; inspect it before an explicit rerun." >&2
  exit 1
fi
bash -n RUN_GPU_SMOKE.sbatch
test -f runtime/solver_update.py
test -f benchmark/generated/target_train.jsonl
sbatch --parsable --chdir="$ROOT" \
  --output="$ROOT/runs/gpu_smoke_%j.out" \
  --error="$ROOT/runs/gpu_smoke_%j.err" RUN_GPU_SMOKE.sbatch | tee runs/gpu_smoke_submission.txt
