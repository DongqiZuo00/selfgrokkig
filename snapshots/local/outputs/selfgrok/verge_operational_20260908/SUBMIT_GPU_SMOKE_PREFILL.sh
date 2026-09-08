#!/bin/bash
# Correct the observed format-interface defect within the original 20 GPU-min ceiling.
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_operational_20260908"
cd "$ROOT"
if test -f runs/gpu_smoke_prefill_submission.txt; then
  echo "A prefill smoke is already submitted; inspect the recorded job before rerunning." >&2
  exit 1
fi
PRIOR_SECONDS="$(sacct -j 41369486 -X -n -P --format=ElapsedRaw | head -n 1)"
case "$PRIOR_SECONDS" in ''|*[!0-9]*) echo "Cannot verify the prior smoke GPU cost" >&2; exit 1 ;; esac
if (( PRIOR_SECONDS + 17 * 60 > 20 * 60 )); then
  echo "The bounded format rerun would exceed the combined 20 GPU-minute ceiling" >&2
  exit 1
fi
bash -n RUN_GPU_SMOKE_PREFILL.sbatch
sbatch --parsable --chdir="$ROOT" \
  --output="$ROOT/runs/gpu_smoke_prefill_%j.out" \
  --error="$ROOT/runs/gpu_smoke_prefill_%j.err" RUN_GPU_SMOKE_PREFILL.sbatch | tee runs/gpu_smoke_prefill_submission.txt
