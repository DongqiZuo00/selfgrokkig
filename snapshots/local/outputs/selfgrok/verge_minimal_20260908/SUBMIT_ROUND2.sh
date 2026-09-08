#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
test -f "$ROOT/runs/round_prepare_named_41417194/READY.json"
test -f "$ROOT/configs/round2_runtime_contract.json"
test -f "$ROOT/runs/round2_cpu_preflight.json"
test ! -f "$ROOT/runs/round2_array_submission.txt"
cd "$ROOT"
sbatch --parsable --chdir="$ROOT" --array=0-3%2 \
  --output="$ROOT/runs/round2_%A_%a.out" --error="$ROOT/runs/round2_%A_%a.err" \
  "$ROOT/RUN_NAMED_BRANCH_ARRAY.sbatch" | tee "$ROOT/runs/round2_array_submission.txt"
