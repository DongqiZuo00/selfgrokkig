#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
PREPARATION="${1:?explicit actual completed preparation required}"
case "$PREPARATION" in "$ROOT"/runs/round_prepare_ignition_*) ;; *) exit 2 ;; esac
test -f "$PREPARATION/READY.json"
test -f "$ROOT/configs/round3_runtime_contract.json"
test -f "$ROOT/runs/round3_cpu_preflight.json"
test ! -f "$ROOT/runs/round3_array_submission.txt"
cd "$ROOT"
sbatch --parsable --chdir="$ROOT" --array=0-3%2 \
  --output="$ROOT/runs/round3_%A_%a.out" --error="$ROOT/runs/round3_%A_%a.err" \
  "$ROOT/RUN_IGNITION_BRANCH_ARRAY.sbatch" "$PREPARATION" | tee "$ROOT/runs/round3_array_submission.txt"
