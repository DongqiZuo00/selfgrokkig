#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
test -f "$ROOT/benchmarks/generated/naming_prompt_v2/screen_tasks.json"
test ! -f "$ROOT/runs/naming_v2_submission.txt"
cd "$ROOT"
sbatch --parsable --chdir="$ROOT" --dependency=afterany:41414619 \
  --output="$ROOT/runs/naming_v2_%j.out" --error="$ROOT/runs/naming_v2_%j.err" \
  "$ROOT/RUN_PILOT.sbatch" runtime/probe_prompt_v2.py naming_v2 | tee "$ROOT/runs/naming_v2_submission.txt"
