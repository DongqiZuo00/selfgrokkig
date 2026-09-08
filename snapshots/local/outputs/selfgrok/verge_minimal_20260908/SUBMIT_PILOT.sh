#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
ENTRY="${1:?entry required}"
LABEL="${2:?new run label required}"
[[ "$LABEL" =~ ^[a-z0-9_]+$ ]]
test -f "$ROOT/$ENTRY"
test ! -f "$ROOT/runs/${LABEL}_submission.txt"
mkdir -p "$ROOT/runs"
cd "$ROOT"
sbatch --parsable --chdir="$ROOT" \
  --output="$ROOT/runs/${LABEL}_%j.out" --error="$ROOT/runs/${LABEL}_%j.err" \
  RUN_PILOT.sbatch "$ENTRY" "$LABEL" | tee "$ROOT/runs/${LABEL}_submission.txt"
