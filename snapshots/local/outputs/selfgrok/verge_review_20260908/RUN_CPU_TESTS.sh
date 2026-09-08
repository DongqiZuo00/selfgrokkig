#!/bin/bash
# CPU audit only. No Slurm submission, model load, network sampling or sealed test.
set -euo pipefail
AUDIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="/blue/du.j/jinjiaguo/self grok"
module load python/3.11
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WORK_ROOT/vendor/rl-grok-recipe:$WORK_ROOT/vendor/rl-grok-recipe/manufactoria"
export MPLCONFIGDIR="$AUDIT_ROOT/mpl"
mkdir -p "$AUDIT_ROOT/evidence"
stamp="$(date -u +%Y%m%dT%H%M%S)_$$"
cd "$AUDIT_ROOT/review_copy/src"
"$WORK_ROOT/envs/uncertainty/bin/python" -m unittest -v \
 test_verge_fresh_review test_verge_round test_verge_runtime test_verge_repair \
 test_verge_repair_integration test_verge_search_protocol test_verge_book \
 test_verge_independent_sampling 2>&1 | tee "$AUDIT_ROOT/evidence/retest_${stamp}.log"
"$WORK_ROOT/envs/uncertainty/bin/python" "$AUDIT_ROOT/evidence/spec_fixtures.py" \
 2>&1 | tee "$AUDIT_ROOT/evidence/spec_${stamp}.log"

