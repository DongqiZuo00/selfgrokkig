#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="/blue/du.j/jinjiaguo/self grok"
module load python/3.11
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/runtime:$ROOT/protocol:$ROOT/stages:$ROOT/benchmark:$WORK/vendor/rl-grok-recipe:$WORK/vendor/rl-grok-recipe/manufactoria"
export MPLCONFIGDIR="$ROOT/cache/mpl"
PYTHON="$WORK/envs/uncertainty/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
LOGS="$ROOT/runs/cpu_$STAMP"
mkdir -p "$LOGS" "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -B -m unittest discover -s protocol -p 'test_*.py' -v 2>&1 | tee "$LOGS/protocol.log"
"$PYTHON" -B -m unittest discover -s stages -p 'test_*.py' -v 2>&1 | tee "$LOGS/stages.log"
"$PYTHON" -B -m unittest discover -s runtime -p 'test_*.py' -v 2>&1 | tee "$LOGS/runtime.log"
"$PYTHON" -B benchmark/test_benchmark.py --parser "$WORK/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py" 2>&1 | tee "$LOGS/benchmark.log"
"$PYTHON" -B integration_round.py --output "$LOGS/integration" 2>&1 | tee "$LOGS/integration.log"
echo "All CPU suites and the synthetic integrated round completed: $LOGS"
