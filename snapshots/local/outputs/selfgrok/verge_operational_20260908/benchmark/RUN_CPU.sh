#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
module load python/3.11
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/mpl"
TASK_SELF_GROK="/blue/du.j/jinjiaguo/self grok"
TASK_PYTHON="$TASK_SELF_GROK/envs/uncertainty/bin/python"
TASK_VENDOR="$TASK_SELF_GROK/vendor/rl-grok-recipe/manufactoria"
TASK_STAMP="$(date -u +%Y%m%dT%H%M%S)_$$"
"$TASK_PYTHON" benchmark.py --wrapper "$TASK_VENDOR/hf_file_wrapper.py" --parser "$TASK_VENDOR/verifier/manufactoria_parser.py" 2>&1 | tee "cpu_verifier_${TASK_STAMP}.log"
"$TASK_PYTHON" test_benchmark.py --parser "$TASK_VENDOR/verifier/manufactoria_parser.py" 2>&1 | tee "cpu_tests_${TASK_STAMP}.log"
