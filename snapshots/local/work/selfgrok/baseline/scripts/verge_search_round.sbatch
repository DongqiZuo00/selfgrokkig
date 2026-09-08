#!/bin/bash
#SBATCH --job-name=verge-search-round
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_search_round_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_search_round_%j.err
set -euo pipefail
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_BOOK_SUITE=verge_book_v2
export VERGE_SEARCH_SUITE=verge_search_v1
round="${1:?Expected search round}"
case "$round" in 0|1|2|3|4|5) ;; *) exit 2 ;; esac
stage="${2:?Expected prepare or finish}"
case "$stage" in prepare|finish) ;; *) exit 2 ;; esac
export VERGE_VERSION="verge_search_v1_r0${round}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TVM_FFI_CACHE_DIR="$WORK_ROOT/caches/tvm-ffi"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/verge_search_round.py" "$stage"
if [ "$stage" = finish ]; then
  python "$EXP_ROOT/src/verge_search_controller.py" coordinate --round "$round"
fi
