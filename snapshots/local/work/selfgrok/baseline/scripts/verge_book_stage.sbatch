#!/bin/bash
#SBATCH --job-name=verge-book
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_book_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_book_%A_%a.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_VERSION="${1:?Expected frozen round namespace}"
export VERGE_BOOK_SUITE="${VERGE_BOOK_SUITE:-verge_book_v1}"
case "$VERGE_BOOK_SUITE" in
  verge_book_v1|verge_book_v2) ;;
  *) echo "Unknown book suite" >&2; exit 2 ;;
esac
case "$VERGE_VERSION" in
  "${VERGE_BOOK_SUITE}_"*) ;;
  *) echo "Round and book suite disagree" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TVM_FFI_CACHE_DIR="$WORK_ROOT/caches/tvm-ffi"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
case "${2:-}" in
  prepare) python "$EXP_ROOT/src/verge_round_prepare.py" ;;
  train) python "$EXP_ROOT/src/verge_round_train.py" ;;
  finish)
    python "$EXP_ROOT/src/verge_round_finish.py"
    python "$EXP_ROOT/src/verge_book_controller.py" dispatch --after "$SLURM_JOB_ID"
    ;;
  *) echo "Expected prepare, train or finish" >&2; exit 2 ;;
esac
