#!/bin/bash
#SBATCH --job-name=verge-control
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_control_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_control_%A_%a.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_BOOK_SUITE=verge_book_v2
export VERGE_CONTROL_SUITE=verge_control_v1
round="${1:?Expected control round}"
case "$round" in 0|1|2|3|4|5) ;; *) exit 2 ;; esac
case "${SLURM_ARRAY_TASK_ID:?Expected array index}" in
  0) label=binary ;;
  1) label=dense ;;
  2) label=dense_then_binary ;;
  *) exit 2 ;;
esac
export VERGE_VERSION="verge_control_v1_${label}_r0${round}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TVM_FFI_CACHE_DIR="$WORK_ROOT/caches/tvm-ffi"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/verge_control_prepare.py"
python "$EXP_ROOT/src/verge_control_train.py"
