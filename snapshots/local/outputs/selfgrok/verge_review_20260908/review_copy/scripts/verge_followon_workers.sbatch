#!/bin/bash
#SBATCH --job-name=verge-followon
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_followon_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_followon_%A_%a.err
set -euo pipefail
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_BOOK_SUITE=verge_book_v2
export VERGE_FOLLOWON_SUITE=verge_followon_v1
index="${SLURM_ARRAY_TASK_ID:?Expected candidate array index}"
case "$index" in ''|*[!0-9]*) exit 2 ;; esac
index=$((10#$index))
if (( index < 0 || index >= 72 )); then exit 2; fi
arms=(verge frozen uncertainty outcome)
arm="${arms[$((index % 12 / 3))]}"
printf -v VERGE_VERSION 'verge_followon_v1_%s_r%02d_c%d' "$arm" "$((index / 12))" "$((index % 3 + 1))"
export VERGE_VERSION
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TVM_FFI_CACHE_DIR="$WORK_ROOT/caches/tvm-ffi"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/verge_followon_prepare.py"
python "$EXP_ROOT/src/verge_followon_train.py"
