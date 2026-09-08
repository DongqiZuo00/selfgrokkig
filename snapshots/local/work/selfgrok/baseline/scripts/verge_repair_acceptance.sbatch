#!/bin/bash
#SBATCH --job-name=verge-v2-accept
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_v2_accept_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_v2_accept_%j.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_VERSION=verge_mistral_repair_acceptance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TVM_FFI_CACHE_DIR="$WORK_ROOT/caches/tvm-ffi"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/verge_repair_acceptance.py"
