#!/bin/bash
#SBATCH --job-name=self-grok-fast-pair
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/uncertainty_target_gain/logs/full_%j.out
#SBATCH --error=experiments/uncertainty_target_gain/logs/full_%j.err

set -euo pipefail

WORK_ROOT="/blue/du.j/jinjiaguo/self grok"
EXP_ROOT="$WORK_ROOT/experiments/uncertainty_target_gain"

cd "$WORK_ROOT"
module purge
module load python/3.11
module load cuda/12.8.1
source "$WORK_ROOT/envs/uncertainty/bin/activate"

export HF_HOME="$WORK_ROOT/caches/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/caches/huggingface/hub"
export HF_DATASETS_CACHE="$WORK_ROOT/caches/huggingface/datasets"
export TORCH_HOME="$WORK_ROOT/caches/torch"
export TRITON_CACHE_DIR="$WORK_ROOT/caches/triton"
export XDG_CACHE_HOME="$WORK_ROOT/caches/xdg"
export MPLCONFIGDIR="$WORK_ROOT/caches/matplotlib"
export VLLM_CACHE_ROOT="$WORK_ROOT/caches/vllm"
export FLASHINFER_WORKSPACE_BASE="$WORK_ROOT/caches/flashinfer_workspace"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/caches/torchinductor"
export TMPDIR="$WORK_ROOT/tmp"
export PIP_CACHE_DIR="$WORK_ROOT/caches/pip"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EXP_ROOT/logs" "$WORK_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
python "$EXP_ROOT/src/fast_orchestrate.py"

# Reaching this line means the fixed-budget fast protocol has either confirmed
# the frozen pair or produced an honest null/non-replication result.
echo "Fast matched-pair experiment completed at $(date --iso-8601=seconds)"
