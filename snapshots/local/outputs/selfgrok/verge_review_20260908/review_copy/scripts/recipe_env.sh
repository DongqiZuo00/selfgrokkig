#!/bin/bash
set -euo pipefail

WORK_ROOT="/blue/du.j/jinjiaguo/self grok"
EXP_ROOT="$WORK_ROOT/experiments/has_transfer_witness"

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
export PIP_CACHE_DIR="$WORK_ROOT/caches/pip"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="$WORK_ROOT/tmp/r${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$EXP_ROOT/logs" "$TMPDIR"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
mkdir -p "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR"
