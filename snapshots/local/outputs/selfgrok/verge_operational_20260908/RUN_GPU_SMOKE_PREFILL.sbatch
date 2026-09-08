#!/bin/bash
#SBATCH --job-name=verge-prefill-smoke
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=00:17:00
set -euo pipefail
# Failure-driven format repair after job 41369486, retained at its original path.
# Submit with --output "$ROOT/runs/gpu_smoke_prefill_%j.out" --error
# "$ROOT/runs/gpu_smoke_prefill_%j.err" after creating ROOT/runs. Slurm opens these
# before this script runs, so output placement is part of the submission command.
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_operational_20260908"
WORK="/blue/du.j/jinjiaguo/self grok"
module purge
module load python/3.11 cuda/12.8.1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/caches/huggingface" TORCH_HOME="$WORK/caches/torch"
export XDG_CACHE_HOME="$ROOT/cache" MPLCONFIGDIR="$ROOT/cache/mpl" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TMPDIR="$ROOT/tmp/smoke_prefill_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$ROOT/runtime"
"$WORK/envs/uncertainty/bin/python" -u live_smoke_prefill.py --output "$ROOT/runs/gpu_smoke_prefill_${SLURM_JOB_ID}"
