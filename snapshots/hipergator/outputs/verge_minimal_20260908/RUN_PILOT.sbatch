#!/bin/bash
#SBATCH --job-name=verge-minimal
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
WORK="/blue/du.j/jinjiaguo/self grok"
module purge
module load python/3.11 cuda/12.8.1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/caches/huggingface" TORCH_HOME="$WORK/caches/torch"
export XDG_CACHE_HOME="$ROOT/cache" MPLCONFIGDIR="$ROOT/cache/mpl" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TMPDIR="$ROOT/tmp/pilot_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$ROOT"
"$WORK/envs/uncertainty/bin/python" -u "${1:?python entry required}" --output "$ROOT/runs/${2:?run label}_${SLURM_JOB_ID}"
