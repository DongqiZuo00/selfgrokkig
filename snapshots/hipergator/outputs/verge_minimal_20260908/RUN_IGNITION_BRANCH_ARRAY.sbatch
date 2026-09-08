#!/bin/bash
#SBATCH --job-name=verge-mini-round3
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
PREPARATION="${1:?explicit completed preparation required}"
BRANCHES=(g1 g2 g3 direct)
BRANCH="${BRANCHES[${SLURM_ARRAY_TASK_ID:?array task required}]}"
module purge
module load python/3.11 cuda/12.8.1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/caches/huggingface" TORCH_HOME="$WORK/caches/torch"
export XDG_CACHE_HOME="$ROOT/cache" MPLCONFIGDIR="$ROOT/cache/mpl" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TMPDIR="$ROOT/tmp/branch_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$ROOT"
"$WORK/envs/uncertainty/bin/python" -u runtime/run_branch_job.py \
  --plan-root "$PREPARATION" --branch "$BRANCH" \
  --runtime-contract "$ROOT/configs/round3_runtime_contract.json" \
  --output "$ROOT/runs/round3_${BRANCH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
