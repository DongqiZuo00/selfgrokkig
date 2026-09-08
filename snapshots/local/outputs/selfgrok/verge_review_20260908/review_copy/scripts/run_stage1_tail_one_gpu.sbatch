#!/bin/bash
#SBATCH --job-name=self-grok-stage1-tail
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --output=experiments/uncertainty_target_gain/logs/stage1_tail_%j.out
#SBATCH --error=experiments/uncertainty_target_gain/logs/stage1_tail_%j.err

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
export SELF_GROK_GPU_COUNT=1
export SELF_GROK_STOP_AFTER_DISCOVERY=1

mkdir -p "$EXP_ROOT/logs" "$WORK_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export SELF_GROK_VLLM_BASE_PORT=$((8100 + SLURM_ARRAY_TASK_ID))
    case "${SELF_GROK_ARRAY_MODE:-}" in
        batch2)
            BATCH2_BRANCHES=(
                fast__discovery__candidate__append_sequence__low__seed42
                fast__discovery__candidate__contains_ordered__low__seed42
                fast__discovery__candidate__exact_sequence__medium__seed42
                fast__discovery__candidate__starts_with__low__seed42
            )
            export SELF_GROK_ONLY_DISCOVERY_BRANCH="${BATCH2_BRANCHES[$SLURM_ARRAY_TASK_ID]}"
            ;;
        confirmation)
            export SELF_GROK_ONLY_CONFIRMATION_INDEX="$SLURM_ARRAY_TASK_ID"
            ;;
        *)
            echo "Unknown SELF_GROK_ARRAY_MODE=${SELF_GROK_ARRAY_MODE:-}" >&2
            exit 2
            ;;
    esac
fi

python "$EXP_ROOT/src/fast_orchestrate.py"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    exit 0
fi

if [[ -f "$EXP_ROOT/manifests/stage1_handoff_ready.json" && ! -f "$EXP_ROOT/manifests/fast_experiment_complete.json" ]]; then
    confirmation_job=$(sbatch --parsable --array=0-8%2 --mem=32gb --job-name=self-grok-confirm --output="$EXP_ROOT/logs/confirmation_%A_%a.out" --error="$EXP_ROOT/logs/confirmation_%A_%a.err" --export=ALL,SELF_GROK_ARRAY_MODE=confirmation "$EXP_ROOT/scripts/run_stage1_tail_one_gpu.sbatch")
    final_job=$(sbatch --parsable --dependency=afterok:"$confirmation_job" "$EXP_ROOT/scripts/run_full.sbatch")
    echo "Submitted confirmation array $confirmation_job and finalizer $final_job"
fi

# Reaching this line means the fixed-budget fast protocol has either confirmed
# the frozen pair or produced an honest null/non-replication result.
echo "Fast matched-pair experiment completed at $(date --iso-8601=seconds)"
