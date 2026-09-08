#!/bin/bash
#SBATCH --job-name=softphi-r1
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/softphi_r1_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/softphi_r1_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048

BRANCHES=(
    softphi_r1__direct
    softphi_r1__starts_with_low
    softphi_r1__append_sequence_high
    softphi_r1__contains_ordered_low
)

python "$EXP_ROOT/src/recipe_run_branch.py" \
    --branch-id "${BRANCHES[$SLURM_ARRAY_TASK_ID]}" \
    --port "$((8600 + SLURM_ARRAY_TASK_ID))"
