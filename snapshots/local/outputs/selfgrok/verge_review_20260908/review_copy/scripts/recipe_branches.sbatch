#!/bin/bash
#SBATCH --job-name=recipe-branch
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/branch_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/branch_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"

BRANCHES=(
    recipe__direct__has__seed42
    recipe__candidate__regex__seed42
    recipe__candidate__compr__seed42
)
python "$EXP_ROOT/src/recipe_run_branch.py" \
    --branch-id "${BRANCHES[$SLURM_ARRAY_TASK_ID]}" \
    --port "$((8200 + SLURM_ARRAY_TASK_ID))"
