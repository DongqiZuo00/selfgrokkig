#!/bin/bash
#SBATCH --job-name=selfgrok-strict-r4
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/selfgrok_strict_r4_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/selfgrok_strict_r4_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048

BRANCHES=(
    sg_strict_r4__direct_from_r3u100
    sg_strict_r4__oracle_from_r3u100
    sg_strict_r4__challenger_regex_from_r3u100
)

python "$EXP_ROOT/src/recipe_run_branch.py" \
    --branch-id "${BRANCHES[$SLURM_ARRAY_TASK_ID]}" \
    --port "$((9000 + SLURM_ARRAY_TASK_ID))"
