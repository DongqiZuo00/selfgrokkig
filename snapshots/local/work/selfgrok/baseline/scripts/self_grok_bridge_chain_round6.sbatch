#!/bin/bash
#SBATCH --job-name=selfgrok-bridge-r6
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/selfgrok_bridge_r6_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/selfgrok_bridge_r6_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048

BRANCHES=(sg_bridge_r6__direct_target sg_bridge_r6__challenger_chain)
python "$EXP_ROOT/src/recipe_run_branch.py" --branch-id "${BRANCHES[$SLURM_ARRAY_TASK_ID]}" --port "$((9200 + SLURM_ARRAY_TASK_ID))"
