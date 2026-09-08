#!/bin/bash
#SBATCH --job-name=verge-r1-branch
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_r1_branch_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_r1_branch_%A_%a.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/verge_round_train.py"
