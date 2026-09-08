#!/bin/bash
#SBATCH --job-name=mistral-v56-painter
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/mistral_v56_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/mistral_v56_%j.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/self_grok_decisive_v5_6_run.py"
