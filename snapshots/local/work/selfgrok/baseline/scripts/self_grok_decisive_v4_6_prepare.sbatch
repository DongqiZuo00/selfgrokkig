#!/bin/bash
#SBATCH --job-name=mistral-v46-prep
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/mistral_v46_prepare_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/mistral_v46_prepare_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export SELF_GROK_DECISIVE_VERSION=self_grok_decisive_mistral_v4_6
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/self_grok_decisive_v4_6_prepare.py"
