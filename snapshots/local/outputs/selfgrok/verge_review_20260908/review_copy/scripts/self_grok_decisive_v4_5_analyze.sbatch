#!/bin/bash
#SBATCH --job-name=mistral-v45-analyze
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/mistral_v45_analyze_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/mistral_v45_analyze_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export SELF_GROK_DECISIVE_VERSION=self_grok_decisive_mistral_v4_5
python "$EXP_ROOT/src/self_grok_decisive_v4_3_analyze.py"
