#!/bin/bash
#SBATCH --job-name=decisive-analyze
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:20:00
#SBATCH --output=experiments/has_transfer_witness/logs/decisive_analyze_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/decisive_analyze_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/decisive_analyze.py"
