#!/bin/bash
#SBATCH --job-name=recipe-analyze
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/analyze_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/analyze_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/recipe_analyze.py"
