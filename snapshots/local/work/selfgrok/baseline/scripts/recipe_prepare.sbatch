#!/bin/bash
#SBATCH --job-name=recipe-prepare
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/prepare_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/prepare_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/download_model.py"
python "$EXP_ROOT/src/recipe_prepare.py"
python -m compileall -q "$EXP_ROOT/src"
