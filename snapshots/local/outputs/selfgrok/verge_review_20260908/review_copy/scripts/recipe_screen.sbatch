#!/bin/bash
#SBATCH --job-name=recipe-screen
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/screen_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/screen_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/recipe_screen.py"
