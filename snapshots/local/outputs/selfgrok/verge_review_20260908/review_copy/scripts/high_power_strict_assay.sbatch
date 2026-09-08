#!/bin/bash
#SBATCH --job-name=strict-assay
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=14-00:00:00
#SBATCH --output=experiments/has_transfer_witness/logs/high_power_assay_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/high_power_assay_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/high_power_strict_assay.py"
