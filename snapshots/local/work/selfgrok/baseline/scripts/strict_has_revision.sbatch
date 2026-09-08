#!/bin/bash
#SBATCH --job-name=strict-has-revision
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/strict_has_revision_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/strict_has_revision_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"

python "$EXP_ROOT/src/recipe_run_branch.py" \
    --branch-id recipe__decisive__regex_dense \
    --port 8300
