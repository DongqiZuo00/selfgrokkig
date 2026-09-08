#!/bin/bash
#SBATCH --job-name=verge-followon-coordinate
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_followon_coordinate_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_followon_coordinate_%j.err
set -euo pipefail
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_BOOK_SUITE=verge_book_v2
export VERGE_FOLLOWON_SUITE=verge_followon_v1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python "$EXP_ROOT/src/verge_followon_controller.py" coordinate
