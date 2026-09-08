#!/bin/bash
#SBATCH --job-name=verge-control-coordinate
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/has_transfer_witness/logs/verge_control_coordinate_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/verge_control_coordinate_%j.err
source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export VERGE_BOOK_SUITE=verge_book_v2
export VERGE_CONTROL_SUITE=verge_control_v1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python "$EXP_ROOT/src/verge_control_controller.py" coordinate --round "${1:?Expected completed control round}"
