#!/bin/bash
#SBATCH --job-name=softphi-r2-analysis
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/has_transfer_witness/logs/softphi_r2_analysis_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/softphi_r2_analysis_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/soft_phi_round2_analyze.py"
