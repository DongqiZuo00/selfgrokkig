#!/bin/bash
#SBATCH --job-name=selfgrok-bridge-r6-analysis
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=experiments/has_transfer_witness/logs/selfgrok_bridge_r6_analysis_%j.out
#SBATCH --error=experiments/has_transfer_witness/logs/selfgrok_bridge_r6_analysis_%j.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
python "$EXP_ROOT/src/self_grok_bridge_chain_round6_analyze.py"
