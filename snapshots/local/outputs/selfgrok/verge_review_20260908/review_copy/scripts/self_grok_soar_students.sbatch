#!/bin/bash
#SBATCH --job-name=mistral-soar-inner
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/mistral_soar_inner_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/mistral_soar_inner_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export ROLLOUT_MAX_COMPLETION_TOKENS=2048

python "$EXP_ROOT/src/self_grok_soar_run_student.py" \
  --outer "$1" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --port "$((12000 + ((SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID * 997) % 40000)))"
