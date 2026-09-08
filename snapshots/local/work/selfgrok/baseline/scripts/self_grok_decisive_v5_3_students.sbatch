#!/bin/bash
#SBATCH --job-name=mistral-v53-student
#SBATCH --account=du.j
#SBATCH --qos=du.j
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=experiments/has_transfer_witness/logs/mistral_v53_student_%A_%a.out
#SBATCH --error=experiments/has_transfer_witness/logs/mistral_v53_student_%A_%a.err

source "/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/scripts/recipe_env.sh"
export SELF_GROK_DECISIVE_VERSION=self_grok_decisive_mistral_v5_3
export ROLLOUT_MAX_COMPLETION_TOKENS=2048
python "$EXP_ROOT/src/self_grok_decisive_run_student.py" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --port "$((20000 + ((SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID * 991) % 25000)))"
