#!/usr/bin/env bash
set -euo pipefail
module load python/3.11
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR='/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/evidence/round2_verifier/mpl'
cd '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' evidence/round2_verifier/select_groups.py \
  > evidence/round2_verifier/selection_cpu.log 2>&1
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' evidence/round2_verifier/replay_training.py \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output evidence/round2_verifier/cpu_replay.json > evidence/round2_verifier/cpu_replay.log 2>&1
cat evidence/round2_verifier/cpu_replay.log
