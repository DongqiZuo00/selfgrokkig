#!/usr/bin/env bash
set -euo pipefail
module load python/3.11
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR='/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/evidence/stage_ignition/mpl'
cd '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' evidence/stage_ignition/analyze_screens.py \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output evidence/stage_ignition/cpu_replay.json \
  > evidence/stage_ignition/cpu_replay_verified.log 2>&1
cat evidence/stage_ignition/cpu_replay_verified.log
sha256sum \
  '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000/adapter_model.safetensors' \
  '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/runs/stage_grammar_v1_41412533/one_binary_update_proof/adapter_model.safetensors' \
  > evidence/stage_ignition/adapter_hash_verification.txt
cat evidence/stage_ignition/adapter_hash_verification.txt
