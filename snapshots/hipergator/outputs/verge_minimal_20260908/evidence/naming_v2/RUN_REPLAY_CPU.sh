#!/usr/bin/env bash
set -euo pipefail
module load python/3.11
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR='/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/evidence/naming_v2/mpl'
cd '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' evidence/naming_v2/replay_naming_v2.py \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output evidence/naming_v2/cpu_replay_metadata_complete.json > evidence/naming_v2/cpu_replay_metadata_complete.log 2>&1
cat evidence/naming_v2/cpu_replay_metadata_complete.log
sacct -j 41415498 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,ElapsedRaw,AllocTRES,ReqMem,MaxRSS -P \
  > evidence/naming_v2/sacct_41415498.txt
cat evidence/naming_v2/sacct_41415498.txt
sha256sum \
  '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000/adapter_model.safetensors' \
  '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/runs/naming_v2_41415498/one_binary_update_proof/adapter_model.safetensors' \
  > evidence/naming_v2/adapter_hash_verification.txt
cat evidence/naming_v2/adapter_hash_verification.txt
