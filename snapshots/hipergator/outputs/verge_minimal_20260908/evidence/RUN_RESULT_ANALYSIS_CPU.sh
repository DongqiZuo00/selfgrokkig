#!/usr/bin/env bash
set -euo pipefail
module load python/3.11
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR='/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908/evidence/result_analysis_mpl'
cd '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' evidence/analyze_target_eval.py \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output evidence/result_analysis_eval_cpu.json > evidence/result_analysis_eval_cpu.log 2>&1
cat evidence/result_analysis_eval_cpu.log
