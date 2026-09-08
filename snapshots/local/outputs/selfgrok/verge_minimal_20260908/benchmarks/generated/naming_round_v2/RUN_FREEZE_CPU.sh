#!/usr/bin/env bash
set -euo pipefail
module load python/3.11
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' benchmarks/freeze_naming_round_v2.py \
  > benchmarks/generated/naming_round_v2/freeze_cpu.log 2>&1
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' benchmarks/test_freeze_naming_round_v2.py \
  > benchmarks/generated/naming_round_v2/cpu_tests.log 2>&1
cat benchmarks/generated/naming_round_v2/freeze_cpu.log
cat benchmarks/generated/naming_round_v2/cpu_tests.log
sha256sum benchmarks/generated/naming_round_v2/target_rows.json \
  benchmarks/generated/naming_round_v2/target_view_metadata.json \
  benchmarks/generated/naming_round_v2/catalogue.json \
  benchmarks/generated/naming_round_v2/stage_validation_evidence.json \
  > benchmarks/generated/naming_round_v2/files_sha256.txt
cat benchmarks/generated/naming_round_v2/files_sha256.txt
