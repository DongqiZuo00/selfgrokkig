#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
cd "$ROOT"
test -f runs/round3_result/summary.json
mkdir -p evidence
# Adapters remain intact on HiPerGator; collect metadata and all raw observations.
tar --exclude='adapter_model.safetensors' -czf evidence/round3_core.tar.gz \
  runs/round_prepare_ignition_41419999 runs/round_prepare_ignition_41419999.out runs/round_prepare_ignition_41419999.err \
  runs/round3_g1_41420288_0 runs/round3_g2_41420288_1 \
  runs/round3_g3_41420288_2 runs/round3_direct_41420288_3 \
  runs/round3_41420288_0.out runs/round3_41420288_0.err \
  runs/round3_41420288_1.out runs/round3_41420288_1.err \
  runs/round3_41420288_2.out runs/round3_41420288_2.err \
  runs/round3_41420288_3.out runs/round3_41420288_3.err \
  runs/round3_result runs/round3_cpu_preflight.json runs/round3_array_submission.txt
sha256sum evidence/round3_core.tar.gz
