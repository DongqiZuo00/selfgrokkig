#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
cd "$ROOT"
test -f runs/round2_result/summary.json
mkdir -p evidence
# Adapters remain intact on HiPerGator; collect metadata and all raw observations.
tar --exclude='adapter_model.safetensors' -czf evidence/round2_core.tar.gz \
  runs/round_prepare_named_41417194 runs/round_prepare_named_41417194.out runs/round_prepare_named_41417194.err \
  runs/round2_g1_41418169_0 runs/round2_g2_41418169_1 \
  runs/round2_g3_41418169_2 runs/round2_direct_41418169_3 \
  runs/round2_41418169_0.out runs/round2_41418169_0.err \
  runs/round2_41418169_1.out runs/round2_41418169_1.err \
  runs/round2_41418169_2.out runs/round2_41418169_2.err \
  runs/round2_41418169_3.out runs/round2_41418169_3.err \
  runs/round2_result runs/round2_cpu_preflight.json runs/round2_array_submission.txt
sha256sum evidence/round2_core.tar.gz
