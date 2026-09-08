#!/bin/bash
set -euo pipefail
ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908"
cd "$ROOT"
test -f runs/round1_result/summary.json
mkdir -p evidence
tar -czf evidence/round1_core.tar.gz \
  runs/round_prepare_41413994 runs/round_prepare_41413994.out runs/round_prepare_41413994.err \
  runs/round1_g1_41414619_0 runs/round1_g2_41414619_1 \
  runs/round1_g3_41414619_2 runs/round1_direct_41414619_3 \
  runs/round1_41414619_0.out runs/round1_41414619_0.err \
  runs/round1_41414619_1.out runs/round1_41414619_1.err \
  runs/round1_41414619_2.out runs/round1_41414619_2.err \
  runs/round1_41414619_3.out runs/round1_41414619_3.err \
  runs/round1_result runs/round1_cpu_preflight.json runs/round1_array_submission.txt
sha256sum evidence/round1_core.tar.gz
