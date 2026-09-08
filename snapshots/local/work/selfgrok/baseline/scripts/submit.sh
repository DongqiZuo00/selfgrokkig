#!/bin/bash
set -euo pipefail
WORK_ROOT="/blue/du.j/jinjiaguo/self grok"
cd "$WORK_ROOT"
bash experiments/uncertainty_target_gain/scripts/software_snapshot.sh
sbatch experiments/uncertainty_target_gain/scripts/run_full.sbatch
