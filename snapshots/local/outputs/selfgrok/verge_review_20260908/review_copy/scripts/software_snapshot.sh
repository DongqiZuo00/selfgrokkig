#!/bin/bash
set -euo pipefail
WORK_ROOT="/blue/du.j/jinjiaguo/self grok"
cd "$WORK_ROOT"
source "$WORK_ROOT/envs/uncertainty/bin/activate"
{
  python --version
  python -m pip freeze
  git -C vendor/rl-grok-recipe rev-parse HEAD
  git -C vendor/ROLL rev-parse HEAD
  git -C vendor/R-Zero rev-parse HEAD
  nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
  module list
} > "$WORK_ROOT/experiments/uncertainty_target_gain/manifests/software_and_hardware.txt" 2>&1
