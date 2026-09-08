#!/bin/bash
set -euo pipefail

EXP_ROOT="/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness"
cd "/blue/du.j/jinjiaguo/self grok"

prepare_job=$(sbatch --parsable "$EXP_ROOT/scripts/recipe_prepare.sbatch")
root_job=$(sbatch --parsable --dependency="afterok:$prepare_job" "$EXP_ROOT/scripts/recipe_root.sbatch")
screen_job=$(sbatch --parsable --dependency="afterok:$root_job" "$EXP_ROOT/scripts/recipe_screen.sbatch")
branch_job=$(sbatch --parsable --dependency="afterok:$screen_job" --array=0-2%2 "$EXP_ROOT/scripts/recipe_branches.sbatch")
analyze_job=$(sbatch --parsable --dependency="afterok:$branch_job" "$EXP_ROOT/scripts/recipe_analyze.sbatch")

source "$EXP_ROOT/scripts/recipe_env.sh"
python "$EXP_ROOT/src/recipe_record_submission.py" \
    --prepare "$prepare_job" \
    --root "$root_job" \
    --screen "$screen_job" \
    --branches "$branch_job" \
    --analyze "$analyze_job"

echo "prepare=$prepare_job root=$root_job screen=$screen_job branches=$branch_job analyze=$analyze_job"
