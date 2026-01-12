#!/bin/bash
# ============================================================
# Quick rank-1 experiments to test intervention metrics logging
# ============================================================
# Usage:
#   ./scripts/run_rank1_exps.sh              # Run all
#   ./scripts/run_rank1_exps.sh --dry-run    # Print commands only
# ============================================================

set -e

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

# Common settings
COMMON_ARGS="--rank 1 --position f1+s1 --logging_steps 1 --max_n_train_example 1000 --epochs 1 --use_wandb --wandb_project loreft-regret-rank1 --debug_interventions"

# Experiments to run
declare -A EXPERIMENTS=(
    ["baseline"]=""
    ["scale_scalar"]="--scale_type scalar"
    ["scale_sigmoid"]="--scale_type sigmoid"
    ["scale_datadep"]="--scale_type datadep"
    ["scale_token"]="--scale_type token"
)

# Learning rates to test
LRS=(5e-4 1e-3)

echo "=== Rank 1 Experiments ==="
echo "Logging every step to track intervention metrics"
echo ""

for exp_name in "${!EXPERIMENTS[@]}"; do
    exp_args="${EXPERIMENTS[$exp_name]}"
    for lr in "${LRS[@]}"; do
        run_name="r1_${exp_name}_lr${lr}"
        cmd="uv run train.py $COMMON_ARGS $exp_args --lr $lr --run_name $run_name"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "=== Running: $run_name ==="
            $cmd
            echo ""
        fi
    done
done

echo "Done!"
