#!/bin/bash
# ============================================================
# Launch Commonsense + Math Reasoning with Strict Positions
# ============================================================
# Runs both tasks with f7+s7 (strict mode) across 3 seeds.
# Paper used f7+l7 (legacy) for BOTH tasks.
# This tests if fixing the off-by-one bug affects scores.
#
# Usage (run from examples/loreft/):
#   ./scripts/launch_reasoning.sh              # Submit all jobs
#   ./scripts/launch_reasoning.sh --dry-run    # Print commands only
#   ./scripts/launch_reasoning.sh --math-only  # Only math tasks
#   ./scripts/launch_reasoning.sh --commonsense-only  # Only commonsense
# ============================================================

set -e

# Seeds from paper (first 3 for non-GLUE tasks)
SEEDS=(42 43 44)

# Parse args
DRY_RUN=false
MATH_ONLY=false
COMMONSENSE_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --math-only) MATH_ONLY=true; echo "=== MATH ONLY ===" ;;
        --commonsense-only) COMMONSENSE_ONLY=true; echo "=== COMMONSENSE ONLY ===" ;;
    esac
done

mkdir -p logs

job_count=0

# --- Commonsense jobs ---
if ! $MATH_ONLY; then
    echo ""
    echo "=== Commonsense Reasoning (f7+s7 - paper used f7+l7) ==="
    for seed in "${SEEDS[@]}"; do
        job_name="commonsense_f7s7_seed${seed}"
        cmd="sbatch --job-name=$job_name --export=ALL,SEED=$seed scripts/commonsense_strict.sbatch"

        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: commonsense seed=$seed"
            $cmd
        fi
        job_count=$((job_count + 1))
    done
fi

# --- Math jobs ---
if ! $COMMONSENSE_ONLY; then
    echo ""
    echo "=== Math Reasoning (f7+s7 - changed from paper's f7+l7) ==="
    for seed in "${SEEDS[@]}"; do
        job_name="math_f7s7_seed${seed}"
        cmd="sbatch --job-name=$job_name --export=ALL,SEED=$seed scripts/math_strict.sbatch"

        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: math seed=$seed"
            $cmd
        fi
        job_count=$((job_count + 1))
    done
fi

echo ""
echo "Total: $job_count jobs"
