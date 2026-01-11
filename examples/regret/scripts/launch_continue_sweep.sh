#!/bin/bash
# ============================================================
# Launch 10x retraining sweep for "ReFT Without Regret" experiments
# ============================================================
# This script retrains with the best LR for each (rank, position)
# configuration, training from scratch for 10 epochs.
#
# Usage (run from examples/regret/):
#   ./scripts/launch_continue_sweep.sh              # Launch all
#   ./scripts/launch_continue_sweep.sh --dry-run    # Print commands
#   ./scripts/launch_continue_sweep.sh --skip-done  # Skip completed
# ============================================================

set -e

# --- Configuration ---
RANKS=(1 2 4 8 16 32 64)
POSITIONS=("f1+l1" "all" "f1+s1" "alls")

# Output settings
SOURCE_PROJECT="loreft-regret"
OUTPUT_PROJECT="loreft-regret-10x"
OUTPUT_DIR="./outputs_10x"
EPOCHS=10

# --- Parse args ---
DRY_RUN=false
SKIP_DONE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-done) SKIP_DONE=true; echo "=== SKIPPING COMPLETED JOBS ===" ;;
    esac
done

# Function to check if job is already done
is_done() {
    local rank="$1"
    local position="$2"
    # Check if any 10x run exists for this rank/position
    ls "${OUTPUT_DIR}/r${rank}___10x_${position}___lr"*/training_args.json 1>/dev/null 2>&1
}

# --- Create directories ---
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# --- Calculate and submit jobs ---
total_jobs=$((${#RANKS[@]} * ${#POSITIONS[@]}))
job_count=0
skipped_count=0

echo "Submitting 10x retraining sweep:"
echo "  Ranks: ${RANKS[*]}"
echo "  Positions: ${POSITIONS[*]}"
echo "  Total: $total_jobs jobs"
echo "  Source project: $SOURCE_PROJECT"
echo "  Output project: $OUTPUT_PROJECT"
echo "  Epochs: $EPOCHS"
echo ""

for rank in "${RANKS[@]}"; do
    for position in "${POSITIONS[@]}"; do
        # Skip if already done
        if $SKIP_DONE && is_done "$rank" "$position"; then
            echo "Skipping (done): rank=$rank, position=$position"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        
        job_name="loreft_10x_r${rank}_${position//+/_}"
        
        cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,POSITION=$position,SOURCE_PROJECT=$SOURCE_PROJECT,OUTPUT_PROJECT=$OUTPUT_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,EPOCHS=$EPOCHS scripts/continue_sweep.sbatch"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: rank=$rank, position=$position (10 epochs from scratch)"
            $cmd
        fi
        
        job_count=$((job_count + 1))
    done
done

echo ""
echo "Submitted $job_count jobs"
if $SKIP_DONE; then
    echo "Skipped $skipped_count completed jobs"
fi
