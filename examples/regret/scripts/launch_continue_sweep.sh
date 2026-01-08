#!/bin/bash
# ============================================================
# Launch continuation sweep for "ReFT Without Regret" experiments
# ============================================================
# This script continues training from the best LR for each
# (rank, position) configuration, running for 10x longer.
#
# Usage (run from examples/regret/):
#   ./scripts/launch_continue_sweep.sh              # Launch all
#   ./scripts/launch_continue_sweep.sh --dry-run    # Print commands
#   ./scripts/launch_continue_sweep.sh --skip-done  # Skip completed
# ============================================================

set -e

# --- Configuration ---
# These should match the original sweep
RANKS=(1 2 4 8 16 32 64)
POSITIONS=("f1+l1" "all" "f1+s1" "alls")

# Output settings
SOURCE_PROJECT="loreft-regret"
OUTPUT_PROJECT="loreft-regret-10x"
OUTPUT_DIR="./outputs_10x"
EPOCHS_MULTIPLIER=10

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
    local run_name="$1"
    [[ -f "${OUTPUT_DIR}/${run_name}___10x/training_args.json" ]]
}

# --- Create directories ---
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# --- Calculate and submit jobs ---
total_jobs=$((${#RANKS[@]} * ${#POSITIONS[@]}))
job_count=0
skipped_count=0

echo "Submitting continuation sweep:"
echo "  Ranks: ${RANKS[*]}"
echo "  Positions: ${POSITIONS[*]}"
echo "  Total: $total_jobs jobs"
echo "  Source project: $SOURCE_PROJECT"
echo "  Output project: $OUTPUT_PROJECT"
echo ""

for rank in "${RANKS[@]}"; do
    for position in "${POSITIONS[@]}"; do
        # Build expected run name (matches original naming convention)
        run_name="r${rank}___${position}___lr*"  # We don't know best LR yet
        
        # For checking if done, we need to check all possible LRs
        # or just check if any 10x run exists for this rank/position
        check_pattern="${OUTPUT_DIR}/r${rank}___${position}___*___10x/training_args.json"
        
        # Skip if already done
        if $SKIP_DONE && ls $check_pattern 1>/dev/null 2>&1; then
            echo "Skipping (done): rank=$rank, position=$position"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        
        job_name="loreft_10x_r${rank}_${position//+/_}"
        
        cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,POSITION=$position,SOURCE_PROJECT=$SOURCE_PROJECT,OUTPUT_PROJECT=$OUTPUT_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,EPOCHS_MULTIPLIER=$EPOCHS_MULTIPLIER scripts/continue_sweep.sbatch"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: rank=$rank, position=$position (10x continuation)"
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

