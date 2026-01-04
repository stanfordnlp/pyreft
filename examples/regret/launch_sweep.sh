#!/bin/bash
# ============================================================
# Launch sweep for "ReFT Without Regret" experiments
# ============================================================
# This script submits a grid of jobs varying rank and LR
# 
# Usage:
#   ./launch_sweep.sh              # Full grid
#   ./launch_sweep.sh --dry-run    # Print commands without submitting
#   ./launch_sweep.sh --skip-done  # Skip completed jobs
# ============================================================

set -e

# --- Sweep configuration ---
RANKS=(1 2 4 8 16 32 64)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)

# Other settings (modify as needed)
MAX_EXAMPLES=50000
EPOCHS=1
WANDB_PROJECT="loreft-regret"
OUTPUT_DIR="./outputs"

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
    [[ -f "${OUTPUT_DIR}/${run_name}/training_args.json" ]]
}

# --- Create logs directory ---
mkdir -p logs

# --- Submit jobs ---
echo "Submitting sweep: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x 2 positions = $((${#RANKS[@]} * ${#LRS[@]} * 2)) jobs"
echo ""

job_count=0
skipped_count=0

# --- Standard f1+l1 position sweep ---
for rank in "${RANKS[@]}"; do
    for lr in "${LRS[@]}"; do
        job_name="loreft_r${rank}_lr${lr}"
        run_name="r${rank}___f1+l1___lr${lr}"
        
        # Skip if already done
        if $SKIP_DONE && is_done "$run_name"; then
            echo "Skipping (done): $run_name"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        
        cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=f1+l1,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: rank=$rank, position=f1+l1, lr=$lr"
            $cmd
        fi
        
        job_count=$((job_count + 1))
    done
done

# --- All positions sweep (with share_weights) ---
for rank in "${RANKS[@]}"; do
    for lr in "${LRS[@]}"; do
        job_name="loreft_r${rank}_all_lr${lr}"
        run_name="r${rank}___all___lr${lr}"
        
        # Skip if already done
        if $SKIP_DONE && is_done "$run_name"; then
            echo "Skipping (done): $run_name"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        
        cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=all,SHARE_WEIGHTS=true,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: rank=$rank, position=all, lr=$lr"
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

