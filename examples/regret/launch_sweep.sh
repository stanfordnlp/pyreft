#!/bin/bash
# ============================================================
# Launch sweep for "ReFT Without Regret" experiments
# ============================================================
# This script submits a grid of jobs varying rank and LR
# 
# Usage:
#   ./launch_sweep.sh              # Full grid
#   ./launch_sweep.sh --dry-run    # Print commands without submitting
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
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# --- Create logs directory ---
mkdir -p logs

# --- Submit jobs ---
echo "Submitting sweep: ${#RANKS[@]} ranks x ${#LRS[@]} LRs = $((${#RANKS[@]} * ${#LRS[@]})) jobs"
echo ""

job_count=0
for rank in "${RANKS[@]}"; do
    for lr in "${LRS[@]}"; do
        job_name="loreft_r${rank}_lr${lr}"
        
        cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
        
        if $DRY_RUN; then
            echo "$cmd"
        else
            echo "Submitting: rank=$rank, lr=$lr"
            $cmd
        fi
        
        job_count=$((job_count + 1))
    done
done

echo ""
echo "Submitted $job_count jobs"

