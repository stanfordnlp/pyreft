#!/bin/bash
#SBATCH --job-name=reft-rank1
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# ============================================================
# Rank-1 experiment for testing intervention metrics
# ============================================================
# Usage:
#   sbatch rank1_sweep.sbatch                      # Default (baseline)
#   sbatch --export=SCALE_TYPE=scalar rank1_sweep.sbatch
# ============================================================

set -e

# Configuration
SCALE_TYPE=${SCALE_TYPE:-""}
LR=${LR:-5e-4}
POSITION=${POSITION:-"f1+s1"}
MAX_EXAMPLES=${MAX_EXAMPLES:-50000}
WANDB_PROJECT=${WANDB_PROJECT:-"loreft-regret-rank1"}

# Build run name
if [ -n "$SCALE_TYPE" ]; then
    RUN_NAME="r1_${SCALE_TYPE}___${POSITION}___lr${LR}"
else
    RUN_NAME="r1___${POSITION}___lr${LR}"
fi

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Run: $RUN_NAME"
echo "Scale type: ${SCALE_TYPE:-none}"
echo "LR: $LR"
echo "============================================"

cd "$SLURM_SUBMIT_DIR/.."

# Build command
CMD="uv run train.py \
    --rank 1 \
    --position $POSITION \
    --lr $LR \
    --logging_steps 1 \
    --max_n_train_example $MAX_EXAMPLES \
    --epochs 1 \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --run_name $RUN_NAME"

# Add scale type if specified
if [ -n "$SCALE_TYPE" ]; then
    CMD="$CMD --scale_type $SCALE_TYPE"
fi

# Add share_weights for "all" positions
if [[ "$POSITION" == "all" || "$POSITION" == "alls" ]]; then
    CMD="$CMD --share_weights"
fi

$CMD

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
