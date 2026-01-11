#!/bin/bash
#SBATCH --job-name=loreft-10x
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=14-0
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# ============================================================
# Retrain with best LR for 10x longer (from scratch)
# ============================================================
# Usage:
#   sbatch --export=RANK=4,POSITION=f1+l1 scripts/continue_sweep.sbatch
#   ./scripts/launch_continue_sweep.sh
# ============================================================

set -e

# --- Configuration (override via --export or environment) ---
RANK=${RANK:-4}
POSITION=${POSITION:-"f1+l1"}
SOURCE_PROJECT=${SOURCE_PROJECT:-"loreft-regret"}
OUTPUT_PROJECT=${OUTPUT_PROJECT:-"loreft-regret-10x-restart"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs_10x"}
EPOCHS=${EPOCHS:-10}

# --- Setup ---
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "============================================"
echo "Config:"
echo "  Rank: $RANK"
echo "  Position: $POSITION"
echo "  Source project: $SOURCE_PROJECT"
echo "  Output project: $OUTPUT_PROJECT"
echo "  Epochs: $EPOCHS"
echo "============================================"

# Change to submit directory (where sbatch was called from)
cd "$SLURM_SUBMIT_DIR"

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# --- Run retraining ---
# The script will fetch the best LR from wandb and train from scratch
uv run continue_training.py \
    --wandb_project "$SOURCE_PROJECT" \
    --output_wandb_project "$OUTPUT_PROJECT" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --rank "$RANK" \
    --position "$POSITION"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
