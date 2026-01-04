#!/bin/bash
#SBATCH --job-name=loreft-regret
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
# LoReFT Sweep Script for "ReFT Without Regret" experiments
# ============================================================
# Usage:
#   sbatch sweep.sbatch                    # Use defaults
#   sbatch --export=RANK=8,LR=5e-4 sweep.sbatch
#   ./launch_sweep.sh                      # Launch full grid
# ============================================================

set -e

# --- Configuration (override via --export or environment) ---
RANK=${RANK:-4}
LR=${LR:-5e-4}
LAYERS=${LAYERS:-"all"}
POSITION=${POSITION:-"f1+l1"}
SHARE_WEIGHTS=${SHARE_WEIGHTS:-false}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_EXAMPLES=${MAX_EXAMPLES:-50000}
EVAL_SPLIT=${EVAL_SPLIT:-0.05}
EVAL_STEPS=${EVAL_STEPS:-50}
MODEL=${MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
WANDB_PROJECT=${WANDB_PROJECT:-"loreft-regret"}
FULL_FINETUNE=${FULL_FINETUNE:-false}

# --- Setup ---
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "============================================"
echo "Config:"
echo "  Model: $MODEL"
echo "  Rank: $RANK"
echo "  LR: $LR"
echo "  Layers: $LAYERS"
echo "  Position: $POSITION"
echo "  Max examples: $MAX_EXAMPLES"
echo "============================================"

# Change to submit directory (where sbatch was called from)
cd "$SLURM_SUBMIT_DIR"

# Create logs directory if needed
mkdir -p logs

# --- Build run name ---
if [ "$FULL_FINETUNE" = "true" ]; then
    RUN_NAME="fullft_lr${LR}"
else
    RUN_NAME="r${RANK}___${POSITION}___lr${LR}"
fi

# --- Build command ---
CMD="uv run train.py \
    --model_name_or_path $MODEL \
    --run_name $RUN_NAME \
    --rank $RANK \
    --lr $LR \
    --layers $LAYERS \
    --position $POSITION \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --max_n_train_example $MAX_EXAMPLES \
    --eval_split $EVAL_SPLIT \
    --eval_steps $EVAL_STEPS \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project $WANDB_PROJECT"

# Add share_weights flag if needed
if [ "$SHARE_WEIGHTS" = "true" ]; then
    CMD="$CMD --share_weights"
fi

# Add full finetune flag and gradient checkpointing if needed
if [ "$FULL_FINETUNE" = "true" ]; then
    CMD="$CMD --full_finetune --gradient_checkpointing"
fi

# --- Run training ---
$CMD

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

