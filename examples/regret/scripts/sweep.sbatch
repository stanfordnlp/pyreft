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
# ReFT config
RANK=${RANK:-4}
LR=${LR:-5e-4}
LAYERS=${LAYERS:-"all"}
POSITION=${POSITION:-"f1+l1"}
SHARE_WEIGHTS=${SHARE_WEIGHTS:-false}
SCALE_TYPE=${SCALE_TYPE:-""}
COMPONENT=${COMPONENT:-"block_output"}
INTERVENTION_TYPE=${INTERVENTION_TYPE:-"loreft"}

# LoRA config
USE_LORA=${USE_LORA:-false}
DISABLE_REFT=${DISABLE_REFT:-false}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_MODULES=${LORA_MODULES:-"q_proj;v_proj"}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# General training config
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_EXAMPLES=${MAX_EXAMPLES:-50000}
EVAL_SPLIT=${EVAL_SPLIT:-0.05}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-500}
EVAL_STEPS=${EVAL_STEPS:-50}
MODEL=${MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
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
echo "  Intervention type: $INTERVENTION_TYPE"
echo "  Rank: $RANK"
echo "  LR: $LR"
echo "  Layers: $LAYERS"
echo "  Position: $POSITION"
echo "  Component: $COMPONENT"
echo "  Max examples: $MAX_EXAMPLES"
echo "  Flash attention: $USE_FLASH_ATTN"
echo "  Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "============================================"

# Change to submit directory (where sbatch was called from)
cd "$SLURM_SUBMIT_DIR"

# Create logs directory if needed
mkdir -p logs

# --- Build run name ---
# Add scale type suffix if enabled
SCALE_SUFFIX=""
if [ -n "$SCALE_TYPE" ]; then
    SCALE_SUFFIX="_${SCALE_TYPE}"
fi

# Add component to name if not block_output (default)
COMPONENT_PART=""
if [ "$COMPONENT" != "block_output" ]; then
    COMPONENT_PART="___${COMPONENT}"
fi

if [ "$FULL_FINETUNE" = "true" ]; then
    RUN_NAME="fullft_lr${LR}"
elif [ "$USE_LORA" = "true" ] && [ "$DISABLE_REFT" = "true" ]; then
    # LoRA-only: include rank and modules in name
    # Shorten module names for readability
    MODULES_SHORT=$(echo "$LORA_MODULES" | sed 's/;/+/g' | sed 's/_proj//g')
    RUN_NAME="lora_r${LORA_RANK}___${MODULES_SHORT}___lr${LR}"
elif [ "$USE_LORA" = "true" ]; then
    # LoRA + ReFT combined
    MODULES_SHORT=$(echo "$LORA_MODULES" | sed 's/;/+/g' | sed 's/_proj//g')
    RUN_NAME="lora_r${LORA_RANK}_${INTERVENTION_TYPE}_r${RANK}${SCALE_SUFFIX}___${MODULES_SHORT}___lr${LR}"
else
    # ReFT-only (loreft or direft)
    RUN_NAME="${INTERVENTION_TYPE}_r${RANK}${SCALE_SUFFIX}___${POSITION}${COMPONENT_PART}___lr${LR}"
fi

# --- Build command ---
CMD="uv run train.py \
    --model_name_or_path $MODEL \
    --run_name $RUN_NAME \
    --intervention_type $INTERVENTION_TYPE \
    --rank $RANK \
    --lr $LR \
    --layers $LAYERS \
    --position $POSITION \
    --component $COMPONENT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --max_n_train_example $MAX_EXAMPLES \
    --eval_split $EVAL_SPLIT \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --eval_steps $EVAL_STEPS \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --debug_interventions"

# Add share_weights flag if needed
if [ "$SHARE_WEIGHTS" = "true" ]; then
    CMD="$CMD --share_weights"
fi

# Add scale_type if specified
if [ -n "$SCALE_TYPE" ]; then
    CMD="$CMD --scale_type $SCALE_TYPE"
fi

# Add flash attention if enabled
if [ "$USE_FLASH_ATTN" = "true" ]; then
    CMD="$CMD --use_flash_attn"
fi

# Add full finetune flag if needed
if [ "$FULL_FINETUNE" = "true" ]; then
    CMD="$CMD --full_finetune"
fi

# Add gradient checkpointing if enabled (for large models)
if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    CMD="$CMD --gradient_checkpointing"
fi

# Add LoRA flags if enabled
if [ "$USE_LORA" = "true" ]; then
    CMD="$CMD --use_lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_modules $LORA_MODULES --lora_dropout $LORA_DROPOUT"
    if [ "$DISABLE_REFT" = "true" ]; then
        CMD="$CMD --disable_reft"
    fi
fi

# --- Run training ---
$CMD

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

