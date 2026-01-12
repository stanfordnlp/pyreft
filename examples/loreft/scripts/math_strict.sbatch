#!/bin/bash
#SBATCH --job-name=loreft-math
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=14-0
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ============================================================
# Math Reasoning with Strict Position Parsing (f7+s7)
# ============================================================
# Changed from paper's f7+l7 to f7+s7 (strict mode).
# Evaluates on: MultiArith, gsm8k, SVAMP, mawps, AddSub, AQuA, SingleEq
# ============================================================

set -e

# --- Configuration ---
SEED=${SEED:-42}
MODEL=${MODEL:-"yahma/llama-7b-hf"}
RANK=${RANK:-8}
POSITION=${POSITION:-"f7+s7"}  # Strict mode (paper used f7+l7)
EPOCHS=${EPOCHS:-12}
LR=${LR:-9e-4}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
DATA_DIR=${DATA_DIR:-"dataset"}
WANDB_PROJECT=${WANDB_PROJECT:-"loreft-math-strict"}

# --- Setup ---
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "============================================"
echo "Config:"
echo "  Task: math"
echo "  Model: $MODEL"
echo "  Seed: $SEED"
echo "  Rank: $RANK"
echo "  Position: $POSITION (strict - paper used f7+l7)"
echo "  Epochs: $EPOCHS"
echo "  LR: $LR"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- Run training + evaluation ---
uv run train.py -task math \
    -data_dir "$DATA_DIR" \
    -model "$MODEL" \
    -seed "$SEED" \
    -l all \
    -r "$RANK" \
    -p "$POSITION" \
    -e "$EPOCHS" \
    -lr "$LR" \
    -type LoreftIntervention \
    -gradient_accumulation_steps "$GRAD_ACCUM" \
    -batch_size "$BATCH_SIZE" \
    -eval_batch_size "$EVAL_BATCH_SIZE" \
    --dropout 0.00 \
    --test_split test \
    --use_normalized_template \
    --share_weights \
    --warmup_ratio 0.1 \
    --greedy_decoding \
    --is_wandb \
    --wandb_name "math_${POSITION}_seed${SEED}"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
