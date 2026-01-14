#!/bin/bash
# ============================================================
# Launch sweep for MoE-LoReFT experiments
# ============================================================
# This script submits MoE experiments varying rank, LR,
# number of experts, and top-k.
#
# Two MoE variants:
#   - moeloreft: MoE on W (the learned source)
#   - moerloreft: MoE on R (the orthogonal projection)
#
# Usage (run from examples/regret/):
#   ./scripts/launch_moe_sweep.sh              # Default: both variants, f1+s1 only
#   ./scripts/launch_moe_sweep.sh --dry-run    # Print commands without submitting
#   ./scripts/launch_moe_sweep.sh --skip-done  # Skip completed/running jobs
#   ./scripts/launch_moe_sweep.sh --rank1-only # Only rank=1 experiments
#   ./scripts/launch_moe_sweep.sh --model-8b   # Use Llama 3.1 8B
#   ./scripts/launch_moe_sweep.sh --all-positions # Positions: f1+s1, f1+s3, f1+s5, f3+s3, f5+s5
#   ./scripts/launch_moe_sweep.sh --moe-w-only # Only MoE-W (moeloreft)
#   ./scripts/launch_moe_sweep.sh --moe-r-only # Only MoE-R (moerloreft)
# ============================================================

set -e

# --- Sweep configuration ---
RANKS=(1 2 4 8 16 32 64)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)

# MoE configurations: (num_experts, top_k)
NUM_EXPERTS_LIST=(4 16)
TOP_K=2  # Fixed top-k for now

# Position (default: f1+s1)
POSITIONS=("f1+s1")

# Other settings
MAX_EXAMPLES=50000
EPOCHS=1
WANDB_PROJECT="loreft-regret"
OUTPUT_DIR="./outputs"

# Model settings (default: Llama 3.2 1B)
MODEL="meta-llama/Llama-3.2-1B-Instruct"
BATCH_SIZE=2
GRAD_ACCUM=16
USE_FLASH_ATTN=false
GPU_MEM="16G"
SBATCH_EXTRA=""

# --- Parse args ---
DRY_RUN=false
SKIP_DONE=false
RANK1_ONLY=false
MODEL_8B=false
ALL_POSITIONS=false
MOE_W_ONLY=false
MOE_R_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-done) SKIP_DONE=true; echo "=== SKIPPING COMPLETED JOBS ===" ;;
        --rank1-only) RANK1_ONLY=true; echo "=== RANK 1 ONLY ==="; RANKS=(1) ;;
        --model-8b) MODEL_8B=true; echo "=== LLAMA 3.1 8B MODE ===" ;;
        --all-positions) ALL_POSITIONS=true; echo "=== ALL POSITIONS MODE ===" ;;
        --moe-w-only) MOE_W_ONLY=true; echo "=== MOE-W ONLY ===" ;;
        --moe-r-only) MOE_R_ONLY=true; echo "=== MOE-R ONLY ===" ;;
    esac
done

# Configure for 8B model if requested
if $MODEL_8B; then
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    BATCH_SIZE=1
    GRAD_ACCUM=32
    GPU_MEM="48G"
    WANDB_PROJECT="loreft-regret-8b"
    OUTPUT_DIR="./outputs-8b"
fi

# Build position list
if $ALL_POSITIONS; then
    POSITIONS=("f1+s1" "f1+s3" "f1+s5" "f3+s3" "f5+s5")
fi

# Function to check if job is already done
is_done() {
    local run_name="$1"
    [[ -f "${OUTPUT_DIR}/${run_name}/training_args.json" ]]
}

# Function to check if job is running/pending in SLURM queue
is_running() {
    local job_name="$1"
    squeue -u $USER -n "$job_name" -h 2>/dev/null | grep -q .
}

# --- Create logs directory ---
mkdir -p logs

# --- Build intervention type list ---
MOE_TYPES=()
if ! $MOE_R_ONLY; then
    MOE_TYPES+=("moeloreft")
fi
if ! $MOE_W_ONLY; then
    MOE_TYPES+=("moerloreft")
fi

# --- Calculate total jobs ---
total_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]} * ${#NUM_EXPERTS_LIST[@]} * ${#MOE_TYPES[@]}))

echo "Submitting MoE sweep:"
echo "  Model: $MODEL"
echo "  Positions: ${POSITIONS[*]}"
echo "  MoE types: ${MOE_TYPES[*]}"
echo "  Num experts: ${NUM_EXPERTS_LIST[*]}"
echo "  Top-k: $TOP_K"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Output dir: $OUTPUT_DIR"
if $MODEL_8B; then
    echo "  GPU memory: $GPU_MEM (bs=$BATCH_SIZE, grad_accum=$GRAD_ACCUM)"
fi
echo "  Grid: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions x ${#NUM_EXPERTS_LIST[@]} expert configs x ${#MOE_TYPES[@]} types = $total_jobs jobs"
echo ""

job_count=0
skipped_count=0

# Model-specific prefix for job naming
if $MODEL_8B; then
    MODEL_PREFIX="8b_"
else
    MODEL_PREFIX=""
fi

# --- MoE sweep ---
for moe_type in "${MOE_TYPES[@]}"; do
    for num_experts in "${NUM_EXPERTS_LIST[@]}"; do
        echo ""
        echo "=== ${moe_type} (experts=$num_experts, top_k=$TOP_K) ==="

        for position in "${POSITIONS[@]}"; do
            # Determine if share_weights is needed
            SHARE_FLAG=""
            if [[ "$position" == "all" || "$position" == "alls" ]]; then
                SHARE_FLAG=",SHARE_WEIGHTS=true"
            fi

            for rank in "${RANKS[@]}"; do
                for lr in "${LRS[@]}"; do
                    pos_short=$(echo "$position" | sed 's/+//')
                    job_name="${MODEL_PREFIX}${moe_type}_r${rank}_e${num_experts}_${pos_short}_lr${lr}"
                    run_name="${moe_type}_r${rank}_e${num_experts}_k${TOP_K}___${position}___lr${lr}"

                    # Skip if already done or running
                    if $SKIP_DONE && is_done "$run_name"; then
                        echo "Skipping (done): $run_name"
                        skipped_count=$((skipped_count + 1))
                        continue
                    fi
                    if $SKIP_DONE && is_running "$job_name"; then
                        echo "Skipping (running): $job_name"
                        skipped_count=$((skipped_count + 1))
                        continue
                    fi

                    cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG},INTERVENTION_TYPE=$moe_type,NUM_EXPERTS=$num_experts,TOP_K=$TOP_K,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN scripts/sweep.sbatch"

                    if $DRY_RUN; then
                        echo "$cmd"
                    else
                        echo "Submitting ${moe_type}: rank=$rank, experts=$num_experts, position=$position, lr=$lr"
                        $cmd
                    fi

                    job_count=$((job_count + 1))
                done
            done
        done
    done
done

echo ""
echo "Submitted $job_count jobs"
if $SKIP_DONE; then
    echo "Skipped $skipped_count completed jobs"
fi
