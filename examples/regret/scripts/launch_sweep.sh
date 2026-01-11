#!/bin/bash
# ============================================================
# Launch sweep for "ReFT Without Regret" experiments
# ============================================================
# This script submits a grid of jobs varying rank and LR
# 
# Usage (run from examples/regret/):
#   ./scripts/launch_sweep.sh              # Default: f1+s1 only
#   ./scripts/launch_sweep.sh --dry-run    # Print commands without submitting
#   ./scripts/launch_sweep.sh --skip-done  # Skip completed jobs
#   ./scripts/launch_sweep.sh --all-positions  # All positions (f1+l1, all, f1+s1, alls)
#   ./scripts/launch_sweep.sh --with-lora  # Include LoRA baseline
#   ./scripts/launch_sweep.sh --rank1-only # Only rank=1 experiments
# ============================================================

set -e

# --- Sweep configuration ---
RANKS=(1 2 4 8 16 32 64)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)

# LoRA ranks (matching ReFT for fair comparison)
LORA_RANKS=(1 2 4 8 16 32 64)
LORA_LRS=(1e-4 2e-4 5e-4 1e-3 2e-3)

# Other settings
MAX_EXAMPLES=50000
EPOCHS=1
WANDB_PROJECT="loreft-regret"
OUTPUT_DIR="./outputs"

# --- Parse args ---
DRY_RUN=false
SKIP_DONE=false
WITH_LORA=false
ALL_POSITIONS=false
RANK1_ONLY=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-done) SKIP_DONE=true; echo "=== SKIPPING COMPLETED JOBS ===" ;;
        --with-lora) WITH_LORA=true; echo "=== INCLUDING LORA BASELINE ===" ;;
        --all-positions) ALL_POSITIONS=true; echo "=== ALL POSITIONS MODE ===" ;;
        --rank1-only) RANK1_ONLY=true; echo "=== RANK 1 ONLY ==="; RANKS=(1); LORA_RANKS=(1) ;;
    esac
done

# Function to check if job is already done
is_done() {
    local run_name="$1"
    [[ -f "${OUTPUT_DIR}/${run_name}/training_args.json" ]]
}

# --- Create logs directory ---
mkdir -p logs

# --- Build position list ---
if $ALL_POSITIONS; then
    POSITIONS=("f1+s1" "alls" "f1+l1" "all")
else
    POSITIONS=("f1+s1")
fi

# --- Calculate total jobs ---
reft_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]}))
lora_jobs=0
if $WITH_LORA; then
    lora_jobs=$((${#LORA_RANKS[@]} * ${#LORA_LRS[@]}))
fi
total_jobs=$((reft_jobs + lora_jobs))

echo "Submitting sweep:"
echo "  Positions: ${POSITIONS[*]}"
echo "  ReFT: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions = $reft_jobs jobs"
if $WITH_LORA; then
    echo "  LoRA: ${#LORA_RANKS[@]} ranks x ${#LORA_LRS[@]} LRs = $lora_jobs jobs"
fi
echo "  Total: $total_jobs jobs"
echo ""

job_count=0
skipped_count=0

# --- ReFT sweep ---
for position in "${POSITIONS[@]}"; do
    # Determine if share_weights is needed
    SHARE_FLAG=""
    if [[ "$position" == "all" || "$position" == "alls" ]]; then
        SHARE_FLAG=",SHARE_WEIGHTS=true"
    fi
    
    for rank in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            pos_short=$(echo "$position" | sed 's/+//')
            job_name="loreft_r${rank}_${pos_short}_lr${lr}"
            run_name="r${rank}___${position}___lr${lr}"
            
            # Skip if already done
            if $SKIP_DONE && is_done "$run_name"; then
                echo "Skipping (done): $run_name"
                skipped_count=$((skipped_count + 1))
                continue
            fi
            
            cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG},MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR scripts/sweep.sbatch"
            
            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "Submitting: rank=$rank, position=$position, lr=$lr"
                $cmd
            fi
            
            job_count=$((job_count + 1))
        done
    done
done

# --- LoRA sweep ---
if $WITH_LORA; then
    echo ""
    echo "=== LoRA Sweep ==="
    LORA_MODULES="q_proj;k_proj;v_proj;o_proj;gate_proj;up_proj;down_proj"
    
    for lora_rank in "${LORA_RANKS[@]}"; do
        for lr in "${LORA_LRS[@]}"; do
            job_name="lora_r${lora_rank}_lr${lr}"
            run_name="lora_r${lora_rank}___all___lr${lr}"
            
            # Skip if already done
            if $SKIP_DONE && is_done "$run_name"; then
                echo "Skipping (done): $run_name"
                skipped_count=$((skipped_count + 1))
                continue
            fi
            
            cmd="sbatch --job-name=$job_name --export=ALL,USE_LORA=true,DISABLE_REFT=true,LORA_RANK=$lora_rank,LORA_MODULES=$LORA_MODULES,LR=$lr,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR scripts/sweep.sbatch"
            
            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "Submitting: LoRA rank=$lora_rank, lr=$lr"
                $cmd
            fi
            
            job_count=$((job_count + 1))
        done
    done
fi

echo ""
echo "Submitted $job_count jobs"
if $SKIP_DONE; then
    echo "Skipped $skipped_count completed jobs"
fi
