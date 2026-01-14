#!/bin/bash
# ============================================================
# Launch sweep for "ReFT Without Regret" experiments
# ============================================================
# This script submits a grid of jobs varying rank and LR
# 
# Usage (run from examples/regret/):
#   ./scripts/launch_sweep.sh              # Default: f1+s1 only, block_output
#   ./scripts/launch_sweep.sh --dry-run    # Print commands without submitting
#   ./scripts/launch_sweep.sh --skip-done  # Skip completed jobs
#   ./scripts/launch_sweep.sh --all-positions  # All positions (f1+l1, all, f1+s1, alls)
#   ./scripts/launch_sweep.sh --with-lora  # Include LoRA baseline
#   ./scripts/launch_sweep.sh --with-mlp   # Add mlp_activation experiments
#   ./scripts/launch_sweep.sh --all-components # All components (block_output, mlp_activation)
#   ./scripts/launch_sweep.sh --with-direft  # Include DiReFT experiments
#   ./scripts/launch_sweep.sh --with-nodireft # Include NoDiReFT experiments (no orthogonality)
#   ./scripts/launch_sweep.sh --with-moeloreft # Include MoE-LoReFT experiments
#   ./scripts/launch_sweep.sh --with-suffix-positions # Add suffix position experiments (s1, s3, s5, f1+s3, f1+s5)
#   ./scripts/launch_sweep.sh --rank1-only # Only rank=1 experiments
# ============================================================

set -e

# --- Sweep configuration ---
RANKS=(1 2 4 8 16 32 64)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)

# LoRA ranks (matching ReFT for fair comparison)
LORA_RANKS=(1 2 4 8 16 32 64)
LORA_LRS=(1e-4 2e-4 5e-4 1e-3 2e-3)

# Component options (for comparing intervention points)
COMPONENTS=("block_output")  # Default: residual stream only

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
WITH_LORA=false
ALL_POSITIONS=false
WITH_MLP=false
ALL_COMPONENTS=false
WITH_DIREFT=false
WITH_NODIREFT=false
WITH_MOELOREFT=false
WITH_SUFFIX_POSITIONS=false
RANK1_ONLY=false
MODEL_8B=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-done) SKIP_DONE=true; echo "=== SKIPPING COMPLETED JOBS ===" ;;
        --with-lora) WITH_LORA=true; echo "=== INCLUDING LORA BASELINE ===" ;;
        --all-positions) ALL_POSITIONS=true; echo "=== ALL POSITIONS MODE ===" ;;
        --with-mlp) WITH_MLP=true; echo "=== INCLUDING MLP_ACTIVATION EXPERIMENTS ===" ;;
        --all-components) ALL_COMPONENTS=true; echo "=== ALL COMPONENTS MODE ===" ;;
        --with-direft) WITH_DIREFT=true; echo "=== INCLUDING DIREFT EXPERIMENTS ===" ;;
        --with-nodireft) WITH_NODIREFT=true; echo "=== INCLUDING NODIREFT EXPERIMENTS ===" ;;
        --with-moeloreft) WITH_MOELOREFT=true; echo "=== INCLUDING MOELOREFT EXPERIMENTS ===" ;;
        --with-suffix-positions) WITH_SUFFIX_POSITIONS=true; echo "=== INCLUDING SUFFIX POSITION EXPERIMENTS ===" ;;
        --rank1-only) RANK1_ONLY=true; echo "=== RANK 1 ONLY ==="; RANKS=(1); LORA_RANKS=(1) ;;
        --model-8b) MODEL_8B=true; echo "=== LLAMA 3.1 8B MODE ===" ;;
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

# Add suffix position experiments if requested
if $WITH_SUFFIX_POSITIONS; then
    POSITIONS+=("s1" "s3" "s5" "f1+s3" "f1+s5")
fi

# --- Build component list ---
if $ALL_COMPONENTS; then
    COMPONENTS=("block_output" "mlp_activation")
elif $WITH_MLP; then
    COMPONENTS=("block_output" "mlp_activation")
else
    COMPONENTS=("block_output")
fi

# --- Calculate total jobs ---
loreft_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]} * ${#COMPONENTS[@]}))
direft_jobs=0
if $WITH_DIREFT; then
    direft_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]} * ${#COMPONENTS[@]}))
fi
nodireft_jobs=0
if $WITH_NODIREFT; then
    nodireft_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]} * ${#COMPONENTS[@]}))
fi
moeloreft_jobs=0
if $WITH_MOELOREFT; then
    moeloreft_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#POSITIONS[@]} * ${#COMPONENTS[@]}))
fi
lora_jobs=0
if $WITH_LORA; then
    lora_jobs=$((${#LORA_RANKS[@]} * ${#LORA_LRS[@]}))
fi
total_jobs=$((loreft_jobs + direft_jobs + nodireft_jobs + moeloreft_jobs + lora_jobs))

echo "Submitting sweep:"
echo "  Model: $MODEL"
echo "  Positions: ${POSITIONS[*]}"
echo "  Components: ${COMPONENTS[*]}"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Output dir: $OUTPUT_DIR"
if $MODEL_8B; then
    echo "  GPU memory: $GPU_MEM (bs=$BATCH_SIZE, grad_accum=$GRAD_ACCUM)"
fi
echo "  LoReFT: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions x ${#COMPONENTS[@]} components = $loreft_jobs jobs"
if $WITH_DIREFT; then
    echo "  DiReFT: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions x ${#COMPONENTS[@]} components = $direft_jobs jobs"
fi
if $WITH_NODIREFT; then
    echo "  NoDiReFT: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions x ${#COMPONENTS[@]} components = $nodireft_jobs jobs"
fi
if $WITH_MOELOREFT; then
    echo "  MoE-LoReFT: ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#POSITIONS[@]} positions x ${#COMPONENTS[@]} components = $moeloreft_jobs jobs"
fi
if $WITH_LORA; then
    echo "  LoRA: ${#LORA_RANKS[@]} ranks x ${#LORA_LRS[@]} LRs = $lora_jobs jobs"
fi
echo "  Total: $total_jobs jobs"
echo ""

job_count=0
skipped_count=0

# Model-specific prefix for job naming
if $MODEL_8B; then
    MODEL_PREFIX="8b_"
else
    MODEL_PREFIX=""
fi

# --- LoReFT sweep ---
for component in "${COMPONENTS[@]}"; do
    # Short name for component (for job naming)
    if [[ "$component" == "block_output" ]]; then
        comp_short=""  # Default, don't add to name
        COMPONENT_FLAG=""
    else
        comp_short="_${component}"
        COMPONENT_FLAG=",COMPONENT=$component"
    fi

    for position in "${POSITIONS[@]}"; do
        # Determine if share_weights is needed
        SHARE_FLAG=""
        if [[ "$position" == "all" || "$position" == "alls" ]]; then
            SHARE_FLAG=",SHARE_WEIGHTS=true"
        fi

        for rank in "${RANKS[@]}"; do
            for lr in "${LRS[@]}"; do
                pos_short=$(echo "$position" | sed 's/+//')
                job_name="${MODEL_PREFIX}loreft_r${rank}_${pos_short}${comp_short}_lr${lr}"

                # Add component to run_name if not block_output
                if [[ "$component" == "block_output" ]]; then
                    run_name="loreft_r${rank}___${position}___lr${lr}"
                else
                    run_name="loreft_r${rank}___${position}___${component}___lr${lr}"
                fi

                # Skip if already done
                if $SKIP_DONE && is_done "$run_name"; then
                    echo "Skipping (done): $run_name"
                    skipped_count=$((skipped_count + 1))
                    continue
                fi

                cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG}${COMPONENT_FLAG},MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN scripts/sweep.sbatch"

                if $DRY_RUN; then
                    echo "$cmd"
                else
                    echo "Submitting LoReFT: rank=$rank, position=$position, component=$component, lr=$lr"
                    $cmd
                fi

                job_count=$((job_count + 1))
            done
        done
    done
done

# --- DiReFT sweep ---
if $WITH_DIREFT; then
    echo ""
    echo "=== DiReFT Sweep ==="
    for component in "${COMPONENTS[@]}"; do
        # Short name for component (for job naming)
        if [[ "$component" == "block_output" ]]; then
            comp_short=""  # Default, don't add to name
            COMPONENT_FLAG=""
        else
            comp_short="_${component}"
            COMPONENT_FLAG=",COMPONENT=$component"
        fi

        for position in "${POSITIONS[@]}"; do
            # Determine if share_weights is needed
            SHARE_FLAG=""
            if [[ "$position" == "all" || "$position" == "alls" ]]; then
                SHARE_FLAG=",SHARE_WEIGHTS=true"
            fi

            for rank in "${RANKS[@]}"; do
                for lr in "${LRS[@]}"; do
                    pos_short=$(echo "$position" | sed 's/+//')
                    job_name="${MODEL_PREFIX}direft_r${rank}_${pos_short}${comp_short}_lr${lr}"

                    # Add component to run_name if not block_output
                    if [[ "$component" == "block_output" ]]; then
                        run_name="direft_r${rank}___${position}___lr${lr}"
                    else
                        run_name="direft_r${rank}___${position}___${component}___lr${lr}"
                    fi

                    # Skip if already done
                    if $SKIP_DONE && is_done "$run_name"; then
                        echo "Skipping (done): $run_name"
                        skipped_count=$((skipped_count + 1))
                        continue
                    fi

                    cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG}${COMPONENT_FLAG},INTERVENTION_TYPE=direft,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN scripts/sweep.sbatch"

                    if $DRY_RUN; then
                        echo "$cmd"
                    else
                        echo "Submitting DiReFT: rank=$rank, position=$position, component=$component, lr=$lr"
                        $cmd
                    fi

                    job_count=$((job_count + 1))
                done
            done
        done
    done
fi

# --- NoDiReFT sweep ---
if $WITH_NODIREFT; then
    echo ""
    echo "=== NoDiReFT Sweep ==="
    for component in "${COMPONENTS[@]}"; do
        # Short name for component (for job naming)
        if [[ "$component" == "block_output" ]]; then
            comp_short=""  # Default, don't add to name
            COMPONENT_FLAG=""
        else
            comp_short="_${component}"
            COMPONENT_FLAG=",COMPONENT=$component"
        fi

        for position in "${POSITIONS[@]}"; do
            # Determine if share_weights is needed
            SHARE_FLAG=""
            if [[ "$position" == "all" || "$position" == "alls" ]]; then
                SHARE_FLAG=",SHARE_WEIGHTS=true"
            fi

            for rank in "${RANKS[@]}"; do
                for lr in "${LRS[@]}"; do
                    pos_short=$(echo "$position" | sed 's/+//')
                    job_name="${MODEL_PREFIX}nodireft_r${rank}_${pos_short}${comp_short}_lr${lr}"

                    # Add component to run_name if not block_output
                    if [[ "$component" == "block_output" ]]; then
                        run_name="nodireft_r${rank}___${position}___lr${lr}"
                    else
                        run_name="nodireft_r${rank}___${position}___${component}___lr${lr}"
                    fi

                    # Skip if already done
                    if $SKIP_DONE && is_done "$run_name"; then
                        echo "Skipping (done): $run_name"
                        skipped_count=$((skipped_count + 1))
                        continue
                    fi

                    cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG}${COMPONENT_FLAG},INTERVENTION_TYPE=nodireft,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN scripts/sweep.sbatch"

                    if $DRY_RUN; then
                        echo "$cmd"
                    else
                        echo "Submitting NoDiReFT: rank=$rank, position=$position, component=$component, lr=$lr"
                        $cmd
                    fi

                    job_count=$((job_count + 1))
                done
            done
        done
    done
fi

# --- MoE-LoReFT sweep ---
if $WITH_MOELOREFT; then
    echo ""
    echo "=== MoE-LoReFT Sweep ==="
    for component in "${COMPONENTS[@]}"; do
        # Short name for component (for job naming)
        if [[ "$component" == "block_output" ]]; then
            comp_short=""  # Default, don't add to name
            COMPONENT_FLAG=""
        else
            comp_short="_${component}"
            COMPONENT_FLAG=",COMPONENT=$component"
        fi

        for position in "${POSITIONS[@]}"; do
            # Determine if share_weights is needed
            SHARE_FLAG=""
            if [[ "$position" == "all" || "$position" == "alls" ]]; then
                SHARE_FLAG=",SHARE_WEIGHTS=true"
            fi

            for rank in "${RANKS[@]}"; do
                for lr in "${LRS[@]}"; do
                    pos_short=$(echo "$position" | sed 's/+//')
                    job_name="${MODEL_PREFIX}moeloreft_r${rank}_${pos_short}${comp_short}_lr${lr}"

                    # Add component to run_name if not block_output
                    if [[ "$component" == "block_output" ]]; then
                        run_name="moeloreft_r${rank}___${position}___lr${lr}"
                    else
                        run_name="moeloreft_r${rank}___${position}___${component}___lr${lr}"
                    fi

                    # Skip if already done
                    if $SKIP_DONE && is_done "$run_name"; then
                        echo "Skipping (done): $run_name"
                        skipped_count=$((skipped_count + 1))
                        continue
                    fi

                    cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,RANK=$rank,LR=$lr,POSITION=$position${SHARE_FLAG}${COMPONENT_FLAG},INTERVENTION_TYPE=moeloreft,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN scripts/sweep.sbatch"

                    if $DRY_RUN; then
                        echo "$cmd"
                    else
                        echo "Submitting MoE-LoReFT: rank=$rank, position=$position, component=$component, lr=$lr"
                        $cmd
                    fi

                    job_count=$((job_count + 1))
                done
            done
        done
    done
fi

# --- LoRA sweep ---
if $WITH_LORA; then
    echo ""
    echo "=== LoRA Sweep ==="
    LORA_MODULES="q_proj;k_proj;v_proj;o_proj;gate_proj;up_proj;down_proj"

    for lora_rank in "${LORA_RANKS[@]}"; do
        for lr in "${LORA_LRS[@]}"; do
            job_name="${MODEL_PREFIX}lora_r${lora_rank}_lr${lr}"
            run_name="lora_r${lora_rank}___q+k+v+o+gate+up+down___lr${lr}"

            # Skip if already done
            if $SKIP_DONE && is_done "$run_name"; then
                echo "Skipping (done): $run_name"
                skipped_count=$((skipped_count + 1))
                continue
            fi

            cmd="sbatch $SBATCH_EXTRA --mem=$GPU_MEM --job-name=$job_name --export=ALL,MODEL=$MODEL,USE_LORA=true,DISABLE_REFT=true,LORA_RANK=$lora_rank,LORA_MODULES=$LORA_MODULES,LR=$lr,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR,USE_FLASH_ATTN=$USE_FLASH_ATTN,GRADIENT_CHECKPOINTING=true scripts/sweep.sbatch"
            
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
