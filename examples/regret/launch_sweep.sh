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
#   ./launch_sweep.sh --skip-lora  # Skip LoRA sweep (ReFT only)
#   ./launch_sweep.sh --rank1-only  # Only run rank=1 experiments (quick test)
#   ./launch_sweep.sh --scale-only  # Only run scale type experiments
#   ./launch_sweep.sh --strict-only # Only run strict mode positions (f1+s1, alls)
# ============================================================

set -e

# --- Sweep configuration ---
# ReFT ranks
RANKS=(1 2 4 8 16 32 64)
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)

# Scale types to test
SCALE_TYPES=(scalar sigmoid datadep token)

# LoRA ranks (matching ReFT for fair comparison)
LORA_RANKS=(1 2 4 8 16 32 64)
LORA_LRS=(1e-4 2e-4 5e-4 1e-3 2e-3)

# Target module sets for LoRA (Llama architecture)
# Format: "name:modules" - will iterate over these
LORA_MODULE_SETS=(
    "all:q_proj;k_proj;v_proj;o_proj;gate_proj;up_proj;down_proj"
)

# Other settings (modify as needed)
MAX_EXAMPLES=50000
EPOCHS=1
WANDB_PROJECT="loreft-regret"
OUTPUT_DIR="./outputs"

# --- Parse args ---
DRY_RUN=false
SKIP_DONE=false
SKIP_LORA=false
RANK1_ONLY=false
SCALE_ONLY=false
STRICT_ONLY=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN MODE ===" ;;
        --skip-done) SKIP_DONE=true; echo "=== SKIPPING COMPLETED JOBS ===" ;;
        --skip-lora) SKIP_LORA=true; echo "=== SKIPPING LORA SWEEP ===" ;;
        --rank1-only) RANK1_ONLY=true; echo "=== RANK 1 ONLY MODE ==="; RANKS=(1); LORA_RANKS=(1) ;;
        --scale-only) SCALE_ONLY=true; echo "=== SCALE TYPE SWEEP ONLY ===" ;;
        --strict-only) STRICT_ONLY=true; echo "=== STRICT POSITIONS ONLY (f1+s1, alls) ===" ;;
    esac
done

# Function to check if job is already done
is_done() {
    local run_name="$1"
    [[ -f "${OUTPUT_DIR}/${run_name}/training_args.json" ]]
}

# --- Create logs directory ---
mkdir -p logs

# --- Calculate total jobs ---
reft_legacy_jobs=$((${#RANKS[@]} * ${#LRS[@]} * 2))  # f1+l1 and all (legacy)
reft_strict_jobs=$((${#RANKS[@]} * ${#LRS[@]} * 2))  # f1+s1 and alls (strict)
reft_scale_jobs=$((${#RANKS[@]} * ${#LRS[@]} * ${#SCALE_TYPES[@]}))  # scale type sweep (f1+l1)
reft_jobs=$((reft_legacy_jobs + reft_strict_jobs + reft_scale_jobs))
num_module_sets=${#LORA_MODULE_SETS[@]}
lora_jobs=$((${#LORA_RANKS[@]} * ${#LORA_LRS[@]} * num_module_sets))
total_jobs=$((reft_jobs + lora_jobs))

# --- Submit jobs ---
echo "Submitting sweep:"
echo "  ReFT (legacy): ${#RANKS[@]} ranks x ${#LRS[@]} LRs x 2 positions = $reft_legacy_jobs jobs"
echo "  ReFT (strict): ${#RANKS[@]} ranks x ${#LRS[@]} LRs x 2 positions = $reft_strict_jobs jobs"
echo "  ReFT (scale types): ${#RANKS[@]} ranks x ${#LRS[@]} LRs x ${#SCALE_TYPES[@]} scale types = $reft_scale_jobs jobs"
echo "  LoRA: ${#LORA_RANKS[@]} ranks x ${#LORA_LRS[@]} LRs x $num_module_sets module sets = $lora_jobs jobs"
echo "  Total: $total_jobs jobs"
echo ""

job_count=0
skipped_count=0

# Skip standard sweeps if --scale-only
if $SCALE_ONLY; then
    echo "=== Skipping standard ReFT and LoRA sweeps (--scale-only) ==="
fi

# --- Standard f1+l1 position sweep (legacy) ---
if ! $SCALE_ONLY && ! $STRICT_ONLY; then
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

    # --- All positions sweep (legacy, with share_weights) ---
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
fi  # end legacy positions

# --- Strict f1+s1 position sweep (includes actual last token) ---
if ! $SCALE_ONLY; then
    echo ""
    echo "=== ReFT Strict Mode Sweep ==="
    for rank in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            job_name="loreft_r${rank}_f1s1_lr${lr}"
            run_name="r${rank}___f1+s1___lr${lr}"
            
            # Skip if already done
            if $SKIP_DONE && is_done "$run_name"; then
                echo "Skipping (done): $run_name"
                skipped_count=$((skipped_count + 1))
                continue
            fi
            
            cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=f1+s1,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
            
            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "Submitting: rank=$rank, position=f1+s1 (strict), lr=$lr"
                $cmd
            fi
            
            job_count=$((job_count + 1))
        done
    done

    # --- Strict alls positions sweep (includes actual last token, with share_weights) ---
    for rank in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            job_name="loreft_r${rank}_alls_lr${lr}"
            run_name="r${rank}___alls___lr${lr}"
            
            # Skip if already done
            if $SKIP_DONE && is_done "$run_name"; then
                echo "Skipping (done): $run_name"
                skipped_count=$((skipped_count + 1))
                continue
            fi
            
            cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=alls,SHARE_WEIGHTS=true,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
            
            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "Submitting: rank=$rank, position=alls (strict), lr=$lr"
                $cmd
            fi
            
            job_count=$((job_count + 1))
        done
    done
fi  # end if ! $SCALE_ONLY

# --- Scale Type Sweep (testing different gating mechanisms) ---
echo ""
echo "=== ReFT Scale Type Sweep ==="

# Determine positions based on --strict-only flag
if $STRICT_ONLY; then
    SCALE_POSITIONS=("f1+s1" "alls")
else
    SCALE_POSITIONS=("f1+l1")
fi

for scale_type in "${SCALE_TYPES[@]}"; do
    for position in "${SCALE_POSITIONS[@]}"; do
        # Determine if share_weights is needed (for "all" or "alls" positions)
        SHARE_FLAG=""
        if [[ "$position" == "all" || "$position" == "alls" ]]; then
            SHARE_FLAG=",SHARE_WEIGHTS=true"
        fi
        
        for rank in "${RANKS[@]}"; do
            for lr in "${LRS[@]}"; do
                pos_short=$(echo "$position" | sed 's/+//')
                job_name="loreft_r${rank}_${scale_type}_${pos_short}_lr${lr}"
                run_name="r${rank}_${scale_type}___${position}___lr${lr}"
                
                # Skip if already done
                if $SKIP_DONE && is_done "$run_name"; then
                    echo "Skipping (done): $run_name"
                    skipped_count=$((skipped_count + 1))
                    continue
                fi
                
                cmd="sbatch --job-name=$job_name --export=ALL,RANK=$rank,LR=$lr,POSITION=$position,SCALE_TYPE=$scale_type${SHARE_FLAG},MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
                
                if $DRY_RUN; then
                    echo "$cmd"
                else
                    echo "Submitting: rank=$rank, scale=$scale_type, position=$position, lr=$lr"
                    $cmd
                fi
                
                job_count=$((job_count + 1))
            done
        done
    done
done

# --- LoRA-only sweep (attention and MLP modules) ---
if ! $SKIP_LORA && ! $SCALE_ONLY; then
    echo ""
    echo "=== LoRA Sweep ==="
    for module_set in "${LORA_MODULE_SETS[@]}"; do
        # Parse "name:modules" format
        module_set_name="${module_set%%:*}"
        modules="${module_set#*:}"
        
        for lora_rank in "${LORA_RANKS[@]}"; do
            for lr in "${LORA_LRS[@]}"; do
                job_name="lora_r${lora_rank}_${module_set_name}_lr${lr}"
                # Build run_name to match what sweep.sbatch generates
                modules_short=$(echo "$modules" | sed 's/;/+/g' | sed 's/_proj//g')
                run_name="lora_r${lora_rank}___${modules_short}___lr${lr}"
                
                # Skip if already done
                if $SKIP_DONE && is_done "$run_name"; then
                    echo "Skipping (done): $run_name"
                    skipped_count=$((skipped_count + 1))
                    continue
                fi
                
                cmd="sbatch --job-name=$job_name --export=ALL,USE_LORA=true,DISABLE_REFT=true,LORA_RANK=$lora_rank,LORA_MODULES=$modules,LR=$lr,MAX_EXAMPLES=$MAX_EXAMPLES,EPOCHS=$EPOCHS,WANDB_PROJECT=$WANDB_PROJECT,OUTPUT_DIR=$OUTPUT_DIR sweep.sbatch"
                
                if $DRY_RUN; then
                    echo "$cmd"
                else
                    echo "Submitting: LoRA rank=$lora_rank, modules=$module_set_name, lr=$lr"
                    $cmd
                fi
                
                job_count=$((job_count + 1))
            done
        done
    done
else
    echo ""
    echo "=== Skipping LoRA Sweep ==="
fi

echo ""
echo "Submitted $job_count jobs"
if $SKIP_DONE; then
    echo "Skipped $skipped_count completed jobs"
fi

