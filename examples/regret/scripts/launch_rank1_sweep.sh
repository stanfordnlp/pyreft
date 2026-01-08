#!/bin/bash
# ============================================================
# Launch rank-1 sweep for intervention metrics analysis
# ============================================================
# Usage:
#   ./scripts/launch_rank1_sweep.sh              # Submit all jobs
#   ./scripts/launch_rank1_sweep.sh --dry-run    # Print commands only
# ============================================================

set -e

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true; echo "=== DRY RUN ===" ;;
    esac
done

# Scale types to test (empty string = baseline)
SCALE_TYPES=("" "scalar" "sigmoid" "datadep" "token")
LRS=(5e-4 1e-3 2e-3)
POSITIONS=("f1+s1")

# Create logs directory
mkdir -p logs

echo "=== Rank 1 Sweep ==="
echo "Scale types: ${SCALE_TYPES[*]}"
echo "LRs: ${LRS[*]}"
echo "Positions: ${POSITIONS[*]}"
echo ""

job_count=0

for scale_type in "${SCALE_TYPES[@]}"; do
    for lr in "${LRS[@]}"; do
        for position in "${POSITIONS[@]}"; do
            if [ -n "$scale_type" ]; then
                job_name="r1_${scale_type}_${position}_lr${lr}"
                export_vars="SCALE_TYPE=$scale_type,LR=$lr,POSITION=$position"
            else
                job_name="r1_baseline_${position}_lr${lr}"
                export_vars="LR=$lr,POSITION=$position"
            fi
            
            cmd="sbatch --job-name=$job_name --export=ALL,$export_vars scripts/rank1_sweep.sbatch"
            
            if $DRY_RUN; then
                echo "$cmd"
            else
                echo "Submitting: $job_name"
                $cmd
            fi
            
            job_count=$((job_count + 1))
        done
    done
done

echo ""
echo "Submitted $job_count jobs"
