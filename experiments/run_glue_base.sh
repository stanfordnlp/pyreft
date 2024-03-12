#!/bin/bash

# This script accepts one argument (a random seed) and passes it to other scripts to run one after another.

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <random_seed>"
    exit 1
fi

# The random seed provided by the user
RANDOM_SEED=$1

echo "Starting the script execution with random seed: $RANDOM_SEED"

# List of scripts to run, assuming these scripts are prepared to take a random seed as their first argument
script1="experiments/run_glue_base_cola.sh"
script2="experiments/run_glue_base_mnli.sh"
script3="experiments/run_glue_base_mrpc.sh"
script4="experiments/run_glue_base_qnli.sh"
script5="experiments/run_glue_base_qqp.sh"
script6="experiments/run_glue_base_rte.sh"
script7="experiments/run_glue_base_sst2.sh"
script8="experiments/run_glue_base_stsb.sh"
# Add more scripts as needed

# Make sure the scripts are executable
chmod +x "$script1" "$script2" "$script3" "$script4" "$script5" "$script6" "$script7" "$script8"

# Run each script one after another, passing the random seed as an argument
echo "Running script 1..."
"$script1" "$RANDOM_SEED"

echo "Running script 2..."
"$script2" "$RANDOM_SEED"

echo "Running script 3..."
"$script3" "$RANDOM_SEED"

echo "Running script 4..."
"$script4" "$RANDOM_SEED"

echo "Running script 5..."
"$script5" "$RANDOM_SEED"

echo "Running script 6..."
"$script6" "$RANDOM_SEED"

echo "Running script 7..."
"$script7" "$RANDOM_SEED"

echo "Running script 8..."
"$script8" "$RANDOM_SEED"

echo "All scripts have been executed."
