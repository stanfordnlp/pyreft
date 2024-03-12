#!/bin/bash

# This script takes one argument: a random seed
# It then runs a Python script with a set of predefined options and the given seed.

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <random_seed>"
    exit 1
fi

# The random seed provided by the user
RANDOM_SEED=$1

python task_steer.py -task glue \
-train_dataset qnli \
-model FacebookAI/roberta-base \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 20 \
-lr 3e-3 \
-type ConditionedSourceLowRankIntervention \
-gradient_accumulation_steps 1 \
-batch_size 32 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0001 \
--warmup_ratio 0.06 \
--logging_steps 20 \
--add_bias

