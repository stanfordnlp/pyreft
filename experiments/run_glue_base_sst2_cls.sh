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
-train_dataset sst2 \
-model FacebookAI/roberta-base \
-seed $RANDOM_SEED \
-l "5;6;11+5;6;10" \
-r 2 \
-p first+last \
-e 6 \
-lr 9e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 1 \
-batch_size 32 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.00 \
--logging_steps 20 \
--allow_cls_grad \
--add_bias \
--dropout 0.05 \
--schedule constant
