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
-train_dataset mnli \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 10 \
-lr 1e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation_matched \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset sst2 \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 10 \
-lr 1e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset mrpc \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 20 \
-lr 2e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.00 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset cola \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 20 \
-lr 1e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model matthews_correlation \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset qnli \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 10 \
-lr 1e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset qqp \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 10 \
-lr 1e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset rte \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 20 \
-lr 5e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model accuracy \
--weight_decay 0.0001 \
--warmup_ratio 0.01 \
--logging_steps 20

python task_steer.py -task glue \
-train_dataset stsb \
-model FacebookAI/roberta-large \
-seed $RANDOM_SEED \
-l "1;3;5;7;9;11" \
-r 2 \
-p first \
-e 10 \
-lr 5e-3 \
-type ConditionedSourceLowRankIntervention \
-batch_size 32 \
-gradient_accumulation_steps 1 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--is_wandb \
--wandb_name wuzhengx \
--metric_for_best_model pearson \
--weight_decay 0.0000 \
--warmup_ratio 0.06 \
--logging_steps 20

