# LoReFT examples

Based on the script [`train.py`](https://github.com/frankaging/pyreft/blob/main/examples/loreft/train.py).

This directory contains all the files needed to reproduce our paper results. We use random seeds `seed_set = {42,43,44,45,46}` throughout. For non-GLUE tasks, we use the first three seeds from the list.

## Datasets

To run these commands, you need to download required datasets. We copy everything from [LLM-Adaptors](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main) for the dataset setup.

## Commonsense reasoning tasks

Commonsense reasoning is made up of a total of 8 different tasks. Here is how to run the script to train once and evaluate on all of them:

```bash
python train.py -task commonsense \
-data_dir <your_dataset_folder_path> \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p f7+l7 -e 6 -lr 9e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 2 \
-batch_size 16 \
-eval_batch_size 4 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding
```

We are also happy to release our wandb logs for our runs reported in the paper: [Commonsense Results](https://wandb.ai/wuzhengx/ReFT_MuadDib_commonsense). It contains all the commands, GPU stats, running time, etc..

## Math reasoning tasks

Similar to commonsense reasoning, math reasoning is made up of different tasks. Here is how to run the script to train once and evaluate on all of them:

```bash
python train.py -task math \
-data_dir <your_dataset_folder_path> \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p f7+l7 -e 12 -lr 9e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 2 \
-batch_size 16 \
-eval_batch_size 4 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--warmup_ratio 0.1 \
--greedy_decoding
```

Note that we only change the number of epoch here. We are also happy to release our wandb logs for our runs reported in the paper: [Math Results](https://wandb.ai/wuzhengx/ReFT_MuadDib_math).

## Instruction following tasks

We finetune our base LMs with the Ultrafeedback dataset:

```bash
python train.py -task ultrafeedback \
-data_dir <your_dataset_folder_path> \
-model meta-llama/Llama-2-7b-hf \
-seed 44 -l 3;9;18;24 -r 4 -p f5+l5 -e 9 -lr 9e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 32 \
-batch_size 4 \
-eval_batch_size 2 \
--test_split test \
--use_normalized_template \
--max_length 768
```


## GLUE tasks

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
python train.py -task glue \
-train_dataset cola \
-model FacebookAI/roberta-base \
-seed 42 \
-l all \
-r 1 \
-p f1 \
-e 3 \
-lr 6e-4 \
-type ConditionedSourceLowRankRotatedSpaceIntervention \
-gradient_accumulation_steps 1 \
-batch_size 32 \
-eval_batch_size 32 \
-test_split validation \
-max_length 256 \
--metric_for_best_model matthews_correlation \
--dropout 0.05 \
--weight_decay 0.00000 \
--warmup_ratio 0.09 \
--logging_steps 20 \
--allow_cls_grad
```



