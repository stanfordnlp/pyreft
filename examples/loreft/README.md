# LoReFT examples

Based on the script [`train.py`](https://github.com/stanfordnlp/pyreft/blob/main/examples/loreft/train.py).

This directory contains all the files needed to reproduce our paper results. We use random seeds `seed_set = {42,43,44,45,46}` throughout. For non-GLUE tasks, we use the first three seeds from the list.

## Datasets

To load the datasets run:

```bash
bash load_datasets.sh
```

We copy everything from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main) for the dataset setup. Specifically, we get:

- Training data for commonsense and math reasoning:
  - [`commonsense_170k.json`](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)
  - [`math_10k.json`](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)

- Evaluation data for commonsense and math reasoning are included in:
  - [`LLM-Adapters/dataset`](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset)

- For instrution following training and evaluation, everything is done through HuggingFace hub. Note that we did not create our own dataset, instead we took previous ones to ensure a fair comparison.

## Commonsense reasoning tasks

Commonsense reasoning is made up of a total of 8 different tasks. Here is how to run the script to train once and evaluate on all of them:

```bash
python train.py -task commonsense \
-data_dir dataset \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p f7+l7 -e 6 -lr 9e-4 \
-type LoreftIntervention \
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
-data_dir dataset \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p f7+l7 -e 12 -lr 9e-4 \
-type LoreftIntervention \
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
-type LoreftIntervention \
-gradient_accumulation_steps 32 \
-batch_size 4 \
-eval_batch_size 2 \
--test_split test \
--use_normalized_template \
--max_length 768
```

Note that `max_length` has to set to 768 to ensure a fair comparison, since our work and previous baselines are run using this constraint. Please note that this might hurt overall `Alpaca-Eval` scores, especially for those evaluators preferring longer generations. We are also happy to release our wandb logs for our runs reported in the paper: [Ultrafeedback Results](https://wandb.ai/wuzhengx/ReFT_MuadDib_ultrafeedback). Note that the evaluation is done offline. The running time includes the generation time. Training time is about 18 mins on a single A100 for our 1K ablation study.

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
-type LoreftIntervention \
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



