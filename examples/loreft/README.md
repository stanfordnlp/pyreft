# LoReFT examples

Based on the script [`train.py`](https://github.com/stanfordnlp/pyreft/blob/main/examples/loreft/train.py).

This directory contains all the files needed to reproduce our paper results. We use random seeds `seed_set = {42,43,44,45,46}` throughout. For non-GLUE tasks, we use the first three seeds from the list.

**Note that ReFT only supports a single GPU for now - make sure you set `CUDA_VISIBLE_DEVICES=0` or something equivalent! We are working on supporting multi-GPU right now.**

## Datasets

To load all of our used datasets run:

```bash
bash load_datasets.sh
```

We copy everything from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main) for the commonsense and math reasoning dataset setup. We use a parsed version of [Ultrafeedback dataset](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) for instruct-tuning. Specifically, we get:

- Training data for commonsense and math reasoning:
  - [`commonsense_170k.json`](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)
  - [`math_10k.json`](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)

- Evaluation data for commonsense and math reasoning are included in:
  - [`LLM-Adapters/dataset`](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset)

- For instrution following training:
  - [`train.json`](https://github.com/frankaging/ultrafeedback-dataset/blob/main/train.json)

## Before you start to replicate our results

`pyreft` is built after we aggregate all of our results. All of our results are based on our original source code under the folder [`original_code`](https://github.com/stanfordnlp/pyreft/blob/main/examples/loreft/original_code). If you want to replicate our results more closely, feel free to read the README in the original source code folder.


## Hyperparameter tuning

As described in our Appendix C in the paper, we are using the last 300 examples from  the GSM8K training set for hyperparameter tuning. Here is an example of our running command:

```bash
python train.py -task gsm8k \
-model yahma/llama-7b-hf \
-seed 42 -l all -r 4 -p f7+l7 -e 12 -lr 9e-4 \
-type NodireftIntervention \
-gradient_accumulation_steps 4 \
-batch_size 8 \
-eval_batch_size 4 \
--dropout 0.05 \
--test_split validation \
--use_normalized_template \
--greedy_decoding \
--warmup_ratio 0.00 \
--weight_decay 0.06
```

We pick the best hyperparameter settings, and train on our commonsense reasoning as well as arithmetic tasks.

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
-data_dir dataset \
-model meta-llama/Llama-2-7b-hf \
-seed 42 -l "3;9;18;24" -r 4 -p f5+l5 -e 9 -lr 9e-4 \
-type LoreftIntervention \
-gradient_accumulation_steps 32 \
-batch_size 4 \
-eval_batch_size 2 \
--test_split test \
--use_normalized_template \
--max_length 768
```

Note that `max_length` has to set to 768 to ensure a fair comparison, since our work and previous baselines are run using this constraint. Please note that this might hurt overall `Alpaca-Eval` scores, especially for those evaluators preferring longer generations. We are also happy to release our wandb logs for our runs reported in the paper: [Ultrafeedback Results](https://wandb.ai/wuzhengx/ReFT_MuadDib_ultrafeedback). Note that the evaluation is done offline. The running time includes the generation time. Training time is about 18 mins on a single A100 for our 1K ablation study.

### Offline evaluation with [Alpaca-Eval v1.0](https://github.com/tatsu-lab/alpaca_eval/)

We use [Alpaca-Eval v1.0](https://github.com/tatsu-lab/alpaca_eval/) to automatically evaluate our instruct-tuned model with GPT-4. To evaluate, you firstn need to install Alpaca-Eval:
```bash
pip install alpaca-eval
```

Since we evaluate against `text-davinci-003`, you need to first [download its generation](https://github.com/tatsu-lab/alpaca_eval/blob/main/results/text_davinci_003/model_outputs.json) provided in the Alpaca-Eval repo. Please make sure you have an OpenAI account, and get your OpenAI API key. Now, you can run the commend to evaluate:
```bash
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>

alpaca_eval --model_outputs <OUTPUT_DIR>/alpaca_eval_test_outputs.json --annotators_config alpaca_eval_gpt4 --reference_outputs <OPENAI_OUTPUT_DIR>/text_davinci_003/model_outputs.json
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

## Using LoReFT with LoRA

If you have `peft` library installed locally, you can also train with LoReFT together with LoRA in parallel:

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
--greedy_decoding \
--use_lora
```

You only need to add `--use_lora` flag to enable this. Feel free to look at the code for our implementation.


