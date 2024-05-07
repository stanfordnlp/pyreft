# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset boolq \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/boolq.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset piqa \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/piqa.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset social_i_qa \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset hellaswag \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/hellaswag.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset winogrande \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/winogrande.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-Challenge \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-Easy \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$3 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset openbookqa \
    --base_model 'yahma/llama-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/openbookqa.txt