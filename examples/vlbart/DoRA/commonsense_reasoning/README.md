<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Finetuning LLaMA on commonsense reasoning tasks using DoRA

This directory includes the DoRA implementation and guidelines for reproducing the results in our paper.

## Setup
1. Install dependencies
```bash
conda create -n dora_llama python=3.10
conda activate dora_llama
pip install -r requirements.txt
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## Code Structure

Refer to `./peft/src/peft/tuners/dora.py` for the implementation of DoRA.

Refer to `./finetune.py` for finetuning LLaMA using DoRA.

Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model.

## Finetuning and Evaluation

### Finetuning (`./llama_7B_Dora.sh`)
This file contains the code to finetune LLaMA-7B using DoRA. User can specify different DoRA configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha, the third argument indicates the destination for saving the fine-tuned model, and the last argument determines the GPU to use.
 
An example could be:
```
sh 7B_Dora.sh 32 64 ./finetuned_result/dora_r32 0
```

### Finetuning (`./llama_7B_Dora_qkv.sh`)
This file contains the code to finetune LLaMA-7B using DoRA but with more customizability, that is user can further specify which modules to only finetune the magnitude component of DoRA by changing `--Wdecompose_target_modules`, please refer to Sec. 5.6 in the paper for more details.

An example could be:
```
sh 7B_Dora_qkv.sh 32 64 ./finetuned_result/dora_qkv_r32 0
```

### Evaluation and DoRA weights

You can directly download the finetuned DoRA weights from [google drive](https://drive.google.com/drive/folders/1tFVtNcpfwdCLQTrHpP-1LJiq5jH3reUc?usp=sharing) and evaluate them with `llama_7B_Dora_eval.sh` as describe below to reproduce the result reported in the paper.

This file contains the code to evaluate LLaMA-7B finetuned with DoRA on the eight commonsense reasoning tasks. The first argument is the address of the DoRA weight, the second argument specifies where you would like to save the evaluation result, and the last argument determines which GPU to use.

An example could be:
```
sh 7B_Dora_eval.sh ./finetuned_result/dora_r32 ./finetuned_result/dora_r32 0
```

### Finetuning and Evaluating LLaMA2-7B & LLaMA3-8B 
This file contains the code to finetune LLaMA2-7B/LLaMA3-8B using DoRA. User can specify different DoRA configuration for finetuning. To be specific, the first argument denotes the rank r, the second argument specifies the corresponding alpha, the third argument indicates the destination for saving the fine-tuned model, and the last argument determines the GPU to use.
An example could be:
```
sh llama2_7B_DoRA_r.sh 32 64 ./finetuned_result/r32_lr2e-4 0
sh llama3_8B_DoRA_r.sh 32 64 ./finetuned_result/r32_lr1e-4 0
```
You can also directly download the finetuned DoRA weights from [google drive](https://drive.google.com/drive/folders/1tFVtNcpfwdCLQTrHpP-1LJiq5jH3reUc?usp=sharing) and evaluate them with `llama2_7B_Dora_eval.sh` and `llama3_8B_Dora_eval.sh` to reproduce the result reported in the paper.

## Accuracy comparison of LoRA and DoRA with varying ranks for LLaMA-7B on the commonsense reasoning tasks
| Model                 | r | lr |    BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|-------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| LLaMA-7B-LoRA		  |   4   | 3e-4 |     2.3 | 46.1 |18.3 |19.7| 55.2| 65.4| 51.9 | 57 | 39.5    |
| LLaMA-7B-LoRA		  |   8   | 3e-4 |    31.3 | 57.0  |  44.0 | 11.8 | 43.3 | 45.7 | 39.2 | 53.8 | 40.7     |
| LLaMA-7B-LoRA		  |   16  | 3e-4 |   69.9 | 77.8 | 75.1 | 72.1 | 55.8 | 77.1 | 62.2 | 78.0 | 70.9    |
| LLaMA-7B-LoRA		  |   32  |3e-4 |    67.5  |  80.8  |  78.2  |  83.4  |  80.4   |  78.0   |  62.6   |  79.1  |  76.3     |
| LLaMA-7B-LoRA		  |   64  |3e-4 |    66.7 | 79.1 | 75.7 | 17.6 | 78.8 | 73.3 | 59.6 | 75.2 | 65.8    |
| LLaMA-7B-DoRA 	  |  [4](https://drive.google.com/drive/folders/1JjFg66znyMEJqfcDuDC9joIOJu2biH61?usp=drive_link)    | 2e-4 |   51.3 | 42.2 | 77.8 | 25.4 | 78.8 | 78.7 | 62.5 | 78.6 | **61.9**   |
| LLaMA-7B-DoRA 	  |   [8](https://drive.google.com/drive/folders/1nf4JDSC9KhHUvxEeBfZjb6skZ5kubAIf?usp=drive_link)   | 2e-4 |    69.9 | 81.8 | 79.7 | 85.2 | 80.1 | 81.5 | 65.7 | 79.8 | **77.9**   |
| LLaMA-7B-DoRA		  |  [16](https://drive.google.com/drive/folders/1cKCXN168uv1bWkI00d20FvyVeZTMU8Ky?usp=drive_link)   | 2e-4 |   70.0 | 82.6 | 79.7 | 83.2 | 80.6 | 80.6 | 65.4 | 77.6 | **77.5**   |
| LLaMA-7B-DoRA 	  |  [32](https://drive.google.com/drive/folders/1Kz27h5BqNv3NOLdH2UhDf12C2JtwJe0Q?usp=drive_link)   | 1e-4 |   69.7 | 83.4 | 78.6 | 87.2 | 81.0 | 81.9 | 66.2 | 79.2 | **78.4**   |
| LLaMA-7B-DoRA		  |  [64](https://drive.google.com/drive/folders/1ts7TAUYlfHKHngUH4XTQiEFIIuxBJhrD?usp=drive_link)    | 2e-4 |   70.1 | 82.0 | 75.6 | 85.9 | 79.7 | 79.1 | 63.7 | 78.4 | **76.8**  |

## Accuracy comparison of LoRA and DoRA for LLaMA2-7B on the commonsense reasoning tasks
| Model                 | r | lr |    BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|-------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| LLaMA2-7B-LoRA		  |   32  |3e-4 |    69.8 | 79.9| 79.5| 83.6| 82.6| 79.8|64.7| 81.0| 77.6    |
| LLaMA2-7B-DoRA		  |  [16](https://drive.google.com/drive/folders/1lMn7WKLw5aQQqwnFnuDpsM3c9FsQtpl2?usp=drive_link)   | 2e-4 |   72.0 |83.1 |79.9| 89.1 |83.0| 84.5| 71.0 |81.2 |**80.5**  |
| LLaMA2-7B-DoRA 	  |  [32](https://drive.google.com/drive/folders/1x2qamDlNRgNtBBi-tPrZ3UTYXdObtskE?usp=drive_link)   | 2e-4 |   71.8 |83.7 |76.0 |89.1 |82.6 |83.7 |68.2| 82.4 |**79.7**   |
## Accuracy comparison of LoRA and DoRA for LLaMA3-8B on the commonsense reasoning tasks
| Model                 | r | lr |    BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|-------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| LLaMA3-8B-LoRA		  |   32  |3e-4 |    70.8 |85.2| 79.9| 91.7 |84.3 |84.2| 71.2| 79.0| 80.8    |
| LLaMA3-8B-DoRA		  |  [16](https://drive.google.com/drive/folders/1WHH_c5sGIdybPZt2Cuk0uEQrKtUOAk5v?usp=drive_link)   | 1e-4 |   74.5 |88.8 |80.3| 95.5| 84.7| 90.1| 79.1| 87.2| **85.0**   |
| LLaMA3-8B-DoRA 	  |  [32](https://drive.google.com/drive/folders/107-Qjf-odzG7q7uMonLy_ulwzhE5URgb?usp=drive_link)   | 1e-4 |   74.6| 89.3| 79.9 |95.5| 85.6| 90.5| 80.4 |85.8 |**85.2**  |
## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.


