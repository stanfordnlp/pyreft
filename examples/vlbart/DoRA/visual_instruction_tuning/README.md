# Visual instruction tuning of LLaVA-V1.5 using DoRA

This directory includes the DoRA implementation and guidelines for reproducing the results in our paper.

## Setup
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install -U ./peft
```

Download the pretrained projector weights from [liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/tree/main) and put it under `./checkpoints/`

## Data
### Finetuning Dataset
Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100
``` 

### Evaluation Dataset
**First download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**, and extract to `./playground/data/eval`. 

#### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.

#### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.

#### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`.

#### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).

#### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.

#### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`.

#### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.

## Finetuning and Evaluation
### Finetuning using DoRA
Example usage for multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./Dora_7b.sh
```
### Evaluation and DoRA weights

You can directly download the finetuned DoRA weights from [google drive](https://drive.google.com/drive/folders/1NQZTX-axmXZcWpSh5yJrBGjFoJXsw9YE) and evaluate it following the descrption below to reproduce the result of the paper.

Example usage for multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash 7B_eval_dora.sh llava-v1.5-7b-dora-release ABSOLUTE_PATH/visual_instruction_tuning/checkpoints/llava-v1.5-7b-dora-release ABSOLUTE_PATH/LLaVA/playground/data/eval ABSOLUTE_PATH/visual_instruction_tuning/eval_result/llava-v1.5-7b-dora-release ABSOLUTE_PATH/visual_instruction_tuning
```
The first argument denotes the name of the folder where you keep the finetuned DoRA weights, the second argument specifies the absolute path to the DoRA weights, the third argument indicates the path to the evaluation datasets, the fourth argument is the path where you want to store the evaluation results, and the last argument is the absolute path to `./visual_instruction_tuning`.

#### Results that required submission to online server for evaluation
1. VQAv2: Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `ABSOLUTE_PATH/visual_instruction_tuning/eval_result/llava-v1.5-7b-dora-release/vqav2/answers_upload`.
2. VisWiz: Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `ABSOLUTE_PATH/visual_instruction_tuning/eval_result/llava-v1.5-7b-dora-release/vizwiz/answers_upload`.
3. MMBench: Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `ABSOLUTE_PATH/visual_instruction_tuning/eval_result/llava-v1.5-7b-dora-release/mmbench/answers_upload`.

## DoRA Result
### Visual instruction tuning evaluation result of DoRA, LoRA, and FT for LLaVA-1.5-7B on a wide range of 7 vision-language
tasks.
| Method                |  # Params (%) | VQAv2 | GQA | VisWiz | SQA | VQAT | POPE | MMBench | Avg  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|---------|---------|---------|
| FT | 100 | 78.5 | 61.9 | 50.0 | 66.8 | 58.2 | 85.9 | 64.3 | 66.5|
| LoRA | 4.61 | 79.1 | 62.9 | 47.8 | 68.4 | 58.2 | 86.4| 66.1 | 66.9|
| DoRA | 4.63 | 78.6 | 62.9 | 52.2 | 69.9 | 57 | 87.2 | 66.1 | 67.6 |

## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [LLaVA](https://github.com/haotian-liu/LLaVA), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.