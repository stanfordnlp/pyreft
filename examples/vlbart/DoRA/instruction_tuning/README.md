# Finetuning LLaMA/LLaMA-2 with the cleaned Alpaca dataset using DoRA and DVoRA

Due to the license issues, we are currently unable to make the code available. Once we have resolved the problem, we will promptly release the code.

This directory will includes the DoRA/DVoRA implementation and guidelines for reproducing the results in our paper.

## Setup
1. Install dependencies
```bash
conda create -n dvora python=3.10
conda activate dvora
pip install -r requirements.txt
pip install -U ./peft
```

## Code Structure
Refer to `./peft/src/peft/tuners/dora.py` for the implementation of DoRA and DVoRA.

## Finetuning and Evaluation
### DoRA Finetuning
```
### LLaMA-7B
sh ./instruct/dora_llama_7b.sh
### LLaMA2-7B
sh ./instruct/dora_llama2_7b.sh
```
### DVoRA Finetuning
```
### LLaMA-7B
sh ./instruct/dvora_llama_7b.sh
### LLaMA2-7B
sh ./instruct/dvora_llama2_7b.sh
```
### Evaluation
After finetuning, the model responses to the MT-Bench questions will be saved under `./answers`, then you can use [Vicuna eval](https://github.com/lm-sys/vicuna-blog-eval) code to get the GPT-4 generated scores.

You can refer to `./answers` for the DoRA/DVoRA/LoRA/VeRA-finetuned model responses to the 80 MT-Bench questions, and directly use them for generating GPT-4 reviews.

## DoRA/DVoRA Result
| Model                 |  # Params(%)  |  Score |
|-----------------------|---------|--------|
| LLaMA-7B-LoRA		        |   2.31  |  5.1  |
| LLaMA-7B-DoRA 	        |  2.33 | **5.5**  |
| LLaMA-7B-VeRA 	        |    0.02 | 4.3 |
| LLaMA-7B-DVoRA		        |   0.04  | **5.0**  |
| LLaMA2-7B-LoRA		        |   2.31 |  5.7  |
| LLaMA2-7B-DoRA 	        |   2.33 | **6.0**  |
| LLaMA2-7B-VeRA 	        |    0.02 | 5.5  |
| LLaMA2-7B-DVoRA		        |   0.04 | **6.0**  |

## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [VeRA](https://openreview.net/attachment?id=NjNfLdxr3A&name=supplementary_material), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.



