# Reward modelling with ReFT

Reward models are trained to score how good a response is when conditioned on some user request. They are trained to assign higher reward to the better response given a pair of responses to the same requested (usually scored by humans). They are an important component of the RLHF pipeline.

Reward models are pretty expensive to train, so we tried to use LoReFT to finetune existing SFT LMs on the reward modelling task, i.e. we only tune a set of LoReFT interventions along with the single-class classification head on the last token. We replicated the training pipeline from [WeiXiongUST/RLHF-Reward-Modeling](https://github.com/WeiXiongUST/RLHF-Reward-Modeling/tree/main), which has an excellent associated writeup: [Xiong et al. (2024)](https://efficient-unicorn-451.notion.site/Reward-Modeling-for-RLHF-abe03f9afdac42b9a5bee746844518d0).

## Training

We use the following command to finetune Google's [`gemma-2b-it`](https://huggingface.co/google/gemma-2b-it) on the reward modelling objective using the trainset of the pairwise preference dataset [`llm-blender/Unified-Feedback`](llm-blender/Unified-Feedback).

```bash
python train.py --output_dir [output_dir] --wandb_entity [username] --per_device_train_batch_size 8  --per_device_eval_batch_size 8 --gradient_accumulation_steps 32 --logging_steps 20 --model_name_or_path google/gemma-2b-it --num_train_epochs 1 --position f1+l1
```

This model achieves an accuracy of **0.67575** on the evaluation set, training for one epoch with effective batch size of 256 in ~21 hours on a single A100 40G. You can see the W&B logs [here](https://wandb.ai/aryamanarora/reft-reward/runs/qwwrl0p9/overview).

We're still running evals + some more hparam tuning, but note that the eval acc of [Mistral-7B reward model](https://huggingface.co/Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback) trained on the same dataset is 0.7740. We will scale up to 7B as well.