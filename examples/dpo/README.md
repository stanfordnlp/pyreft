# Representation Fine-Tuning for Direct Preference Optimization

This is a tutorial for using ReFT with the [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) objective. 

Follow the [`dpo.ipynb`](dpo.ipynb) notebook for a walk-through of training a ReFT model with DPO to answer questions truthfully based on the [TruthfulQA](https://arxiv.org/abs/2109.07958) dataset.

The DPO ReFT trainer is based on the DPOTrainer implementation in the `trl` library (note: please use v0.8.6). The adapted trainer is implemented in [`dpo_trainer.py`](dpo_trainer.py).

We also provide a python script, [`dpo.py`](dpo.py), which you can run instead of the notebook tutorial. To see the W&B logs from running `dpo.py` with default parameters, see the [project here](https://wandb.ai/amirzur1212/reft_dpo/workspace).