# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import os
import re
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb
from pprint import pformat
import modeling_bart

import math

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        self.deepspeed = args.deepspeed

    def create_config(self):
        from transformers import  BartConfig

        if 'bart' in self.args.backbone:
            config_class = BartConfig
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone)

        args = self.args


        for k, v in vars(args).items():
            setattr(config, k, v)

        config.feat_dim = args.feat_dim
        config.pos_dim = args.pos_dim
        config.n_images = 2
        config.n_boxes = args.n_boxes
        config.n_image_tokens = args.n_image_tokens
        config.downsample = args.downsample
        config.oneddownsample = args.oneddownsample
        config.sparse_sample = args.sparse_sample

        config.mid_dim = args.mid_dim
        config.reduction_factor = args.reduction_factor

        config.use_hyperformer = args.use_hyperformer
        config.use_compacter = args.use_compacter

        tasks = re.split("[, ]+", args.tasks) # tranform to list
        config.adapter_config = None
        
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.losses = args.losses

        return config

    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model

    def print_trainable_params_percentage(self, model):

        orig_param_size = sum(p.numel() for p in model.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(model)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

        return percentage

    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False

    def partial_eval(self):
        # the purpose is to fix some of the norm statistics
        model = self.model

        def LM_LN_eval(model):
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval()

        def only_LN_eval(model):
            for name, sub_module in model.named_modules():
                if "visual_embedding" in name: # skip trainable parameters
                    continue
                if isinstance(sub_module, (nn.LayerNorm)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        def only_BN_eval(model):
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    # print(f"Change {name} to eval mode...")
                    sub_module.eval() # freeze the LN statistics and dropout

        if self.args.freeze_ln_statistics:
            only_LN_eval(model)

        if self.args.freeze_bn_statistics:
            only_BN_eval(model)

    def unfreeze_parameters(self):       


        targets = ["visual_embedding"]

        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")

        if self.args.unfreeze_language_model:
            targets = ["lm_head", "shared"]
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")
            for name, sub_module in self.model.named_modules():
                if isinstance(sub_module, (modeling_bart.JointEncoder, modeling_bart.BartDecoder)):
                    for param_name, param in sub_module.named_parameters():
                        print(f"{param_name} is trainable...")
                        param.requires_grad = True

        if self.args.unfreeze_lm_head:
            targets = ["lm_head", "shared"] # shared and lm_head share the same weight
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...")

        if self.args.unfreeze_bias:
            targets = ["bias"]
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                if any(t in n for t in targets):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")


        if self.args.unfreeze_encoder_layer_norms:
            target1 = "encoder."
            target2 = "layer_norm"
            target3 = "layernorm"
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                # if any(t in n for t in targets):
                if target1 in n and (target2 in n or target3 in n):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")

        if self.args.unfreeze_decoder_layer_norms:
            target1 = "decoder."
            target2 = "layer_norm"
            target3 = "layernorm"
            # unfreeze the parameters in targets anyway
            for n, p in self.model.named_parameters():
                # if any(t in n for t in targets):
                if target1 in n and (target2 in n or target3 in n):
                    p.requires_grad = True
                    print(f"{n} is trainable...({p.numel()})")



        for name, sub_module in self.model.named_modules():

            if self.args.unfreeze_layer_norms:
                if isinstance(sub_module, (nn.LayerNorm)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.unfreeze_batch_norms:
                if isinstance(sub_module, (nn.BatchNorm2d)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        print(self.model)
            
    def create_tokenizer(self, **kwargs):
        from transformers import BartTokenizer, BartTokenizerFast

        if 'bart' in self.args.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast

        tokenizer_name = self.args.backbone

        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs
            )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]

        if 'adamw' in self.args.optim:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optim = AdamW(optimizer_grouped_parameters,
                        lr=self.args.lr, eps=self.args.adam_eps)

        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = self.args.optimizer(optimizer_grouped_parameters, self.args.lr)

        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        if self.verbose:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print('Warmup ratio:', warmup_ratio)
            print("Warm up Iters: %d" % warmup_iters)

        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)

        if self.deepspeed:
            self.model.save_checkpoint(self.args.output, name)
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
