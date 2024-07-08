import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
import numpy as np
def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    #_logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def initialize_clip(VISUAL_CONFIG, num_patches = 240, adapter_config=None):
    import clip
    clip_model, preprocess = clip.load(VISUAL_CONFIG.clip_model_name, jit=False, adapter_config=adapter_config)
    if VISUAL_CONFIG.clip_model_name == "ViT-B/32" and VISUAL_CONFIG.reset_pos_embedding:

        #from timm.models.vision_transformer import resize_pos_embed
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        pos_embed.weight = resize_pos_embed(clip_model.visual.positional_embedding.unsqueeze(0), pos_embed.unsqueeze(0))
        clip_model.visual.positional_embedding = pos_embed
        # model.visual.positional_embedding = model.visual.positional_embedding.to("cuda")
        #print(model.visual.positional_embedding.device)
        # pass
    if VISUAL_CONFIG.freeze_clip:
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
    return clip_model

def initialize_vit(VISUAL_CONFIG, model_type = "ViT-B_32", pretrained_dir = "data/ViT-B_32.npz", img_size = (384, 640), num_patches = 240):
    from vit.models.modeling import VisionTransformer, CONFIGS
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size = 224, zero_head=True, num_classes=1)
    model.load_from(np.load(pretrained_dir))

    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
    pos_embed.weight = resize_pos_embed(model.transformer.embeddings.position_embeddings, pos_embed.unsqueeze(0))
    model.transformer.embeddings.position_embeddings = pos_embed
    if VISUAL_CONFIG.freeze_clip:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model

def initialize_optimizer(visual_model, lr, momentum, weight_decay):
    optimizer = torch.optim.SGD(visual_model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    return optimizer

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.sgd_lr
    
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from torch.optim import Optimizer

class FusedOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        param_groups = []
        for optimizer in self.optimizers:
            param_groups += optimizer.param_groups
        #super(FusedOptimizer, self).__init__([], {})
        self.param_groups = param_groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
