# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG
from lxrt.adapters import AdapterConfig
from clip.adapter_config import VisionAdapterConfig


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers

from tools.load_stagte_dict import load_state_dict_flexible, load_state_dict_flexible_with_fp16

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        if args.use_adapter:
            adapter_config = AdapterConfig()
            adapter_config.tasks = ["vqa"]
            adapter_config.d_model = 768
            adapter_config.reduction_factor = args.reduction_factor
            adapter_config.use_gate = args.use_gate
        else:
            adapter_config = None

        if args.use_vis_adapter:
            vis_adapter_config = VisionAdapterConfig()
            vis_adapter_config.use_gate = args.use_gate
        else:
            vis_adapter_config = None

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode,
            adapter_config=adapter_config,
            vis_adapter_config=vis_adapter_config
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None, task=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(feats[0].device)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(feats[0].device)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(feats[0].device)

        output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask,
                            task=task)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        load_state_dict_flexible_with_fp16(self.model, state_dict)
        # Load weights to model
        #self.model.load_state_dict(state_dict, strict=False)




