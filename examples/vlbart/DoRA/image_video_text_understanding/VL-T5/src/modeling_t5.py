from dataclasses import dataclass

from my_transformers.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
from transformers.activations import ACT2FN

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

from modeling_bart import Downsample, SparseSample, OneDDownsample

from adapters import (
    AdapterLayer, 
    AdapterController,
    OutputParallelAdapterLayer, 
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController,
    MetaLayersAdapterController
)

from adapters.hypercomplex.layers import PHMLinear

from prompt import (
    PromptController,
)

# from utils import *

logger = logging.get_logger(__name__)


class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images

        if self.config.individual_vis_layer_norm:

            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]

            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))

            for i in range(config.additional_visual_embedding_layers):
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))

                if self.config.use_vis_layer_norm:
                    feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))

            for i in range(config.additional_visual_embedding_layers):
                feat_embedding.append(nn.Linear(config.d_model, config.d_model))
                feat_embedding.append(nn.ReLU(True))

            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area


    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2) # [B, N, 1]
        pos = torch.cat([pos, area], dim=2) # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        # absolute_vis_pos_embedding = self.absolute_vis_pos_layer_norm(absolute_vis_pos_embedding)


        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) #.expand(B,-1)
            # assert obj_order_ids.max().item() < 32200, obj_order_ids
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding


class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, task_embed=None):
        super().__init__(config, embed_tokens, task_embed)
        self.config = config

        assert self.config.is_decoder is False

        self.visual_embedding = VisualEmbedding(self.config, embed_tokens)

        self.downsample = None
        self.sparse_sample = None
        if config.oneddownsample:
            self.downsample = OneDDownsample(config.n_boxes)
        elif config.downsample:
            sqrt_size = int(config.n_boxes ** 0.5)
            output_size = (sqrt_size, sqrt_size)
            self.downsample = Downsample(output_size)
        elif config.sparse_sample:
            self.sparse_sample = SparseSample(config.n_boxes)

        if config.encoder_prompt_config:
            self.prompt_modules = PromptController(config.encoder_prompt_config)
        else:
            self.prompt_modules = None

        self.init_weights()

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        return self.prefix_embedding(input_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.prompt_modules is not None:
            prefix_embeds = self.prompt_modules(inputs_embeds.shape[0], inputs_embeds.device, task)
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        B, L = inputs_embeds.size()[:-1]

        if self.downsample is not None:
            vis_inputs = self.downsample(vis_inputs)

        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(
            vis_feats, boxes, img_order_ids, obj_order_ids)

        if self.sparse_sample is not None:
            vis_embeds = self.sparse_sample(vis_embeds)

        V_L = vis_embeds.size(1)

        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if vis_attention_mask is None:
            vis_attention_mask = attention_mask.new_ones(B, V_L)

        if self.prompt_modules is not None:
            prefix_attention_mask = torch.ones(
                B, prefix_embeds.shape[1], dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (B, L+V_L),
            inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = L + V_L
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                L, L)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :L, :L] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, L:, :] = 0
            # position_bias[:, :, :, L:] = 0
            position_bias = position_bias + extended_attention_mask

            task_embedding = None
            if task is not None and self.task_embed is not None:
                task_embedding = self.task_embed(task)

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                block_adapters = None
                if self.adapter_layers_hyper_net:
                    block_adapters = self.adapter_layers_hyper_net(task_embedding, i)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    block_adapters=block_adapters,
                    task=task,
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        # super(T5ForConditionalGeneration, self).__init__(config)
        super().__init__(config)

        self.config = config

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = JointEncoder(encoder_config, self.shared, self.shared_task_embed)
        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        
        self.decoder = T5Stack(decoder_config, self.shared, self.shared_task_embed)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.decoder_prompt_config:
            self.prompt_modules = PromptController(config.decoder_prompt_config)
        else:
            self.prompt_modules = None

        if config.use_lm_head_adapter:
            self.output_adapter = OutputParallelAdapterLayer(config, self.model.shared.num_embeddings)

        adapter_config = config.adapter_config

        if adapter_config is not None:
            if getattr(adapter_config, "train_task_adapters", None) and getattr(adapter_config, "hypercomplex_adapters", None):
                if adapter_config.shared_phm_rule:
                    phm_dim = adapter_config.hypercomplex_division
                    self.factorized_phm_rule = adapter_config.factorized_phm_rule
                    if self.factorized_phm_rule:
                        self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                            requires_grad=adapter_config.learn_phm)
                        self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule_left.data.normal_(mean=0, std=adapter_config.phm_init_range)
                            self.phm_rule_right.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule_left.data.uniform_(-1, 1)
                            self.phm_rule_right.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError
                    else:
                        self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim),\
                            requires_grad=adapter_config.learn_phm)
                        if adapter_config.phm_c_init == "normal":
                            self.phm_rule.data.normal_(mean=0, std=adapter_config.phm_init_range)
                        elif adapter_config.phm_c_init == "uniform":
                            self.phm_rule.data.uniform_(-1, 1)
                        else:
                            raise NotImplementedError 
                    self.set_phm_rule()

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_phm_rule(self):
        def set_phm_rule(module):
            # TODO: we need to check there is one of these, and this is activated.
            for name, sub_module in module.named_modules():
                if isinstance(sub_module, PHMLinear):
                    if self.factorized_phm_rule:
                        sub_module.set_phm_rule(phm_rule_right=self.phm_rule_right, 
                                                phm_rule_left=self.phm_rule_left)
                    else:
                        sub_module.set_phm_rule(phm_rule=self.phm_rule)
        set_phm_rule(self.encoder)
        set_phm_rule(self.decoder)

    def get_prompt(self, bsz, device):
        input_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.prefix_embedding(input_tokens) # (B, L, d_model)

        temp_results = self.decoder(inputs_embeds=prefix_prompt, use_cache=True, return_dict=True)

        past_key_values = temp_results.past_key_values

        # past_key_values = list(past_key_values)

        # for layer in range(len(past_key_values)):
        #     past_key_values[layer] = list(past_key_values[layer])
        #     past_key_values[layer].append(None)
        #     past_key_values[layer].append(None)
        
        return past_key_values

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.vis_encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.vis_encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size


    # @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        task=None,
        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task=task,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device)

        if self.config.encoder_prompt_config is not None and self.config.encoder_prompt_config.prompt_len > 0:
            prefix_attention_mask = torch.ones(
                attention_mask.shape[0], 
                self.config.encoder_prompt_config.prompt_len, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device,
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        if self.prompt_modules is not None and past_key_values is None:
            prefix_embeds = self.prompt_modules(B, attention_mask.device, task)

            past_key_values = self.decoder(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True).past_key_values


        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
        )
        # print('decoder_outputs')
        # print(decoder_outputs)

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(
            #     lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

            # print('loss')
            # print(loss)

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            # vis_encoder_last_hidden_state=vis_encoder_outputs.last_hidden_state,
            # vis_encoder_hidden_states=vis_encoder_outputs.hidden_states,
            # vis_encoder_attentions=vis_encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
        )

    def vis_forward(self, batch, device):
        if hasattr(self, "vis_encoder"):
            # self.vis_encoder.eval() # freeze the batchnorm statistics
            images = batch["images"].to(device)

            if self.config.vis_pooling_output:
                _, vis_feats = self.vis_encoder(images)
            else:
                vis_feats, _ = self.vis_encoder(images)
            # vis_feats: (B, dim, L ** 0.5, L ** 0.5)
            B, L, D = vis_feats.shape
            vis_pos = torch.zeros(B, L, 4, dtype=vis_feats.dtype)

            batch["vis_feats"] = vis_feats
            batch["boxes"] = vis_pos

        return batch

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        if "task" in kwargs:
            output["task"] = kwargs["task"]

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs


@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # cross_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None


if __name__ == "__main__":
    import transformers
    import re

    from adapters import MetaAdapterConfig, AdapterConfig, CompactorConfig

    config = transformers.AutoConfig.from_pretrained("t5-base")

    config.feat_dim = 2048
    config.pos_dim = 4
    config.n_images = 2

    config.use_vis_order_embedding = True
    config.additional_visual_embedding_layers = 0
    config.preseqlen = 0
    config.decoder_preseqlen = 0

    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    config.use_vis_layer_norm = True
    config.individual_vis_layer_norm = True
    config.losses = 'lm,obj,attr,feat'
    config.tasks = "vqa,gqa"

    config.share_vis_lang_layer_norm = False
    config.classifier = False

    config.downsample = False
    config.sparse_sample = False

    config.add_layer_norm_before_adapter = True
    config.add_layer_norm_after_adapter = True

    config.use_lm_head_adapter = False

    config.use_hyperformer = False
    config.use_adapter = False
    config.use_compacter = True

    if config.use_hyperformer or config.use_adapter or config.use_compacter:

        assert config.use_hyperformer + config.use_adapter + config.use_compacter <= 1, "You can only at most one kind of adapters."
        if config.use_hyperformer:
            CONFIG_CLASS = MetaAdapterConfig
        elif config.use_adapter:
            CONFIG_CLASS = AdapterConfig
        elif config.use_compacter:
            CONFIG_CLASS = CompactorConfig

        config.adapter_config = CONFIG_CLASS()
        config.adapter_config.tasks = re.split("[, ]+", config.tasks) # tranform to list
        config.adapter_config.input_dim = 768
        config.adapter_config.d_model = 768
        config.adapter_config.use_single_adapter = True

    else:
        config.adapter_config = None

    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

    num_added_toks = 0
    additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
            [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config.default_obj_order_ids = tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

    model = VLT5.from_pretrained("t5-base", config=config)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    
    vis_feats = torch.randn(1, 36, 2048)
    vis_pos = torch.randn(1, 36, 4)

    generation_output = model.generate(
                **inputs,
                vis_inputs=(vis_feats, vis_pos),
                task="gqa"
    )
    
    print(generation_output)

    print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))


    orig_param_size = 222903552

    adapter_param = 0
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (AdapterController, TaskEmbeddingController, AdapterLayersHyperNetController, AdapterLayersOneHyperNetController, MetaLayersAdapterController)):
            print(f"{name} is trainable...")
            for param_name, param in sub_module.named_parameters():
                adapter_param += param.numel()

    visual_param = 0
    targets = ["visual_embedding", "prefix_embedding"]
    # unfreeze the parameters in targets anyway
    for n, p in model.named_parameters():
        if any(t in n for t in targets):
            visual_param += p.numel()
            print(f"{n} is trainable...")

    print(f"adapter params: {adapter_param}, {adapter_param / orig_param_size * 100:.3f} %")

    print(f"visual params: {visual_param}, {visual_param / orig_param_size * 100:.3f} %")

    total_param = adapter_param + visual_param
    print(f"total params: {total_param}, {total_param / orig_param_size * 100:.3f} %")
