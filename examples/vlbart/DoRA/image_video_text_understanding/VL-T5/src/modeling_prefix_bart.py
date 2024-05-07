
import math
import random
from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    BartPretrainedModel,
    BartConfig,
    ACT2FN,
    shift_tokens_right, _make_causal_mask, _expand_mask
)

from my_transformers.modeling_bart import BartModel, BartForConditionalGeneration, BartDecoder, BartEncoder
from modeling_bart import VisualEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

logger = logging.get_logger(__name__)


def _make_causal_mask(input_ids_shape: torch.Size, prefix_len: int, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    tgt_len = tgt_len + prefix_len
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class JointEncoder(BartEncoder):
    """
    BartEncoder + visual embedding
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.config = config

        self.visual_embedding = VisualEmbedding(config, self.embed_tokens)

        if config.preseqlen > 0:
            self.encoder_prefix_tokens = torch.arange(config.preseqlen).long()
            self.encoder_prefix_embedding = nn.Sequential(
                nn.Embedding(config.preseqlen, config.d_model),
                nn.Linear(config.d_model, config.mid_dim),
                nn.Tanh(),
                nn.Linear(config.mid_dim, config.encoder_layers * config.d_model * 2),
            )
        else:
            self.encoder_prefix_embedding = None

        self.init_weights()

    def get_prompt(self, bsz, device):
        input_tokens = self.encoder_prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        prefix_prompt = self.encoder_prefix_embedding(input_tokens) # (B, L, DIM)

        prefix_prompt = prefix_prompt.view(
            2 * self.config.encoder_layers, 
            bsz,
            self.config.encoder_attention_heads,
            self.config.preseqlen,
            -1,
        ).split(2)

        # print(len(prefix_prompt))

        # for idx in range(len(prefix_prompt)):
        #     print(prefix_prompt[idx].shape)
        
        return prefix_prompt

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        past_key_values=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        inputs_embeds = inputs_embeds + embed_pos

        B, L = inputs_embeds.size()[:-1]

        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(vis_feats, boxes, img_order_ids, obj_order_ids)
        V_L = vis_embeds.size(1)

        if self.config.share_vis_lang_layer_norm:
            inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

            inputs_embeds = self.layernorm_embedding(inputs_embeds)
        else:
            inputs_embeds = self.layernorm_embedding(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if self.config.preseqlen > 0:
            prefix_attention_mask = torch.ones(
                attention_mask.shape[0], self.config.preseqlen, dtype=attention_mask.dtype, device=attention_mask.device
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if vis_attention_mask is None:
            vis_attention_mask = torch.ones(B, V_L, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # print('attention_mask, ', attention_mask.size())
        # print('vis_attention_mask, ', vis_attention_mask.size())

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

            attention_mask = attention_mask[:, :, self.config.preseqlen:]

        # print('ext_attention_mask, ', attention_mask.size())
        # print('attention_mask')
        # print(attention_mask.size())
        # print(attention_mask)

        # compute prefix
        if past_key_values is None:
            past_key_values = self.get_prompt(B, inputs_embeds.device)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            # for prefix
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        past_key_value,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, past_key_value, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class VLBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.preseqlen = config.preseqlen
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        #----- Modified-----#
        # self.encoder = BartEncoder(config, self.shared)
        self.encoder = JointEncoder(config, self.shared)
        #-------------------#
        self.decoder = BartDecoder(config, self.shared)

        self.config = config

        if config.decoder_preseqlen > 0:
            self.decoder_prefix_tokens = torch.arange(config.decoder_preseqlen).long()
            self.decoder_prefix_embedding = nn.Sequential(
                nn.Embedding(config.decoder_preseqlen, config.d_model),
                nn.Linear(config.d_model, config.mid_dim),
                nn.Tanh(),
                nn.Linear(config.mid_dim, config.decoder_layers * config.d_model * 2),
            )

            # self.cross_prefix_tokens = torch.arange(config.decoder_preseqlen).long()
            # self.cross_prefix_embedding = nn.Sequential(
            #     nn.Embedding(config.decoder_preseqlen, config.d_model),
            #     nn.Linear(config.d_model, config.d_model),
            #     nn.Tanh(),
            #     nn.Linear(config.d_model, config.decoder_layers * config.d_model * 2),
            # )
        else:
            self.decoder_prefix_embedding = None

        self.init_weights()

    def get_prompt(self, bsz, device):
        input_tokens = self.decoder_prefix_tokens.unsqueeze(0).expand(bsz, -1).to(device) # (B, L)
        
        self_prefix_prompt = self.decoder_prefix_embedding(input_tokens) # (B, L, DIM)
        self_prefix_prompt = self_prefix_prompt.view(
            2 * self.config.decoder_layers, 
            bsz,
            self.config.decoder_attention_heads,
            self.config.decoder_preseqlen,
            -1,
        ).split(2)

        return self_prefix_prompt

        # cross_prefix_prompt = self.cross_prefix_embedding(input_tokens) # (B, L, DIM)
        # cross_prefix_prompt = cross_prefix_prompt.view(
        #     2 * self.config.decoder_layers, 
        #     bsz,
        #     self.config.decoder_attention_heads,
        #     self.config.decoder_preseqlen,
        #     -1,
        # ).split(2)

        # final_prompts = []

        # for a, b in zip(self_prefix_prompt, cross_prefix_prompt):
        #     print(a.shape, b.shape)
        #     final_prompts.append(torch.cat([a, b], dim=0))
            
        #     print(final_prompts[-1].shape)

        # return final_prompts

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.encoder.visual_embedding.obj_order_embedding = self.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=torch.float, device=input_ids.device)
        
        if self.config.preseqlen > 0:
            prefix_attention_mask = torch.ones(
                attention_mask.shape[0], self.config.preseqlen, dtype=attention_mask.dtype, device=attention_mask.device
            )

            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)

        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        if self.decoder_prefix_embedding is not None and past_key_values is None:
            past_key_values = self.get_prompt(B, attention_mask.device)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VLBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        self.model = VLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        reduce_loss=False,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,

            vis_inputs=vis_inputs,
            vis_attention_mask=vis_attention_mask,

            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "vis_attention_mask" in kwargs:
            output["vis_attention_mask"] = kwargs['vis_attention_mask']

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


if __name__ == "__main__":
    import transformers

    config = transformers.AutoConfig.from_pretrained("facebook/bart-base")

    config.feat_dim = 2048
    config.pos_dim = 4
    config.n_images = 2

    config.use_vis_order_embedding = True
    config.additional_visual_embedding_layers = 0
    config.preseqlen = 2
    config.decoder_preseqlen = 2

    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    config.use_vis_layer_norm = True
    config.individual_vis_layer_norm = True
    config.losses = 'lm,obj,attr,feat'

    config.share_vis_lang_layer_norm = False
    config.classifier = False

    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base")

    num_added_toks = 0
    additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
            [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config.default_obj_order_ids = tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

    model = VLBart.from_pretrained("facebook/bart-base", config=config)
    model.resize_token_embeddings(model.model.shared.num_embeddings + num_added_toks)
    model.tokenizer = tokenizer

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    
    vis_feats = torch.randn(1, 36, 2048)
    vis_pos = torch.randn(1, 36, 4)

    generation_output = model.generate(
                **inputs,
                vis_inputs=(vis_feats, vis_pos)
    )
    
    print(generation_output)

    print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))

    vis_feats = torch.randn(1, 36, 2048)
    vis_pos = torch.randn(1, 36, 4)

    generation_output = model.generate(
                **inputs,
                vis_inputs=(vis_feats, vis_pos),
    )
    
    print(generation_output)

    print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))