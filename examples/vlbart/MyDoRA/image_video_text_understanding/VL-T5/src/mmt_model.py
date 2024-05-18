
import torch
import torch.nn as nn


from modeling_t5 import VLT5
class VLT5MMT(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            reduce_loss=True,
            return_dict=True
        )

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result


from modeling_bart import VLBart
class VLBartMMT(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            reduce_loss=True,
            return_dict=True
        )

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result
