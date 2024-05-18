
import torch
import torch.nn as nn


from modeling_t5 import VLT5
class VLT5COCOCaption(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss,
            task=task,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            task=task,
            **kwargs,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result


from modeling_bart import VLBart
class VLBartCOCOCaption(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        intervention_locations = batch['intervention_locations'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss,
            task=task,
            intervention_locations=intervention_locations
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device

        batch = self.vis_forward(batch, device)
        task = batch["task"]
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        intervention_locations = batch['intervention_locations'].to(device)

        generation_args = {
            "base": {
                "input_ids":input_ids,
                "vis_inputs":(vis_feats, vis_pos),
                "task":task,
                **kwargs
            },
            "unit_locations": {"sources->base": (None, 
                intervention_locations.permute(1, 0, 2))},
            "intervene_on_prompt": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "early_stopping": True,
            "model": self,
        }
        # TODO: temperature, top_p, top_k
        _, output = self.intervenable.generate(**generation_args)
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result