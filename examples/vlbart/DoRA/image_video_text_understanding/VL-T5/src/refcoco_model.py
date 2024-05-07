import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modeling_t5 import VLT5
class VLT5RefCOCO(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_attention_mask = batch['vis_attention_mask'].to(device)

        B, V_L = vis_feats.size()[:2]

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss_mask = batch['exists_target'].to(device=device)

        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        result = {
            'loss': loss
        }

        with torch.no_grad():
            logits = output['logits'].detach()
            # logits = logits.view(B, 2, self.config.vocab_size)
            logits = logits.view(B, L, self.config.vocab_size)

            # target = lm_labels[:, 0].view(B)

            pred = logits[:, 0].argmax(dim=1).view(B)
            # correct = pred == target

            pred = pred.cpu().numpy()

            correct = np.zeros([B])
            for i in range(B):
                correct[i] = pred[i] in batch['all_target_ids'][i]

            result['pred'] = pred
            result['correct'] = correct

        return result

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_attention_mask = batch['vis_attention_mask'].to(device)

        B, V_L = vis_feats.size()[:2]

        decoder_input_ids = torch.ones(B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach()
        logits = logits.view(B, self.config.vocab_size)

        pred = logits.argmax(dim=1).view(B)
        pred = pred.cpu().numpy()

        correct = np.zeros([B])
        for i in range(B):
            correct[i] = pred[i] in batch['all_target_ids'][i]

        result = {}
        result['pred'] = pred
        result['correct'] = correct

        return result

from modeling_bart import VLBart
from collections import defaultdict
class VLBartRefCOCO(VLBart):
    def __init__(self, config):
        super().__init__(config)

        out_map = defaultdict(lambda: -1)
        for i in range(100):
            out_map[f'<vis_extra_id_{i}>'] = i

        self.out_map = out_map

    def train_step(self, batch):
        self.train()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_attention_mask = batch['vis_attention_mask'].to(device)

        B, V_L = vis_feats.size()[:2]

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.config.pad_token_id),
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            labels=lm_labels,
            reduce_loss=True,
            return_dict=True
        )
        assert 'loss' in output

        # lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }

        with torch.no_grad():
            logits = output['logits'].detach().view(B, L, self.lm_head.out_features)[:, 1]
            logits = logits.view(B, self.lm_head.out_features)

            pred = logits.argmax(dim=1).view(B)
            pred = pred.cpu().numpy()
            pred = self.lm_head.out_features - pred - 1

        correct = np.zeros([B])
        for i in range(B):
            correct[i] = pred[i] in batch['all_targets'][i]

        result['pred'] = pred
        result['correct'] = correct

        return result

    @torch.no_grad()
    def test_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        vis_attention_mask = batch['vis_attention_mask'].to(device)

        B, V_L = vis_feats.size()[:2]

        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

        output = self(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.config.pad_token_id),
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        logits = output['logits'].detach().view(B, 2, self.lm_head.out_features)[:, 1]
        logits = logits.view(B, self.lm_head.out_features)

        pred = logits.argmax(dim=1).view(B)
        pred = pred.cpu().numpy()

        pred = self.lm_head.out_features - pred - 1

        correct = np.zeros([B])
        for i in range(B):
            correct[i] = pred[i] in batch['all_targets'][i]

        result = {}
        result['pred'] = pred
        result['correct'] = correct

        return result
