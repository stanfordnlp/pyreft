
import torch
import torch.nn as nn
import numpy as np


from modeling_t5 import VLT5
class VLT5NLVR(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)

        if batch.get("images", None) is not None:
            _, _, channel, width, height = batch['images'].shape

            # size of images: (B, 2, 3, 224, 224)
            batch['images'] = batch['images'].view(B*2, channel, width, height)
            batch = self.vis_forward(batch, device)
            # size of vis_feats: (B*2, V_L, feature_size)
            # size of boxes: (B*2, V_L, 4)
            V_L = batch['vis_feats'].size(1)
        else:
            # size of vis_feats: (B, 2, V_L, feature_size)
            V_L = batch['vis_feats'].size(2)

        task = batch["task"]

        vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, -1)
        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

        lm_labels = batch["target_ids"].to(device)

        img_order_ids = [0] * V_L + [1] * V_L
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            labels=lm_labels,
            return_dict=True,
            task=task
        )
        assert 'loss' in output

        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss = loss.mean()

        result = {
            'loss': loss
        }

        logits = output['logits'].detach()[:, 0]
        logits = logits.view(B, self.lm_head.out_features)
        true_logit = logits[:, self.true_id]
        false_logit = logits[:, self.false_id]

        pred_true = true_logit > false_logit
        pred_true = pred_true.long().cpu().numpy()
        result['pred_ans_id'] = pred_true

        return result

    def test_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        
        if batch.get("images", None) is not None:
            _, _, channel, width, height = batch['images'].shape

            # size of images: (B, 2, 3, 224, 224)
            batch['images'] = batch['images'].view(B*2, channel, width, height)
            batch = self.vis_forward(batch, device)
            # size of vis_feats: (B*2, V_L, feature_size)
            # size of boxes: (B*2, V_L, 4)
            V_L = batch['vis_feats'].size(1)
        else:
            # size of vis_feats: (B, 2, V_L, feature_size)
            V_L = batch['vis_feats'].size(2)

        task = batch["task"]

        vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, -1)
        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

        img_order_ids = [0] * V_L + [1] * V_L
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

        decoder_input_ids = torch.ones(B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            task=task,
        )

        logits = output['logits'].detach()[:, 0]
        logits = logits.view(B, self.lm_head.out_features)
        true_logit = logits[:, self.true_id]
        false_logit = logits[:, self.false_id]

        pred_true = true_logit > false_logit
        pred_true = pred_true.long().cpu().numpy()

        result = {}
        result['pred_ans_id'] = pred_true

        return result


from modeling_bart import VLBart
class VLBartNLVR(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)

        if batch.get("images", None) is not None:
            _, _, channel, width, height = batch['images'].shape

            # size of images: (B, 2, 3, 224, 224)
            batch['images'] = batch['images'].view(B*2, channel, width, height)
            batch = self.vis_forward(batch, device)
            # size of vis_feats: (B*2, V_L, feature_size)
            # size of boxes: (B*2, V_L, 4)
            V_L = batch['vis_feats'].size(1)
        else:
            # size of vis_feats: (B, 2, V_L, feature_size)
            V_L = batch['vis_feats'].size(2)

        task = batch["task"]

        vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, -1)
        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

        lm_labels = batch["target_ids"].to(device)


        img_order_ids = [0] * V_L + [1] * V_L
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            labels=lm_labels,
            return_dict=True,
            task=task,
        )
        assert 'loss' in output

        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss = loss.mean()

        result = {
            'loss': loss
        }

        logits = output['logits'].detach()[:, 1]
        logits = logits.view(B, self.lm_head.out_features)
        true_logit = logits[:, self.true_id]
        false_logit = logits[:, self.false_id]

        pred_true = true_logit > false_logit
        pred_true = pred_true.long().cpu().numpy()
        result['pred_ans_id'] = pred_true

        return result

    def test_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        
        if batch.get("images", None) is not None:
            _, _, channel, width, height = batch['images'].shape

            # size of images: (B, 2, 3, 224, 224)
            batch['images'] = batch['images'].view(B*2, channel, width, height)
            batch = self.vis_forward(batch, device)
            # size of vis_feats: (B*2, V_L, feature_size)
            # size of boxes: (B*2, V_L, 4)
            V_L = batch['vis_feats'].size(1)
        else:
            # size of vis_feats: (B, 2, V_L, feature_size)
            V_L = batch['vis_feats'].size(2)

        task = batch["task"]

        vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, -1)
        vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)

        img_order_ids = [0] * V_L + [1] * V_L
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

        decoder_input_ids = torch.tensor(
            [self.config.decoder_start_token_id, self.config.bos_token_id],
            dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
            task=task,
        )

        logits = output['logits'].detach()[:, 1]
        logits = logits.view(B, self.lm_head.out_features)
        true_logit = logits[:, self.true_id]
        false_logit = logits[:, self.false_id]

        pred_true = true_logit > false_logit
        pred_true = pred_true.long().cpu().numpy()

        result = {}
        result['pred_ans_id'] = pred_true

        return result