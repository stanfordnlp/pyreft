# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param import args
from pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from src.tasks.vision_helpers import GroupedBatchSampler, create_aspect_ratio_groups_cache
from lxrt.visual_transformers import adjust_learning_rate
from src.tools.load_stagte_dict import load_state_dict_flexible_with_fp16, load_state_dict_flexible
import gc
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')

if args.distributed:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    args.gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
args.gpus = torch.cuda.device_count()
args.gpu = args.local_rank if args.local_rank != -1 else 0
args.device = torch.device("cuda", args.gpu)

def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1, distributed = False, aspect_ratio_group_factor= -1) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    tset = LXMERTTorchDataset(dset, topk)

    if distributed:
        train_sampler = DistributedSampler(
        tset,
        num_replicas=args.world_size,
        rank=args.local_rank,
        shuffle=shuffle,
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(tset)
        if not shuffle:
            train_sampler = torch.utils.data.SequentialSampler(tset)
    
    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups_cache(tset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, bs)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, bs, drop_last=True)

    data_loader = DataLoader(
        tset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=tset.collate_fn,
        pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)


train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True, distributed=args.distributed, aspect_ratio_group_factor = args.aspect_ratio_group_factor)
valid_batch_size = 16 if args.multiGPU else 16
valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000)



LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA')


def to_gpu(tensor, device = None):
    if tensor is not None and isinstance(tensor, torch.Tensor):
        if device is not None:
            return tensor.to(device)
        else:
            return tensor.cuda()
    return tensor

class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=train_tuple.dataset.answer_table.num_answers
        )

        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)

        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)
        #print(list(state_dict))
    
        self.model = self.model.to(args.device)

        if args.distributed:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            from transformers import AdamW, get_linear_schedule_with_warmup

            if args.use_separate_optimizer_for_visual:
                from lxrt.visual_transformers import FusedOptimizer

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("visual_model" not in n) ) ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if ( (any(nd in n for nd in no_decay)) and ("visual_model" not in n ))],
                        "weight_decay": 0.0,
                    },
                ]
                optim = AdamW(optimizer_grouped_parameters,
                            lr=args.lr,
                            #betas=(0.9, 0.98),
                            eps=args.adam_epsilon)
                
                #sgd_parameters = self.model.bert.encoder.visual_model.parameters()

                if args.use_adam_for_visual:
                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in self.model.bert.encoder.visual_model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("visual_model" not in n) ) ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [p for n, p in self.model.bert.encoder.visual_model.named_parameters() if ( (any(nd in n for nd in no_decay)) and ("visual_model" not in n ))],
                            "weight_decay": 0.0,
                        },
                    ]
                    sgd = AdamW(optimizer_grouped_parameters,
                            lr=args.sgd_lr,
                            #betas=(0.9, 0.98),
                            eps=args.adam_epsilon)
                else:
                    sgd = torch.optim.SGD(self.model.bert.encoder.visual_model.parameters(), args.sgd_lr,
                                    momentum=args.sgd_momentum,
                                    weight_decay=args.sgd_weight_decay)

                self.optim = FusedOptimizer([optim, sgd])
                batch_per_epoch = len(train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs) // args.gradient_accumulation_steps
                self.scheduler = get_linear_schedule_with_warmup(
                    optim, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total)

            else:
                self.optim = AdamW(optimizer_grouped_parameters,
                            lr=args.lr,
                            #betas=(0.9, 0.98),
                            eps=args.adam_epsilon)
                
                batch_per_epoch = len(train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs) // args.gradient_accumulation_steps
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optim, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total
                )

            if args.fp16:
                if args.use_separate_optimizer_for_visual:
                    self.model, [optim, sgd] = amp.initialize(self.model, self.optim.optimizers, enabled=args.fp16, opt_level=args.fp16_opt_level)
                    self.optim = FusedOptimizer([optim, sgd])
                else:
                    self.model, self.optim = amp.initialize(self.model, self.optim, enabled=args.fp16, opt_level=args.fp16_opt_level)
                from apex.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model)
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[args.gpu], find_unused_parameters=True
                )
        else:
            # GPU Options
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)
            # Optimizer
            from lxrt.optimization import BertAdam
            batch_per_epoch = len(train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            warmup_ratio = 0.05
            warmup_iters = int(t_total * warmup_ratio)
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print("Warm up Iters: %d" % warmup_iters)
            self.optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)
        if args.load is not None:
            self.load(args.load)
            torch.cuda.empty_cache()
            gc.collect()
        
    def forward(self, examples):
        '''train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda() '''

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        new_examples = {}
        for key in list(examples.keys()):
            if key != "uid":
                new_examples[key] = to_gpu(examples[key])
        
        loss, losses, ans_logit = self.model(
            **new_examples
        )
        return loss, losses.detach().cpu(), ans_logit

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Train
        best_eval_loss = 9595.
        for epoch in range(args.start_epoch, args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            uid2ans = {}
            from utils import TrainingMeter
            train_meter = TrainingMeter()
            
            if args.use_separate_optimizer_for_visual:
                adjust_learning_rate(self.optim.optimizers[-1], epoch, args)

            for i, batch in enumerate(tqdm(train_ld, total=len(train_ld))):
                if args.skip_training and i == 4:
                    break
                loss, losses, ans_logit = self.forward(batch)
                if args.multiGPU:
                    loss = loss.mean()
                losses = losses.squeeze(0)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    if args.use_separate_optimizer_for_visual:
                        with amp.scale_loss(loss, self.optim.optimizers) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                else:
                    loss.backward()
                
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), args.max_grad_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    
                    self.optim.step()
                    if args.distributed:
                        self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    #self.optim.step()

                loss = loss.item()
                losses = losses.cpu().numpy()
                logit = ans_logit

                total_loss += loss
                total_losses += losses

                if args.task_qa:
                    score, label = logit.max(1)
                    for uid, l in zip(batch["uid"], label.cpu().numpy()):
                        ans = train_tuple.dataset.answer_table.id2ans(l)
                        uid2ans[uid] = ans
                
                train_meter.update(
                    {'totol_loss': loss*args.gradient_accumulation_steps,
                    "masked_lm": losses[0],
                    "matched": losses[1],
                    "qa_loss": losses[2] if len(losses) == 3 else 0.0,
                    }
                )

                if i != 0 and i % args.report_step == 0 and args.local_rank <= 0:
                    print("Epoch {}, Training Step {} of {}".format(epoch, i // args.gradient_accumulation_steps, len(train_ld) // args.gradient_accumulation_steps ))
                    train_meter.report()
                    train_meter.clean()
                
                if i != 0 and args.save_step != -1 and (i // args.gradient_accumulation_steps) % args.save_step == 0 and args.local_rank <= 0:
                    self.save("Epoch{}Step{}".format(epoch+1, i // args.gradient_accumulation_steps ))

            #if args.task_qa:
            #    train_tuple.evaluator.evaluate(uid2ans, pprint=True)

            # Save
            if args.local_rank <= 0:
                self.save("Epoch%02d" % (epoch+1))
            # Eval
            #avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1)

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        for i, batch in enumerate(tqdm(eval_ld)):
            loss, losses, logit = self.valid_batch(batch)

            total_loss += loss
            total_losses += losses
            if args.task_qa:
                score, label = logit.max(1)
                for uid, l in zip(batch["uid"], label.cpu().numpy()):
                    ans = train_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans
            if i == iters:
                break
        if args.local_rank <= 0:
            print("The valid loss is %0.4f" % (total_loss / len(eval_ld)))
            losses_str = "The losses are "
            total_losses = total_losses.squeeze(0)
            for name, loss in zip(LOSSES_NAME, total_losses / len(eval_ld)):
                losses_str += "%s: %0.4f " % (name, loss)
            print(losses_str)

            if args.task_qa:
                eval_tuple.evaluator.evaluate(uid2ans, pprint=True)

        return total_loss / len(eval_ld)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_LXRT.pth" % name))
        if args.use_separate_optimizer_for_visual:
            torch.save(self.optim.optimizers[0].state_dict(), os.path.join(args.output, "%s_LXRT_AdamOptim.pth" % name))
            torch.save(self.optim.optimizers[1].state_dict(), os.path.join(args.output, "%s_LXRT_SGDOptim.pth" % name))
        else:
            torch.save(self.optim.state_dict(), os.path.join(args.output, "%s_LXRT_AdamOptim.pth" % name))
        torch.save(self.scheduler.state_dict(), os.path.join(args.output, "%s_LXRT_Scheduler.pth" % name))

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location='cpu')
        '''new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key.replace("module.", "")] = value
            else:
                new_state_dict[key] = value'''
        load_state_dict_flexible_with_fp16(self.model, state_dict)
        #self.model.load_state_dict(new_state_dict)
        if os.path.exists("{}_LXRT_SGDOptim.pth".format(path)):
            # load sgd
            print("Load SGD from {}".format("{}_LXRT_SGDOptim.pth".format(path)))
            sgd_state = torch.load("{}_LXRT_SGDOptim.pth".format(path), map_location='cpu')
            self.optim.optimizers[-1].load_state_dict(sgd_state)
        if args.not_load_adam_optimizer:
            pass
        elif os.path.exists("{}_LXRT_AdamOptim.pth".format(path)):
            # load sgd
            print("Load Adam")
            sgd_state = torch.load("{}_LXRT_AdamOptim.pth".format(path), map_location='cpu')
            self.optim.optimizers[0].load_state_dict(sgd_state)
        
        if args.not_load_scheduler:
            pass
        elif os.path.exists("{}_LXRT_Scheduler.pth".format(path)):
            # load sgd
            print('Load scheduler')
            sgd_state = torch.load("{}_LXRT_Scheduler.pth".format(path), map_location='cpu')
            self.scheduler.load_state_dict(sgd_state)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path, map_location="cpu")

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        load_state_dict_flexible_with_fp16(self.model, state_dict)
        #self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    import sys
    if args.gpu == 0:
        print("\n\n")
        print(" ".join(sys.argv))
        print("\n\n")

    lxmert = LXMERT(max_seq_length=20)

    lxmert.train(train_tuple, valid_tuple)
