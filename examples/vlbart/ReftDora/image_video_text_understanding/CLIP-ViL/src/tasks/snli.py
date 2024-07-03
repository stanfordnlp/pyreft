# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa

from utils import TrainingMeter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from vision_helpers import GroupedBatchSampler, create_aspect_ratio_groups
from lxrt.visual_transformers import adjust_learning_rate

from tasks.vqa_model import VQAModel
from tasks.snli_data import SNLIDataset

from lxrt.adapters import AdapterController
from lxrt.modeling import VisualFeatEncoder
from clip.model import VisualAdapter

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, distributed=False, aspect_ratio_group_factor = -1, exhaustive = False) -> DataTuple:
    tset = SNLIDataset(splits)
    #asser
    dset = None
    evaluator = None#VQAEvaluator(dset)

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
        group_ids = create_aspect_ratio_groups(tset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, bs, exhaustive = exhaustive)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, bs, drop_last=True)

    data_loader = DataLoader(
        tset,
        batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn = tset.collate_fn
    )
    '''else:
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )'''

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.args = args
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True, distributed=args.distributed,
             aspect_ratio_group_factor=args.aspect_ratio_group_factor
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size,
                shuffle=False, drop_last=False,
                distributed=False,
                aspect_ratio_group_factor=args.aspect_ratio_group_factor
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = VQAModel(3)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        self.model = self.model.to(args.device)

        if args.use_adapter:
            self.freeze_whole_model()
            self.unfreeze_parameters()

        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)
        
        # Loss and Optimizer
        self.bce_loss = nn.CrossEntropyLoss(ignore_index=-1)
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
                visn_model = self.model.lxrt_encoder.model.bert.encoder.visual_model
                if args.use_adam_for_visual:

                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in visn_model.named_parameters() if ( (not any(nd in n for nd in no_decay)) and ("visual_model" not in n) ) ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [p for n, p in visn_model.named_parameters() if ( (any(nd in n for nd in no_decay)) and ("visual_model" not in n ))],
                            "weight_decay": 0.0,
                        },
                    ]
                    sgd = AdamW(optimizer_grouped_parameters,
                            lr=args.sgd_lr,
                            #betas=(0.9, 0.98),
                            eps=args.adam_epsilon)
                else:
                    sgd_parameters = visn_model.parameters()
                    sgd = torch.optim.SGD(sgd_parameters, args.sgd_lr,
                                    momentum=args.sgd_momentum,
                                    weight_decay=args.sgd_weight_decay)

                self.optim = FusedOptimizer([optim, sgd])
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs) // args.gradient_accumulation_steps
                self.scheduler = get_linear_schedule_with_warmup(
                    optim, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total
                )
            else:
                self.optim = AdamW(optimizer_grouped_parameters,
                            lr=args.lr,
                            #betas=(0.9, 0.98),
                            eps=args.adam_epsilon)
                
                batch_per_epoch = len(self.train_tuple.loader)
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
        elif 'bert' in args.optim:
                # GPU options
            #self.model = self.model.cuda()
            if args.multiGPU:
                self.model.lxrt_encoder.multi_gpu()

            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs) // args.gradient_accumulation_steps
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def print_trainable_params_percentage(self, model):
        # if "bart-base" in self.args.backbone:
        #     orig_param_size = 139420416
        # elif "t5-base" in self.args.backbone:
        #     orig_param_size = 222903552
        # else:
        #     print(f"Don't know the parameters number of this {self.args.backbone}")
        #     orig_param_size = -1

        orig_param_size = sum(p.numel() for p in model.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(model)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}%")

        print(trainable_size)

        return percentage

    def freeze_whole_model(self):
        for n, p in self.model.named_parameters():
            p.requires_grad = False

    def unfreeze_parameters(self):    
        targets = ["logit_fc"]
        # unfreeze the parameters in targets anyway
        for n, p in self.model.named_parameters():
            if any(t in n for t in targets):
                p.requires_grad = True
                print(f"{n} is trainable...")
   
        for name, sub_module in self.model.named_modules():
            # if self.args.unfreeze_vis_encoder:
            #     if isinstance(sub_module, (CLIPResNetEncoder)):
            #         print(f"{name} is trainable...")
            #         # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True


            # if self.args.use_vis_adapter:
            #     if isinstance(sub_module, (VisualAdapter)):
            #         print(f"{name} is trainable...")
            #         # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
            #         for param_name, param in sub_module.named_parameters():
            #             param.requires_grad = True

            if isinstance(sub_module, VisualFeatEncoder):
                print(f"{name} is trainable...")
                # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

            if self.args.use_adapter:
                if isinstance(sub_module, nn.LayerNorm):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

                if isinstance(sub_module, (AdapterController)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_vis_adapter:
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

                if isinstance(sub_module, (VisualAdapter)):
                    print(f"{name} is trainable...")
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

            if self.args.use_bn:
                if isinstance(sub_module, nn.BatchNorm2d):
                    print(f"{name} is trainable...")
                    # if len(name.split(".")) < 7: # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        train_meter = TrainingMeter()
        for epoch in range(args.epochs):
            if args.use_separate_optimizer_for_visual:
                adjust_learning_rate(self.optim.optimizers[-1], epoch, args)
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
                
                if args.skip_training:
                    break
                
                self.model.train()
                #self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                
                loss = self.bce_loss(logit, target.squeeze(-1))
                
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
                    #if args.distributed:
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    #self.optim.step()

                train_meter.update(
                    {'loss': loss.detach().mean().item() * args.gradient_accumulation_steps,
                    "acc": (logit.argmax(-1) == target.squeeze(-1)).sum() / len(target)}
                )

                score, label = logit.max(1)

                if i != 0 and i % args.report_step == 0 and args.local_rank <= 0:
                    print("Epoch {}, Training Step {} of {}".format(epoch, i, len(loader)))
                    train_meter.report()
                    train_meter.clean()

            log_str = "\nEpoch %d: \n" % (epoch)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if args.local_rank <= 0:
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            if args.local_rank <= 0:
                print(log_str, end='')

        if args.local_rank <= 0:
            self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        correct_counter = 0
        all_counter = 0
        for i, datum_tuple in enumerate(tqdm(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                correct_counter += (logit.argmax(-1).cpu() == target.squeeze(-1)).sum().item()
                all_counter += len(logit)
        
        return correct_counter / all_counter

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        
        return self.predict(eval_tuple, dump)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location="cpu")
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    if args.distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        args.gpus = torch.cuda.device_count()
        args.world_size = args.gpus * args.nodes
    args.gpus = torch.cuda.device_count()
    args.gpu = args.local_rank if args.local_rank != -1 else 0
    args.device = torch.device("cuda", args.gpu)
    import sys
    if args.gpu == 0:
        print("\n\n")
        print(" ".join(sys.argv))
        print("\n\n")
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            result = vqa.predict(
                get_data_tuple(args.test, bs=args.batch_size,
                               shuffle=False, drop_last=False, aspect_ratio_group_factor=args.aspect_ratio_group_factor, exhaustive=True))
            print(result)
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=args.batch_size,
                               shuffle=False, drop_last=False, aspect_ratio_group_factor=args.aspect_ratio_group_factor),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        
        vqa.train(vqa.train_tuple, vqa.valid_tuple)