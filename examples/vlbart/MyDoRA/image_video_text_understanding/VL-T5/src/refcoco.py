
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args

from refcoco_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict
import wandb

# set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        from refcoco_model import VLT5RefCOCO, VLBartRefCOCO

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5RefCOCO
        elif 'bart' in args.backbone:
            model_class = VLBartRefCOCO

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            n_correct = 0
            n_total = 0
            for batch in self.val_loader:
                exists_target = batch['exists_target']

                n_correct += exists_target.sum().item()
                n_total += len(exists_target)

            print(f'Val Oracle acc: {n_correct / n_total * 100:.2f}%')

            n_correct = 0
            n_total = 0
            for batch in self.test_loader:
                exists_target = batch['exists_target']

                n_correct += exists_target.sum().item()
                n_total += len(exists_target)

            print(f'Test Oracle acc: {n_correct / n_total * 100:.2f}%')

        if self.verbose:
            loss_meter = LossMeter()

            best_valid_acc = 0.
            best_epoch = 0

            if 't5' in self.args.backbone:
                project_name = "VLT5_RefCOCOg"
            elif 'bart' in self.args.backbone:
                project_name = "VLBart_RefCOCOg"

            if self.args.RefCOCO_GT:
                project_name += '_GT'

            wandb.init(project=project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)


            epoch_results = {
                'loss': 0,

            }

            n_correct = 0
            n_total = 0

            for step_i, batch in enumerate(self.train_loader):

                batch['log_train_accuracy'] = self.args.log_train_accuracy

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.args.log_train_accuracy:
                    correct = results['correct']
                    n_correct += sum(correct)
                    n_total += len(correct)

                if self.verbose:
                    loss_meter.update(loss.item())
                    # acc_meter.update(results['acc'].item())

                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | '
                    desc_str += f'Loss {loss_meter.val:.3f} |'
                    # desc_str += f' Acc {acc_meter.val:.3f} |'

                    if self.args.log_train_accuracy:
                        desc_str += f' Correct {n_correct:.0f}'
                        desc_str += f' (Acc {n_correct/n_total*100:.1f}%)'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()

            if self.args.log_train_accuracy:
                train_score_dict = {
                    'n_correct': n_correct,
                    'n_total': n_total
                }
                train_score_dict = reduce_dict(train_score_dict, self.args.gpu)

            # Validation
            # valid_score_dict = self.evaluate(self.val_loader)
            # valid_score_dict = reduce_dict(valid_score_dict, self.args.gpu)

            if self.verbose:
                if self.args.log_train_accuracy:
                    train_acc = train_score_dict['n_correct']/train_score_dict['n_total'] * 100
                    train_n_correct = int(train_score_dict['n_correct'])
                    train_n_total = int(train_score_dict['n_total'])

                # Validation
                valid_score_dict = self.evaluate(self.val_loader)
                valid_acc = valid_score_dict['n_correct']/valid_score_dict['n_total'] * 100
                valid_n_correct = int(valid_score_dict['n_correct'])
                valid_n_total = int(valid_score_dict['n_total'])

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''

                if self.args.log_train_accuracy:
                    log_str += f"\nEpoch {epoch}: Train"
                    log_str += f" Acc {train_acc:.2f}% |"
                    log_str += f" # correct {train_n_correct} # total {train_n_total}"

                log_str += f"\nEpoch {epoch}: Valid"
                log_str += f" Acc {valid_acc:.2f}% |"
                log_str += f" # correct {valid_n_correct} # total {valid_n_total}"

                log_str += f"\nEpoch {best_epoch}: Best  Acc {best_valid_acc:.2f}%\n"

                wandb_log_dict = {}

                if self.args.log_train_accuracy:
                    wandb_log_dict['Train/Acc'] = train_acc

                wandb_log_dict['Valid/Acc'] = valid_acc

                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

            # Test Set
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)

            test_score_dict = self.evaluate(self.test_loader)
            test_acc = test_score_dict['n_correct']/test_score_dict['n_total'] * 100
            test_n_correct = int(test_score_dict['n_correct'])
            test_n_total = int(test_score_dict['n_total'])

            wandb_log_dict = {}
            wandb_log_dict['Test/Acc'] = test_acc
            wandb.log(wandb_log_dict, step=epoch)

            log_str = ''
            log_str += f"\nTest Acc {test_acc:.2f}%"
            log_str += f"\nTest # correct {test_n_correct} # total {test_n_total}"

            print(log_str)

            wandb.log({'finished': True})

        if self.args.distributed:
            dist.barrier()

    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():

            score_dict = {}
            n_correct = 0
            n_total = 0

            if self.verbose:
                iterator = tqdm(loader, ncols=120, desc="Prediction")
            else:
                iterator = loader

            for i, batch in enumerate(iterator):

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                correct = results['correct']
                n_correct += sum(correct)
                n_total += len(correct)

        score_dict = {}
        score_dict['n_correct'] = n_correct
        score_dict['n_total'] = n_total

        return score_dict

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )
    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )
    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        # if not args.test:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
