import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from utils import LossMeter
from dist_utils import reduce_dict

import wandb

from vis_encoder import get_vis_encoder
from transformers.models.t5.modeling_t5 import T5LayerNorm
import modeling_t5
import modeling_bart
from clip.model import VisualAdapter

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

        from pretrain_model import VLT5Pretraining, VLBartPretraining, DSVLBartPretraining

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5Pretraining
        elif 'bart' in args.backbone:
            model_class = VLBartPretraining

            # if self.args.deepspeed:
            #     model_class = DSVLBartPretraining

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

        if self.include_vis_encoder:
            # train vision encoder end-to-end
            vis_encoder_type = self.args.feature_type.split("_")[-1]

            if self.args.use_vis_adapter:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=self.args.vis_adapter_type,
                    reduction_factor=self.args.vis_reduction_factor,
                )
            else:
                self.vis_encoder = get_vis_encoder(
                    backbone=vis_encoder_type, 
                    image_size=eval(self.args.image_size)[0],
                    adapter_type=None,
                )

            self.model.vis_encoder = self.vis_encoder

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

        self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if args.deepspeed:
                from my_deepspeed import deepspeed_init
                
                self.model, self.optim, self.lr_scheduler = deepspeed_init(self)
            elif self.args.fp16 and _use_native_amp:
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
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 9595.

            if 't5' in self.args.backbone:
                project_name = "VLT5_Pretrain"
            elif 'bart' in self.args.backbone:
                project_name = "VLBart_Pretrain"

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
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()
            self.partial_eval()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=250)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):
                
                if self.deepspeed:
                    # results = self.model(batch=batch, train_mode=True)
                    # if args.fp16:
                    #     batch["images"] = batch["images"].half()
                    results = self.model.module.train_step(batch)
                elif self.args.fp16 and _use_native_amp:
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

                if self.deepspeed:
                    self.model.backward(loss)
                elif self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0 and not self.deepspeed:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.deepspeed:
                    self.model.step()
                elif self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler and not self.deepspeed:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

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

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            wandb.log({f'Train Loss/{name}': avg_loss}, step=epoch)

                losses_str += '\n'
                print(losses_str)

            dist.barrier()

            # Validation
            valid_results, valid_uid2ans = self.evaluate_epoch(epoch=epoch)

            valid_results = reduce_dict(valid_results, average=False)
            if self.verbose:
                valid_loss = valid_results['total_loss']
                valid_loss_count = valid_results['total_loss_count']

                avg_valid_loss = valid_loss / valid_loss_count
                losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                for name, loss in valid_results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(valid_results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            wandb.log({f'Valid Loss/{name}': avg_loss}, step=epoch)

                losses_str += '\n'
                print(losses_str)

            if 'qa' in self.args.losses:
                dset2score, dset2cnt, score, cnt = self.val_loader.dataset.evaluator.evaluate(valid_uid2ans)

                if len(dset2score) == 0:
                    dset2score = {'vqa': 0, 'gqa': 0, 'visual7w': 0}
                    dset2cnt = {'vqa': 1, 'gqa': 1, 'visual7w': 1}
                    cnt = 3
                    score = 0

                dset2score = reduce_dict(dset2score, average=False)
                dset2cnt = reduce_dict(dset2cnt, average=False)
                score_cnt_dict = reduce_dict({'score': score, 'cnt': cnt}, average=False)

                if self.args.gpu == 0:
                    score = score_cnt_dict['score']
                    cnt = score_cnt_dict['cnt']
                    accu = score / cnt
                    dset2accu = {}
                    for dset in dset2cnt:
                        dset2accu[dset] = dset2score[dset] / dset2cnt[dset]
                    accu_str = "Overall QA Acc %0.4f" % (accu)
                    wandb.log({f'Valid QA Acc/Overall': accu}, step=epoch)
                    sorted_keys = sorted(dset2accu.keys())
                    for key in sorted_keys:
                        accu_str += ", %s Acc %0.4f" % (key, dset2accu[key])
                        wandb.log({f'Valid QA Acc/{key}': dset2accu[key]}, step=epoch)
                    print(accu_str)
                    accu_str += '\n\n'

            dist.barrier()

            if self.verbose:
                # Save
                if avg_valid_loss < best_eval_loss:
                    best_eval_loss = avg_valid_loss
                #     self.save("BEST_EVAL_LOSS")
                self.save("Epoch%02d" % (epoch + 1))

            dist.barrier()

        if self.verbose:
            wandb.log({'finished': True})

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        uid2ans = {}

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=250)

            for step_i, batch in enumerate(self.val_loader):
                
                if self.deepspeed:
                    # results = self.model(batch, train_mode=False)
                    # if args.fp16:
                        # batch["images"] = batch["images"].half()
                    results = self.model.module.valid_step(batch)
                elif self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                if 'qa' in self.args.losses:
                    qa_pred = results['qa_pred']
                    for uid, ans in zip(batch['uid'], qa_pred):
                        uid2ans[uid] = ans

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            if 'qa' not in self.args.losses:
                uid2ans = None

            return epoch_results, uid2ans

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')
    
    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed()
        pass
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
    
    # use different type of inputs features
    if args.feature_type == "butd":
        from pretrain_data import get_loader
    elif args.feature_type.startswith("raw"):
        feature_type = args.feature_type.split("_")[-1]

        if args.vis_pooling_output:
            feat_dim_dict = {
                "RN50": 1024,
                "RN101": 512,
                "RN50x4": 640,
            }
        else:
            feat_dim_dict = {
                "RN50": 2048,
                "RN101": 2048,
                "RN50x4": 2560,
                "ViT": 768
            }
        args.feat_dim = feat_dim_dict[feature_type]

        from pretrain_raw_data import get_loader
    else:
        raise NotImplementedError
        feat_dim_dict = {
            "RN50": 2048,
            "RN101": 2048,
            "RN50x4": 2560,
            "ViT": 768
        }
        args.feat_dim = feat_dim_dict[args.feature_type]
        from vqa_clip_data import get_loader

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,)

    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk,)

    trainer = Trainer(args, train_loader, val_loader, train=True)

    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss') # total loss

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'coco' in args.train:
        dsets.append('COCO')
    if 'vg' in args.train:
        dsets.append('VG')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        if args.run_name == "":
            args.run_name = run_name

    # if args.distributed:
    main_worker(args.local_rank, args)
