
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

from vcr_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
from dist_utils import reduce_dict, all_gather
import wandb

set_global_logging_level(logging.ERROR, ["transformers"])

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

        from vcr_model import VLT5VCR, VLBartVCR

        if 't5' in args.backbone:
            model_class = VLT5VCR
        elif 'bart' in args.backbone:
            model_class = VLBartVCR

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

        self.model = self.create_model(model_class, config)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)
            if self.verbose:
                print(f'Vocab resize: {self.tokenizer.vocab_size} -> {self.model.model.shared.num_embeddings}')
                assert self.model.model.shared.weight is self.model.lm_head.weight
                assert self.model.model.shared.weight is self.model.model.encoder.visual_embedding.obj_order_embedding.weight

        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

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
            loss_meter = LossMeter()
            qa_loss_meter = LossMeter()
            qar_loss_meter = LossMeter()

            best_valid_Q_AR = 0.
            best_epoch = 0

            if 't5' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLT5_VCR"
                else:
                    project_name = "T5_VCR"
            elif 'bart' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLBart_VCR"
                else:
                    project_name = "Bart_VCR"

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
                pbar = tqdm(total=len(self.train_loader), ncols=200)


            epoch_results = {
                'loss': 0,

            }

            Q_A_results = 0
            QA_R_results = 0
            Q_AR_results = 0
            n_total = 0

            n_accu = 0
            train_loss = 0
            train_qa_loss = 0
            train_qar_loss = 0

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

                if self.args.gradient_accumulation_steps > 1:
                    loss /= self.args.gradient_accumulation_steps

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

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

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    # if step_i == 0:
                    #     update = False
                    # elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                    #     update = True
                    # else:
                    #     update = False
                    update = ((step_i+1) % self.args.gradient_accumulation_steps == 0) or (step_i == len(self.train_loader) - 1)
                n_accu += 1

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # self.model.zero_grad()
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

                if self.verbose:
                    train_loss += loss.detach().item()
                    train_qa_loss += results['qa_loss'].item() / self.args.gradient_accumulation_steps
                    train_qar_loss += results['qar_loss'].item() / self.args.gradient_accumulation_steps

                    if self.args.log_train_accuracy:
                        qa_pred = results['qa_pred']
                        qar_pred = results['qar_pred']

                        a_labels = batch['answer_labels']
                        r_labels = batch['rationale_labels']

                        Q_A_correct = a_labels == qa_pred
                        QA_R_correct = r_labels == qar_pred
                        Q_AR_correct = Q_A_correct & QA_R_correct

                        Q_A_results += sum(Q_A_correct)
                        QA_R_results += sum(QA_R_correct)
                        Q_AR_results += sum(Q_AR_correct)
                        n_total += len(qa_pred)

                    if update:
                        if self.args.gradient_accumulation_steps > 1:
                            train_loss *= self.args.gradient_accumulation_steps / n_accu
                            train_qa_loss *= self.args.gradient_accumulation_steps / n_accu
                            train_qar_loss *= self.args.gradient_accumulation_steps / n_accu

                        loss_meter.update(train_loss)
                        qa_loss_meter.update(train_qa_loss)
                        qar_loss_meter.update(train_qar_loss)
                        desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step} |'
                        desc_str += f' Loss {loss_meter.val:.3f} |'
                        desc_str += f' QA Loss {qa_loss_meter.val:.3f} |'
                        desc_str += f' QAR Loss {qar_loss_meter.val:.3f} |'

                        train_loss = 0
                        train_qa_loss = 0
                        train_qar_loss = 0
                        n_accu = 0

                        if self.args.log_train_accuracy:
                            desc_str += f' Q -> A {Q_A_results} ({Q_A_results/n_total*100:.1f}%)'
                            desc_str += f' QA -> R {QA_R_results} ({QA_R_results/n_total*100:.1f}%)'
                            desc_str += f' Q -> AR {Q_AR_results} ({Q_AR_results/n_total*100:.1f}%)'

                        pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            if self.args.log_train_accuracy:
                train_score_dict = {
                    'Q_A': Q_A_results,
                    'QA_R': QA_R_results,
                    'Q_AR': Q_AR_results,
                    'n_total': n_total
                }
                train_score_dict = reduce_dict(train_score_dict, self.args.gpu)

            # Validation
            valid_score_dict = self.evaluate_val(self.val_loader)

            if self.verbose:
                if self.args.log_train_accuracy:
                    train_Q_A = train_score_dict['Q_A']/train_score_dict['n_total'] * 100
                    train_QA_R = train_score_dict['QA_R']/train_score_dict['n_total'] * 100
                    train_Q_AR = train_score_dict['Q_AR']/train_score_dict['n_total'] * 100
                    train_n_total = int(train_score_dict['n_total'])

                valid_Q_A = valid_score_dict['Q_A']/valid_score_dict['n_total'] * 100
                valid_QA_R = valid_score_dict['QA_R']/valid_score_dict['n_total'] * 100
                valid_Q_AR = valid_score_dict['Q_AR']/valid_score_dict['n_total'] * 100
                valid_n_total = int(valid_score_dict['n_total'])

                if valid_Q_AR > best_valid_Q_AR:
                    best_valid_Q_AR = valid_Q_AR
                    best_epoch = epoch
                    self.save("BEST")

                log_str = ''

                if self.args.log_train_accuracy:
                    log_str += f"\nEpoch {epoch}: Train |"
                    log_str += f" # examples: {train_n_total} |"
                    log_str += f" Q -> A {train_Q_A:.2f}%"
                    log_str += f" QA -> R {train_QA_R:.2f}%"
                    log_str += f" Q -> AR {train_Q_AR:.2f}%"

                log_str += f"\nEpoch {epoch}: Valid |"
                log_str += f" # examples: {valid_n_total} |"
                log_str += f" Q -> A {valid_Q_A:.2f}%"
                log_str += f" QA -> R {valid_QA_R:.2f}%"
                log_str += f" Q -> AR {valid_Q_AR:.2f}%"

                #log_str += "\nEpoch %d: Valid Q -> AR %0.2f" % (epoch, valid_Q_AR)
                log_str += f"\nBest Epoch {best_epoch}: Q -> AR {best_valid_Q_AR:.2f}%\n"

                wandb_log_dict = {}
                # wandb_log_dict['Train/Loss'] = loss_meter.val

                if self.args.log_train_accuracy:
                    wandb_log_dict['Train/Q_A'] = train_Q_A
                    wandb_log_dict['Train/QA_R'] = train_QA_R
                    wandb_log_dict['Train/Q_AR'] = train_Q_AR

                wandb_log_dict['Valid/Q_A'] = valid_Q_A
                wandb_log_dict['Valid/QA_R'] = valid_QA_R
                wandb_log_dict['Valid/Q_AR'] = valid_Q_AR

                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

        # Test Set
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)

        if self.verbose:
            dump_path = os.path.join(self.args.output, 'test_submit.csv')

            print('Dumping test set results at', dump_path)
            self.evaluate_test(self.test_loader, dump_path=dump_path)
            wandb.save(dump_path, base_path=self.args.output)

            print('Done!')

            wandb.log({'finished': True})

        if self.args.distributed:
            dist.barrier()
            exit()

    def evaluate_val(self, loader):
        self.model.eval()
        with torch.no_grad():

            uid2results = {}

            if self.verbose:
                iterator = tqdm(loader, ncols=200, desc="Prediction")
            else:
                iterator = loader

            for i, batch in enumerate(iterator):
                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                qa_pred = results['qa_pred']
                qar_pred = results['qar_pred']

                a_labels = batch['answer_labels']
                r_labels = batch['rationale_labels']

                for j, uid in enumerate(batch['uids']):
                    uid2results[uid] = (qa_pred[j], qar_pred[j], a_labels[j], r_labels[j])

        if self.args.distributed:
            dist.barrier()

        # print(f'GPU{self.args.gpu} # uid2results: {len(uid2results)}')

        uid2results_list = all_gather(uid2results)

        # print(f'GPU{self.args.gpu} # uid2results_list: {len(uid2results_list)}')

        uid2results = {}
        for uid2res in uid2results_list:
            for k, v in uid2res.items():
                uid2results[k] = v

        # print(f'GPU{self.args.gpu} after merge # uid2results: {len(uid2results)}')

        Q_A_results = 0
        QA_R_results = 0
        Q_AR_results = 0
        n_total = 0
        for uid, (a_pred, r_pred, a_label, r_label) in uid2results.items():

            Q_A_correct = a_label == a_pred
            QA_R_correct = r_label == r_pred
            Q_AR_correct = Q_A_correct & QA_R_correct

            Q_A_results += Q_A_correct
            QA_R_results += QA_R_correct
            Q_AR_results += Q_AR_correct
            n_total += 1

        score_dict = {}
        score_dict['Q_A'] = Q_A_results
        score_dict['QA_R'] = QA_R_results
        score_dict['Q_AR'] = Q_AR_results
        score_dict['n_total'] = n_total
        return score_dict

    def evaluate_test(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            submit_df = []
            for i, batch in enumerate(tqdm(loader, ncols=150, desc="Prediction")):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)
                answer_probs = results['answer_probs']
                rationale_probs = results['rationale_probs']

                B = len(batch['boxes'])

                for i in range(B):
                    annot_id = batch['annot_ids'][i]

                    submit_dict = {
                        'annot_id': annot_id
                    }

                    for answer_i in range(4):
                        submit_dict[f'answer_{answer_i}'] = answer_probs[i, answer_i]

                        for rationale_i in range(4):
                            submit_dict[f'rationale_conditioned_on_a{answer_i}_{rationale_i}'] = rationale_probs[i, answer_i, rationale_i]

                    submit_df.append(submit_dict)

        import pandas as pd

        # https://github.com/rowanz/r2c/tree/master/models#submitting-to-the-leaderboard
        # https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/example-submission.csv

        columns = [
            'annot_id',
            'answer_0',
            'answer_1',
            'answer_2',
            'answer_3',
            'rationale_conditioned_on_a0_0',
            'rationale_conditioned_on_a0_1',
            'rationale_conditioned_on_a0_2',
            'rationale_conditioned_on_a0_3',
            'rationale_conditioned_on_a1_0',
            'rationale_conditioned_on_a1_1',
            'rationale_conditioned_on_a1_2',
            'rationale_conditioned_on_a1_3',
            'rationale_conditioned_on_a2_0',
            'rationale_conditioned_on_a2_1',
            'rationale_conditioned_on_a2_2',
            'rationale_conditioned_on_a2_3',
            'rationale_conditioned_on_a3_0',
            'rationale_conditioned_on_a3_1',
            'rationale_conditioned_on_a3_2',
            'rationale_conditioned_on_a3_3'
        ]

        submit_df = pd.DataFrame(submit_df, columns=columns)

        if dump_path is None:
            dump_path = os.path.join(self.args.output, 'test_submit.csv')

        submit_df.to_csv(dump_path, index=False)

        return submit_dict

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    train_loader = val_loader = []
    if not args.test_only:
        print(f'Building train loader at GPU {gpu}')
        train_loader = get_loader(
            args,
            split=args.train, mode='train', batch_size=args.batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers,
            topk=args.train_topk,
        )

        print(f'Building val loader at GPU {gpu}')
        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.valid_batch_size,
            distributed=args.distributed, gpu=args.gpu,
            workers=args.num_workers,
            topk=args.valid_topk,
        )

    test_loader = None
    if gpu == 0:
        print(f'Building test loader at GPU {gpu}')
        test_loader = get_loader(
            args,
            split=args.test, mode='val', batch_size=args.valid_batch_size,
            distributed=False, gpu=args.gpu,
            workers=args.num_workers,
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
