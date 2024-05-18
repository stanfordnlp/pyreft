from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import csv
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

import preprocess
from qa_answer_table import AnswerTable


project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
vcr_dir = dataset_dir.joinpath('VCR/')
vcr_img_dir = vcr_dir.joinpath('vcr1images')
vcr_feature_dir = vcr_dir.joinpath('features')


# Load VG Classes
vg_classes = []
with open(vg_dir.joinpath('objects_vocab.txt')) as f:
    for obj in f.readlines():
        vg_classes.append(obj.split(',')[0].lower().strip())

vg_attrs = []
with open(vg_dir.joinpath('attributes_vocab.txt')) as f:
    for attr in f.readlines():
        vg_attrs.append(attr.split(',')[0].lower().strip())


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)


def get_datum(datum):
    data = []

    if 't5' in datum['backbone'] or 'bart' in datum['backbone']:
        # QA
        for task in datum['losses']:
            new_datum = deepcopy(datum)
            new_datum['vcr_task'] = 'QA'
            new_datum['task'] = task
            data.append(new_datum)

        # QAR
        for task in datum['losses']:
            new_datum = deepcopy(datum)
            new_datum['vcr_task'] = 'QAR'
            new_datum['task'] = task
            data.append(new_datum)

    return data



class VCRPretrainDataset(Dataset):
    def __init__(self, split='train', rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data sources: ', self.source)


        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.losses = args.losses.split(',')

        data_info_path = dataset_dir.joinpath(f'VCR/{self.source}.jsonl')
        with open(data_info_path) as f:
            data_info_dicts = [json.loads(s) for s in f]
            if self.topk > 0:
                data_info_dicts = data_info_dicts[:self.topk]
            for datum in data_info_dicts:
                datum['backbone'] = self.args.backbone
                datum['losses'] = self.losses

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data_info_dicts), total=len(data_info_dicts), ncols=100) for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data_info_dicts) for datum in _data]

        if self.verbose:
            print(f"Loaded {len(data)} data from", self.source)


        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose and is_train and ('t5' in self.args.backbone or 'bart' in self.args.backbone):
            from collections import Counter
            task_counter = Counter()
            for datum in data:
                try:
                    task_counter.update([datum['task']])
                except KeyError:
                    print(datum)
                    exit()

            print(task_counter)
            for k, v in task_counter.items():
                print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print("# examples:", len(data))

        self.source_to_h5 = {
            'train': vcr_feature_dir.joinpath(f'train_boxes36.h5'),
            'val': vcr_feature_dir.joinpath(f'val_boxes36.h5'),
            'test': vcr_feature_dir.joinpath(f'test_boxes36.h5'),

            'train_GT': vcr_feature_dir.joinpath(f'train_boxes_GT.h5'),
            'val_GT': vcr_feature_dir.joinpath(f'val_boxes_GT.h5'),
            'test_GT': vcr_feature_dir.joinpath(f'test_boxes_GT.h5'),
        }

        self.n_boxes = args.n_boxes


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # uid = datum['uid']
        # out_dict['uid'] = uid

        test = 'test' in datum['annot_id']
        out_dict['is_test'] = test

        ###### Image ######

        img_id = datum['img_id']
        out_dict['img_id'] = img_id

        img_path = vcr_img_dir.joinpath(datum['img_fn'])
        # assert img_path.exists()
        out_dict['img_path'] = img_path

        # source = self.img_ids_to_source[img_id]
        source = self.source

        f = self.source_to_h5[source]
        f_GT = self.source_to_h5[f'{source}_GT']

        if isinstance(f, Path):
            f = h5py.File(f, 'r')
            self.source_to_h5[source] = f

        if isinstance(f_GT, Path):
            f_GT = h5py.File(f_GT, 'r')
            self.source_to_h5[f'{source}_GT'] = f_GT

        img_h = f[f'{img_id}/img_h'][()]
        img_w = f[f'{img_id}/img_w'][()]
        gt_boxes = f_GT[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)

        n_gt_boxes = min(len(gt_boxes), 36)
        gt_boxes = gt_boxes[:n_gt_boxes]
        n_pred_boxes = 36 - n_gt_boxes

        pred_boxes = f[f'{img_id}/boxes'][:n_pred_boxes]

        boxes = np.concatenate([gt_boxes, pred_boxes], axis=0)

        # Normalize the boxes (to 0 ~ 1)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h

        np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(boxes, 1+5e-2)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        boxes = torch.from_numpy(boxes)

        assert boxes.size() == (36, 4), (boxes.size(),
                                            datum['img_id'], gt_boxes.shape, pred_boxes.shape)

        boxes.clamp_(min=0.0, max=1.0)

        out_dict['boxes'] = boxes

        gt_feats = f_GT[f'{img_id}/features'][:n_gt_boxes]
        pred_feats = f[f'{img_id}/features'][:n_pred_boxes]

        feats = np.concatenate([gt_feats, pred_feats], axis=0)

        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats

        pred_obj_ids = f[f'{img_id}/obj_id'][:n_pred_boxes]
        gt_obj_ids = f_GT[f'{img_id}/obj_id'][:n_gt_boxes]

        obj_ids = np.concatenate([gt_obj_ids, pred_obj_ids], axis=0)

        pred_attr_ids = f[f'{img_id}/attr_id'][:n_pred_boxes]
        gt_attr_ids = f_GT[f'{img_id}/attr_id'][:n_gt_boxes]

        attr_ids = np.concatenate([gt_attr_ids, pred_attr_ids], axis=0)

        ###### Text #####x
        question = datum['question']
        answers = datum['answer_choices']
        rationales = datum['rationale_choices']

        if not test:
            answer_label = datum['answer_label']
            rationale_label = datum['rationale_label']
            out_dict['answer_label'] = answer_label
            out_dict['rationale_label'] = rationale_label

        object_tags = f_GT[f'{img_id}/captions'][()].flatten().tolist()

        object_tags = [tag.decode() if isinstance(tag, bytes)
                       else tag for tag in object_tags]

        out_dict['annot_id'] = datum['annot_id']

        def flat(tokenized, names=None, max_len=None):
            tokens = []
            for token in tokenized[:max_len]:
                if isinstance(token, list):
                    for i, id in enumerate(token):
                        # if names is not None:
                        name = names[id]

                        if i > 0:
                            tokens.append('and')

                        tokens.append(name)
                        if 't5' in self.args.backbone or 'bart' in self.args.backbone:
                            token = f'<vis_extra_id_{id}>'
                            # token = str(id)
                            tokens.append(token)

                else:
                    tokens.append(token)

            tokens = tokens[:max_len]

            flat_str = " ".join(tokens)
            flat_str = flat_str.replace(' ?', '?').replace(" '", "'").replace(" !", "!").replace(" .", ".").replace(" ,", ",")
            flat_str = " ".join(flat_str.split())

            return flat_str

        q_max_len = None
        a_max_len = None
        r_max_len = None

        out_dict['qa_all_target_texts'] = []
        out_dict['qar_all_target_texts'] = []

        # QA
        if datum['vcr_task'] == 'QA':
            flat_q = flat(question, object_tags, max_len=q_max_len)
            answer = answers[answer_label]
            flat_a = flat(answer, object_tags, max_len=a_max_len)

            # input_text = f'vcr qa question: {flat_q} answer: {flat_a} relevant:'
            if 't5' in self.args.backbone or 'bart' in self.args.backbone:
                sent = f'question: {flat_q} answer: {flat_a}'

        # QAR
        elif datum['vcr_task'] == 'QAR':
            flat_q = flat(question, object_tags, max_len=q_max_len)
            answer = answers[answer_label]
            flat_a = flat(answer, object_tags, max_len=a_max_len)

            rationale = rationales[rationale_label]

            flat_r = flat(rationale, object_tags, max_len=r_max_len)

            if 't5' in self.args.backbone or 'bart' in self.args.backbone:
                sent = f'question: {flat_q} answer: {flat_a} rationale: {flat_r}'

        if 't5' in self.args.backbone or 'bart' in self.args.backbone:

            # text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                if 't5' in self.args.backbone:
                    prefix = "span prediction:"
                    source_text, target_text = preprocess.corrupt_spans(
                        sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)
                elif 'bart' in self.args.backbone:
                    prefix = "denoise text:"
                    source_text, target_text = preprocess.corrupt_bart(
                        sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

            elif task == 'caption':
                if datum['vcr_task'] == 'QA':
                    source_text = f'answer prediction: question: {flat_q}'
                    target_text = flat_a

                elif datum['vcr_task'] == 'QAR':
                    source_text = f'rationale prediction: question: {flat_q} answer: {flat_a}'
                    target_text = flat_r

            elif task == 'ground_caption':
                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "describe visual inputs:"
                prefix = "caption region:"
                source_text, target_text = preprocess.ground_caption(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            elif task == 'refer':
                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "refer expressions:"
                prefix = "visual grounding:"
                source_text, target_text = preprocess.refer_expression(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.max_text_length)

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        # args = batch[0]['args']
        args = self.args

        V_L = len(batch[0]['boxes'])

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].shape[-1]

        # args = batch[0]['args']

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'feat' in args.losses:
            feat_labels = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        # ans = torch.zeros(B, dtype=torch.long)
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            boxes[i] += entry['boxes']
            vis_feats[i] += entry['vis_feats']

            if 'ans' in entry:
                # ans[i] = entry['ans']
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            # uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        if 't5' in args.backbone or 'bart' in args.backbone:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['task'] = tasks

            batch_entry['source_text'] = source_text
            batch_entry['target_text'] = target_text

        batch_entry['loss_weights'] = loss_weights

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        # batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        return batch_entry


def get_loader(args, split='vcr_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):


    verbose = (gpu == 0)
    dataset = VCRPretrainDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        # distributed=distributed,
        is_train=(mode == 'train'),
        )


    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)


    return loader