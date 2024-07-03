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
import numpy as np
import csv
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast, VLT5Tokenizer

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
vcr_dir = dataset_dir.joinpath('VCR')
vcr_img_dir = vcr_dir.joinpath('vcr1images')
vcr_feature_dir = vcr_dir.joinpath('features')


class VCRFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.split = split
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

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

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'VCR/{source}.jsonl')
            with open(data_info_path) as f:
                _data_info_dicts = [json.loads(s) for s in f]
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")


        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes


        self.source_to_h5 = {
            'train': vcr_feature_dir.joinpath(f'train_boxes36.h5'),
            'val': vcr_feature_dir.joinpath(f'val_boxes36.h5'),
            'test': vcr_feature_dir.joinpath(f'test_boxes36.h5'),

            'train_GT': vcr_feature_dir.joinpath(f'train_boxes_GT.h5'),
            'val_GT': vcr_feature_dir.joinpath(f'val_boxes_GT.h5'),
            'test_GT': vcr_feature_dir.joinpath(f'test_boxes_GT.h5'),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        uid = f"{datum['img_id']}_{datum['question_number']}"
        out_dict['uid'] = uid

        test = 'test' in datum['annot_id']
        out_dict['is_test'] = test

        ###### Image ######
        assert self.args.use_vision
        img_id = datum['img_id']
        out_dict['img_id'] = img_id

        img_path = vcr_img_dir.joinpath(datum['img_fn'])
        # assert img_path.exists()
        out_dict['img_path'] = img_path

        source = self.img_ids_to_source[img_id]

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
        np.testing.assert_array_less(-boxes, 0+1e-5)
        boxes = torch.from_numpy(boxes)

        assert boxes.size() == (36, 4), (boxes.size(), datum['img_id'], gt_boxes.shape, pred_boxes.shape)

        boxes.clamp_(min=0.0, max=1.0)

        out_dict['boxes'] = boxes

        gt_feats = f_GT[f'{img_id}/features'][:n_gt_boxes]

        pred_feats = f[f'{img_id}/features'][:n_pred_boxes]

        feats = np.concatenate([gt_feats, pred_feats], axis=0)

        feats = torch.from_numpy(feats)

        assert feats.size() == (36, 2048), (feats.size(), datum['img_id'], gt_feats.shape, pred_feats.shape)

        out_dict['vis_feats'] = feats

        ###### Text #####

        # question = datum['question_orig']
        question = datum['question']
        answers = datum['answer_choices']
        rationales = datum['rationale_choices']

        if not test:
            answer_label = datum['answer_label']
            rationale_label = datum['rationale_label']
            out_dict['answer_label'] = answer_label
            out_dict['rationale_label'] = rationale_label

        object_tags = f_GT[f'{img_id}/captions'][()].flatten().tolist()

        object_tags = [tag.decode() if isinstance(tag, bytes) else tag for tag in object_tags]

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
        qa_input_texts = []
        qa_target_texts = []

        flat_q = flat(question, object_tags, max_len=q_max_len)

        for i, answer in enumerate(answers):
            flat_a = flat(answer, object_tags, max_len=a_max_len)

            input_text = f'vcr qa question: {flat_q} answer: {flat_a}'
            qa_input_texts.append(input_text)

            if not test:
                if i == answer_label:
                    target_text = 'true'
                else:
                    target_text = 'false'
                qa_target_texts.append(target_text)

        out_dict['qa_input_texts'] = qa_input_texts
        if not test:
            out_dict['qa_target_texts'] = qa_target_texts

        # QAR
        qar_input_texts = []
        qar_target_texts = []

        if test:
            for answer in answers:
                flat_a = flat(answer, object_tags, max_len=a_max_len)

                for i, rationale in enumerate(rationales):
                    flat_r = flat(rationale, object_tags)

                    input_text = f'vcr qar question: {flat_q} answer: {flat_a} rationale: {flat_r}'
                    qar_input_texts.append(input_text)

        else:
            flat_a = flat(answers[answer_label], object_tags, max_len=a_max_len)

            for i, rationale in enumerate(rationales):
                flat_r = flat(rationale, object_tags, max_len=r_max_len)

                input_text = f'vcr qar question: {flat_q} answer: {flat_a} rationale: {flat_r}'
                qar_input_texts.append(input_text)

                if i == rationale_label:
                    target_text = 'true'
                else:
                    target_text = 'false'
                qar_target_texts.append(target_text)

        out_dict['qar_input_texts'] = qar_input_texts

        if not test:
            out_dict['qar_target_texts'] = qar_target_texts

        return out_dict

    def collate_fn(self, batch):

        batch_entry = {}

        args = self.args

        B = len(batch)

        test = batch[0]['is_test']

        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        qa_input_texts = []
        qa_target_texts = []
        qar_input_texts = []
        qar_target_texts = []

        qa_all_target_texts = []
        qar_all_target_texts = []

        answer_labels = []
        rationale_labels = []

        img_ids = []
        annot_ids = []
        uids = []

        for i, entry in enumerate(batch):

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']

            qa_input_texts.extend(entry['qa_input_texts'])
            qar_input_texts.extend(entry['qar_input_texts'])

            if not test:
                qa_target_texts.extend(entry['qa_target_texts'])
                qar_target_texts.extend(entry['qar_target_texts'])

                qa_all_target_texts.append(entry['qa_all_target_texts'])
                qar_all_target_texts.append(entry['qar_all_target_texts'])

                answer_labels.append(entry['answer_label'])
                rationale_labels.append(entry['rationale_label'])

            annot_ids.append(entry['annot_id'])
            img_ids.append(entry['img_id'])
            uids.append(entry['uid'])

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats

        batch_entry['qa_input_texts'] = qa_input_texts
        batch_entry['qar_input_texts'] = qar_input_texts

        batch_entry['qa_input_ids'] = self.tokenizer(qa_input_texts, padding=True, truncation=True, max_length=args.max_text_length, return_tensors='pt').input_ids
        batch_entry['qar_input_ids'] = self.tokenizer(qar_input_texts, padding=True, truncation=True, max_length=args.max_text_length, return_tensors='pt').input_ids

        if not test:

            batch_entry['answer_labels'] = np.array(answer_labels)
            batch_entry['rationale_labels'] = np.array(rationale_labels)

            batch_entry['qa_target_texts'] = qa_target_texts
            batch_entry['qar_target_texts'] = qar_target_texts

            qa_target_ids = self.tokenizer(qa_target_texts, padding=True, return_tensors='pt').input_ids
            qar_target_ids = self.tokenizer(qar_target_texts, padding=True, return_tensors='pt').input_ids

            if 't5' in self.args.backbone:
                qa_target_ids = qa_target_ids[:, :1]
                qar_target_ids = qar_target_ids[:, :1]
            elif 'bart' in self.args.backbone:
                qa_target_ids = qa_target_ids[:, :2]
                qar_target_ids = qar_target_ids[:, :2]

            batch_entry['qa_target_ids'] = qa_target_ids
            batch_entry['qar_target_ids'] = qar_target_ids

            batch_entry['qa_all_target_texts'] = qa_all_target_texts
            batch_entry['qar_all_target_texts'] = qar_all_target_texts

        batch_entry['annot_ids'] = annot_ids
        batch_entry['img_ids'] = img_ids

        batch_entry['uids'] = uids

        batch_entry['task'] = 'vcr'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = VCRFineTuneDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

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

    loader.task = 'vcr'

    return loader

