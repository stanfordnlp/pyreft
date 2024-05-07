from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/video/').resolve()
activity_dir = dataset_dir.joinpath('ActivityNet-QA')


class ActivityNetQAFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.args.BUTD100 = False

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        source_dir = activity_dir.joinpath(f'{self.source}.csv')

        df = pd.read_csv(source_dir)
        data = []
        repetives_for_video = defaultdict(int)

        for index, row in df.iterrows():
            repetives_for_video[row["video_id"]] += 1
            data_dict = row.to_dict()
            data_dict["question_id"] = \
                f'{data_dict["video_id"]}_{repetives_for_video[row["video_id"]]}'
            data.append(data_dict) 
      
        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        self.source_features = torch.load(activity_dir.joinpath(f's3d.pth'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            video_id = datum['video_id']
            out_dict['video_id'] = video_id

            feats = self.source_features[video_id]
            out_dict['vis_feats'] = feats # (L, D)

            boxes = torch.zeros(feats.shape[0], 4) # (L, 4)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        if 'question' in datum:
            sent = datum['question']

        input_ids = self.tokenizer.encode(f'{self.args.prompt}{sent}', max_length=20, truncation=True)

        out_dict['question_id'] = datum['question_id']

        out_dict['type'] = datum['type']

        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        if 'answer' in datum:
            answer = datum['answer']

            out_dict['answer'] = answer

            target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if args.use_vision:
            V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        question_ids = []
        answers = []
        img_ids = []
        img_paths = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers

        batch_entry['args'] = args
        batch_entry['task'] = 'activitynet'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    dataset = ActivityNetQAFineTuneDataset(
        split,
        raw_dataset=None,
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

    if verbose:
        loader.evaluator = ActivityNetQAEvaluator(dataset.data)

    loader.task = 'activitynet'

    return loader


class ActivityNetQAEvaluator:
    # Codes borrowed from https://github.com/MILVLG/activitynet-qa/blob/master/evaluation/eval.py
    def __init__(self, dataset=None):
        self.dataset = dataset

    def eval(self, preds):
        corrects = {i: 0 for i in range(0, 9)}

        result_d = {}
        type_count = {i: 0 for i in range(0, 9)}
        for res in self.dataset:
            result_d[res['question_id']] = res
            res_type = res['type']
            type_count[res_type] += 1

        for pt in preds:
            pt_answer = pt['answer']
            pt_question_id = pt['question_id']
            pt_type = result_d[pt_question_id]['type']
            if pt_answer == result_d[pt_question_id]['answer']:
                corrects[pt_type] += 1

        return corrects, type_count

    def output(self, corrects, type_count):

        all_type_corrects_count = sum(corrects.values())
        free_type_corrects_count = sum(list(corrects.values())[3:])

        accuracy = {}
        for type_id in corrects:
            accuracy[type_id] = corrects[type_id] / (float(type_count[type_id]) + 1e-10)

        all_type_accuracy = all_type_corrects_count / (float(sum(type_count.values())) + 1e-10)

        free_type_accuracy = free_type_corrects_count / (float(sum(list(type_count.values())[3:])) + 1e-10)

        return {
            "motion_accuracy": accuracy[0], 
            "spatial_relation_accuracy": accuracy[1], 
            "temperal_relation_accuracy": accuracy[2], 
            "yes_no_accuracy": accuracy[3], 
            "color_accuracy": accuracy[4], 
            "object_accuracy": accuracy[5], 
            "location_accuracy": accuracy[6], 
            "number_accuracy": accuracy[7], 
            "other_accuracy": accuracy[8],
            "free_type_accuracy": free_type_accuracy,
            "all_type_accuracy": all_type_accuracy,
        }

        # print ('Accuracy (per question type):')
        # print('\tMotion: {:.04f}\n\tSpatial Relation: {:.04f}\n\tTemporal Relation: {:.04f}\n\tFree: {:.04f}\n\tAll: {:.04f}'.format(accuracy[0], accuracy[1], accuracy[2], free_type_accuracy, all_type_accuracy))
        # print ('Accuracy of the Free type questions(per answer type):')
        # print('\tYes/No: {:.04f}\n\tColor: {:.04f}\n\tObject: {:.04f}\n\tLocation: {:.04f}\n\tNumber: {:.04f}\n\tOther: {:.04f}'.format(accuracy[3], accuracy[4], accuracy[5], accuracy[6], accuracy[7], accuracy[8]))
