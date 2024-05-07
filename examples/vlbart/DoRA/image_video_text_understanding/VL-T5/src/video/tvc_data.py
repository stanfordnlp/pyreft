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

import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/video/').resolve()

TASK = "tvc"


def resize(input_tensor, length):
    L, D = input_tensor.shape
    if L < length:
        # pad
        input_tensor = torch.cat([input_tensor, torch.zeros(length - L, D)], dim=0)
    elif L > length:
        input_tensor = input_tensor.t()
        input_tensor = F.adaptive_max_pool1d(input_tensor, length)
        input_tensor = input_tensor.t()
    
    return input_tensor

class TVCFineTuneDataset(Dataset):
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

        subtitles_path = dataset_dir.joinpath(f"ann/tv_subtitles.jsonl")

        self.subtitles = {}
        with open(subtitles_path, "r") as f:
            for line in f:
                d = json.loads(line)
                self.subtitles[d["vid_name"]] = d["sub"]

        annotations = [
            dataset_dir.joinpath(f'ann/tvc/tvc_{s}.jsonl')
            for s in self.source.split(",")
        ]

        self.source_dir = dataset_dir.joinpath(f'vis_features/tv/clip-vit')

        data = []
        for ann in annotations:
            with open(ann) as f:
                for line in f:
                    d = json.loads(line)
                    # train data
                    if "descs" in d:
                        for desc in d["descs"]:
                            datum = {}
                            datum["vid_name"] = d["vid_name"]
                            datum["duration"] = d["duration"]
                            datum["ts"] = d["ts"]
                            datum["answer"] = desc["desc"]
                            datum["question_id"] = desc["desc_id"]
                            datum["type"] = desc["type"]
                            data.append(datum)
                    # test data
                    else:
                        datum = {}
                        datum["vid_name"] = d["vid_name"]
                        datum["duration"] = d["duration"]
                        datum["ts"] = d["ts"]
                        datum["question_id"] = d["clip_id"]
                        datum["type"] = f"{TASK}"
                        data.append(datum)

        self.types = set()

        for d in data:
            self.types.add(d["type"])

        print(self.types)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            video_id = datum['vid_name']
            out_dict['video_id'] = video_id

            path = self.source_dir.joinpath(f"{video_id}.npz")
            feats = np.load(path)["features"]

            feats = torch.from_numpy(feats)
            feats = resize(feats, self.n_boxes)

            out_dict['vis_feats'] = feats # (L, D)

            boxes = torch.zeros(feats.shape[0], 4) # (L, 4)

            out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']

        sent = ""

        subs = []

        for t in self.subtitles[video_id]:
            # subs.append(f"({t['start']:.1f}-{t['end']:.1f}) {t['text'].strip()}")
            subs.append(f"{t['text'].strip()}")

        subs = " ".join(subs)

        subs = f"[Subs] {subs}"

        # duration

        duration = f"[TS] ({datum['ts']})"

        sent = " ".join([subs, duration])

        input_ids = self.tokenizer.encode(f"{sent} {self.args.prompt}", max_length=600, truncation=True)

        out_dict['question_id'] = datum['question_id']

        out_dict['type'] = datum['type']
        out_dict['ts'] = datum['ts']

        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        if 'answer' in datum:
            answer = datum['answer']
            out_dict['answer'] = answer
            # print(feats.shape, feats.dtype)
            # print(sent)
            # print(answer)

            target_ids = self.tokenizer.encode(answer, max_length=20, truncation=True)

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
        video_ids = []
        tss = []

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
            video_ids.append(entry["video_id"])
            tss.append(entry['ts'])
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
        batch_entry['video_ids'] = video_ids
        batch_entry['tss'] = tss

        batch_entry['args'] = args
        batch_entry['task'] = TASK

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    dataset = TVCFineTuneDataset(
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
        loader.evaluator = TVCEvaluator(dataset.data, dataset.types)

    loader.task = TASK

    return loader


class TVCEvaluator:
    def __init__(self, data, types):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results


if __name__ == "__main__":
    from param import parse_args

    args = parse_args()

    d = TVCFineTuneDataset(split='train_release', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=args, mode='train')

    for i in range(3):
       print(d[i])