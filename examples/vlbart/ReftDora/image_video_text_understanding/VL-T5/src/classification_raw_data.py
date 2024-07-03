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
from copy import deepcopy
import re
import glob

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

import torchvision as tv
from vis_encoder import _transform


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
imagenet_dir = dataset_dir.joinpath('tiny-imagenet-200')
imagenet_feature_dir = imagenet_dir.joinpath('clip_features')


def get_class_name(path):
    class_to_name = dict()
    fp = open(path, 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()
    return class_to_name

# Need to be modified
class ImageNetFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
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

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        data = glob.glob(
            str(imagenet_feature_dir.joinpath(f"{split}/data_clip_RN101_att/*"))
        )

        print(f"length of {split}: {len(data)}")

        self.label_dict = get_class_name(
            imagenet_dir.joinpath("words.txt")
        )

        self.gts_for_data = self.__get_labels_from_data(data)

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

        self.feature_type = self.args.feature_type

    def __get_labels_from_data(self, data):
        gts_for_data = {}
        for d in data:
            img_id = d.split("/")[-1].split("_")[0]
            
            label = self.label_dict[img_id]

            gts_for_data[d] = label

        return gts_for_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        img_id = datum.split("/")[-1].split(".")[0]

        ###### Image ######
        with h5py.File(datum, 'r') as f:

            feats = f[f"{img_id}/features"][...]
            out_dict['vis_feats'] = feats # (L, D)

            boxes = torch.zeros(feats.shape[0], 4) # (L, 4)

            out_dict['boxes'] = boxes

        ###### Text #####
        input_ids = self.tokenizer.encode(f'{self.args.prompt}', max_length=20, truncation=True)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        answer_code = img_id.split("_")[0]

        answer = self.label_dict[answer_code]

        target_ids = self.tokenizer.encode(answer, max_length=20, truncation=True)

        out_dict['question_id'] = datum
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['answer'] = answer


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

        answers = []
        img_ids = []
        img_paths = []
        question_ids = []

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

            if 'answer' in entry:
                answers.append(entry['answer'])

            if 'question_id' in entry:
                question_ids.append(entry['question_id'])

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

        batch_entry['answers'] = answers
        batch_entry['question_ids'] = question_ids

        batch_entry['args'] = args
        batch_entry['task'] = 'cls'

        return batch_entry



class CIFAR10(tv.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, args=None):
        super().__init__(root, train, transform, target_transform, download)

        self.args = args
        self.image_size = eval(self.args.image_size)
        self.transform = _transform(self.image_size)


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


        self.gts_for_data = self.__get_labels_from_data(self.data)

    def __get_labels_from_data(self, data):
        gts_for_data = {}
        for index, _ in enumerate(data):
            target = self.targets[index]
            
            label = self.classes[target]

            gts_for_data[index] = label

        return gts_for_data

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        out_dict = {}
        out_dict['args'] = self.args

        out_dict['image'] = img

        ###### Text #####
        input_ids = self.tokenizer.encode(f'{self.args.prompt}', max_length=20, truncation=True)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        answer = self.classes[target]

        target_ids = self.tokenizer.encode(answer, max_length=20, truncation=True)

        out_dict['question_id'] = index
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['answer'] = answer

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        answers = []
        img_ids = []
        img_paths = []
        question_ids = []
        images = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            images.append(entry['image'])
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            if 'answer' in entry:
                answers.append(entry['answer'])

            if 'question_id' in entry:
                question_ids.append(entry['question_id'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        batch_entry['images'] = torch.stack(images)

        batch_entry['answers'] = answers
        batch_entry['question_ids'] = question_ids

        batch_entry['args'] = args
        batch_entry['task'] = 'cls'

        return batch_entry



def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    if args.cls_task == "tinyimagenet":
        dataset = ImageNetFineTuneDataset(
            split,
            rank=gpu,
            topk=topk,
            verbose=verbose,
            args=args,
            mode=mode)

    elif args.cls_task == "cifar10":
        dataset = CIFAR10(
            ".",
            train=(mode == 'train'), 
            download=True, 
            args=args,
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

    loader.task = 'cls'

    if verbose:
        loader.gts_for_data = dataset.gts_for_data

    return loader
