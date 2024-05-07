import sacrebleu
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

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent

dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')

flickr30k_dir = dataset_dir.joinpath('flickr30k')
flickr30k_feature_dir = flickr30k_dir.joinpath('features')
wmt_data_dir = dataset_dir.joinpath('multi30k-dataset/data/task1/')


class MMTDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

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

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        with open(wmt_data_dir.joinpath(f'raw/{self.source}.en')) as f:
            source_text_list = f.readlines()

        with open(wmt_data_dir.joinpath(f'raw/{self.source}.de')) as f:
            target_text_list = f.readlines()

        with open(wmt_data_dir.joinpath(f'image_splits/{self.source}.txt')) as f:
            image_ids = f.readlines()


        assert len(source_text_list) == len(target_text_list)
        assert len(source_text_list) == len(image_ids)

        data = []
        for source_text, target_text, image_id in zip(source_text_list, target_text_list, image_ids):
            datum = {
                'img_id': image_id.strip().split('.')[0],
                'source_text': source_text.strip(),
                'target_text': target_text.strip()
            }
            data.append(datum)

        if self.verbose:
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.source_to_h5 = {
            'train': flickr30k_feature_dir.joinpath('trainval_boxes36.h5'),
            'val': flickr30k_feature_dir.joinpath('trainval_boxes36.h5'),
            'test_2016_flickr': flickr30k_feature_dir.joinpath('trainval_boxes36.h5'),
            'test_2017_flickr': flickr30k_feature_dir.joinpath('test2017_boxes36.h5'),
            'test_2018_flickr': flickr30k_feature_dir.joinpath('test2018_boxes36.h5'),
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            source = self.source
            f = self.source_to_h5[source]

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5[source] = f

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            if self.args.max_n_boxes == 100:
                assert n_boxes == 100
                assert len(feats) == 100
                assert len(boxes) == 100

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            out_dict['boxes'] = boxes[:n_boxes]
            out_dict['vis_feats'] = feats[:n_boxes]

        ###### Text #####
        prefix = "translate English to German:"
        input_tokens = [prefix]
        source_text = datum['source_text']
        input_tokens.append(source_text)

        if self.args.oscar_tags:
            input_tokens.append('tags:')
            obj_ids = f[f'{img_id}/obj_id'][()]
            for obj_id in obj_ids:
                obj = self.vg_classes[obj_id]
                if obj not in input_tokens:
                    input_tokens.append(obj)
        input_text = ' '.join(input_tokens)

        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])

        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)


        target_text = datum['target_text']
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            target_ids = self.tokenizer.encode(
                target_text, max_length=self.args.gen_max_length, truncation=True)
        else:
            target_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(target_text)[:self.args.gen_max_length - 1] + ['[SEP]'])

        assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
        out_dict['target_text'] = target_text
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
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        img_ids = []
        img_paths = []
        input_text = []
        target_text = []
        # targets = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            if 'target_text' in entry:
                target_text.append(entry['target_text'])

                # targets.append([entry['target_text']])

            # sentences.append(entry['sent'])

            # if 'targets' in entry:
            #     targets.append(entry['targets'])


        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths
        else:
            batch_entry['boxes'] = torch.zeros(B, 0, 4)
            batch_entry['vis_feats'] = torch.zeros(B, 0, 2048)
            batch_entry['vis_attention_mask'] = torch.zeros(B, 0)

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text
        batch_entry['target_text'] = target_text
        # batch_entry['targets'] = targets

        # batch_entry['args'] = args
        batch_entry['task'] = 'mmt'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = MMTDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset)

    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = MMTEvaluator()

    loader.task = 'mmt'

    return loader


class MMTEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predicts, answers):
        """
        import sacrebleu

        refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
                ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
        sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

        bleu = sacrebleu.corpus_bleu(sys, refs)

        bleu.score
        48.530827009929865
        """

        try:
            bleu = sacrebleu.corpus_bleu(predicts, answers,
                                     lowercase=True)
        except EOFError:
            print('# preds', len(predicts))
            print('# tgts', len(answers))
            exit()
        return {
            'BLEU': bleu.score
        }

    # def dump_result(self, quesid2ans: dict, path):
    #     """
    #     Dump results to a json file, which could be submitted to the VQA online evaluation.
    #     VQA json file submission requirement:
    #         results = [result]
    #         result = {
    #             "question_id": int,
    #             "answer": str
    #         }
    #     :param quesid2ans: dict of quesid --> ans
    #     :param path: The desired path of saved file.
    #     """
    #     with open(path, 'w') as f:
    #         result = []
    #         for ques_id, ans in quesid2ans.items():
    #             result.append({
    #                 'question_id': ques_id,
    #                 'answer': ans
    #             })
    #         json.dump(result, f, indent=4, sort_keys=True)
