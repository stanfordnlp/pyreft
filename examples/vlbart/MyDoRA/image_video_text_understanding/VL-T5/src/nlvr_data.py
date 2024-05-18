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
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
vqa_dir = dataset_dir.joinpath('vqa')
nlvr_dir = dataset_dir.joinpath('nlvr')
nlvr_feature_dir = nlvr_dir.joinpath('features')



class NLVRFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.split = split
        if self.verbose:
            print('Data source: ', self.split)

        data = self.raw_dataset.data

        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        self.data = data

        if self.verbose:
            # if 'sent' not in self.data_out:
            #     print("# all images:", len(self.data))
            # else:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        if 't5' in self.args.backbone:
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

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.source_to_h5 = {
            'train': nlvr_feature_dir.joinpath(f'train_obj36.h5'),
            'valid': nlvr_feature_dir.joinpath(f'valid_obj36.h5'),
            'test': nlvr_feature_dir.joinpath(f'test_obj36.h5'),
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        ###### Image ######
        source = self.split

        f = self.source_to_h5[source]

        if isinstance(f, Path):
            # path = self.data_source_to_h5_path[source]
            f = h5py.File(f, 'r')
            # self.split_to_h5_features[split_i] = f
            self.source_to_h5[source] = f

        feats2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            feats = np.zeros(
                shape=(self.n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats2.append(feats)
        feats = np.stack(feats2)  # [2, n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats

        # Normalize the boxes (to 0 ~ 1)
        boxes2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes2.append(boxes)
        boxes = np.stack(boxes2)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        out_dict['boxes'] = boxes

        ###### Text #####
        # caption = datum['caption']
        sent = datum['sent']

        input_ids = self.tokenizer.encode(f'{self.args.prompt}{sent}{self.args.post_prompt}')

        question_id = uid
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        if 'label' in datum:
            label = datum['label']

            if label == 1:
                answer = 'true'
            elif label == 0:
                answer = 'false'

            out_dict['answer'] = answer

            target_ids = self.tokenizer.encode(answer)

            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
        else:
            label = None
        out_dict['label'] = label

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)
        V_L = batch[0]['boxes'].size(1)
        S_W_L = max(entry['input_length'] for entry in batch)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].size(-1)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, 2, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, 2, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, dtype=torch.long)
        if 'target_ids' in batch[0]:
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if batch[0]['label'] is not None:
            labels = torch.zeros(B, dtype=torch.long)


        sentences = []
        question_ids = []
        answers = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            boxes[i] += entry['boxes']
            vis_feats[i] += entry['vis_feats']

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])

            if entry['label'] is not None:
                labels[i] += entry['label']

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers

        if batch[0]['label'] is not None:
            batch_entry['labels'] = labels

        batch_entry['task'] = 'nlvr'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = NLVR2Dataset(split, verbose)

    dataset = NLVRFineTuneDataset(
        split,
        raw_dataset=_dset,
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
        loader.evaluator = NLVR2Evaluator(_dset)

    loader.task = 'nlvr'

    return loader


class RandomNonreplacmentSampler(Sampler):
    def __init__(self, data_source=None, num_samples=None, shuffle=True, seed=0):
        self.data_source = data_source
        self._num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        assert len(data_source) >= num_samples

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class NLVR2Dataset:
    """
    An NLVR2 data example in json file:
    {
        "identifier": "train-10171-0-0",
        "img0": "train-10171-0-img0",
        "img1": "train-10171-0-img1",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.",
        "uid": "nlvr2_train_0"
    }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(nlvr_dir.joinpath(f'{split}.json'))))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {}
        self.identifier2uid = {}
        for datum in self.data:
            self.id2datum[datum['uid']] = datum

            self.identifier2uid[datum['identifier']] = datum['uid']


    def __len__(self):
        return len(self.data)


class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate_train(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer
        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))

    def evaluate(self, quesid2ans: dict):
        # https://github.com/lil-lab/nlvr/blob/master/nlvr2/eval/metrics.py

        labeled_examples = self.dataset.data
        predictions = quesid2ans

        total_num = len(labeled_examples)
        if len(predictions) < total_num:
            print("Some predictions are missing!")
            print("Got " + str(len(predictions)) + " predictions but expected " + str(total_num))

            for example in labeled_examples:
                identifier = example["identifier"]
                uid = self.dataset.identifier2uid[identifier]
                if not uid in predictions:
                    print("Missing prediction for item " + str(identifier))
            exit()

        num_correct = 0.
        consistency_dict = {}

        for example in labeled_examples:
            anon_label = example["identifier"].split("-")
            anon_label[2] = ''
            anon_label = '-'.join(anon_label)
            if not anon_label in consistency_dict:
                consistency_dict[anon_label] = True
            identifier = example["identifier"]
            uid = self.dataset.identifier2uid[identifier]
            prediction = quesid2ans[uid]
            # if prediction.lower() == example["label"].lower():
            if int(prediction) == int(example["label"]):
                num_correct += 1.
            else:
                consistency_dict[anon_label] = False

        # Calculate consistency.
        num_consistent = 0.
        unique_sentence = len(consistency_dict)
        for identifier, consistent in consistency_dict.items():
            if consistent:
                num_consistent += 1

        score_dict = {}
        accuracy = num_correct / total_num
        consistency = num_consistent / unique_sentence
        score_dict['accuracy'] = accuracy
        score_dict['consistency'] = consistency

        return score_dict
