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


from torch.utils.data.distributed import DistributedSampler


import transformers
from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

from qa_answer_table import AnswerTable

from PIL import Image
from vis_encoder import _transform
from vqa_raw_data import augmentation_transform


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
gqa_dir = dataset_dir.joinpath('GQA')

gqa_img_dir = gqa_dir.joinpath('images/')
vg_img_dir = vg_dir.joinpath('VG_100K/')


class GQAFineTuneDataset(Dataset):
    def __init__(self, split='train,valid', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
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

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            data_info_path = dataset_dir.joinpath(f'GQA/{source}.json')
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                # source_img_ids.append([d['img_id'] for d in _data_info_dicts])
                for _d in _data_info_dicts:
                    self.img_ids_to_source[_d['img_id']] = source
                    _d['source'] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

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

        self.image_size = eval(self.args.image_size)

        if mode == "train" and self.args.use_data_augmentation:
            self.transform = augmentation_transform(self.image_size)
        else:
            self.transform = _transform(self.image_size)

        self.source_to_featname = {
            'train': 'others',
            'valid': 'others',
            'submit': 'testdev',

            'testdev': 'testdev'
        }

        self.featname_to_h5 = {
            'others': vg_img_dir,
            'testdev': gqa_img_dir,
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # uid = datum['uid']
        # out_dict['uid'] = uid
        # out_dict['uid'] = uid

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            source = self.img_ids_to_source[img_id]

            featname = self.source_to_featname[source]

            # f = self.source_to_h5[source]
            # f = self.featname_to_h5[featname]

            # if isinstance(f, Path):
            #     # path = self.data_source_to_h5_path[source]
            #     f = h5py.File(f, 'r')
            #     # self.split_to_h5_features[split_i] = f
            #     # self.source_to_h5[source] = f
            #     self.featname_to_h5[featname] = f

            # feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            # f[f'{img_id}/features'].read_direct(feats)
            # feats = torch.from_numpy(feats)
            # out_dict['vis_feats'] = feats

            # # Normalize the boxes (to 0 ~ 1)
            # img_h = f[f'{img_id}/img_h'][()]
            # img_w = f[f'{img_id}/img_w'][()]
            # boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            # boxes[:, (0, 2)] /= img_w
            # boxes[:, (1, 3)] /= img_h
            # np.testing.assert_array_less(boxes, 1+1e-5)
            # # np.testing.assert_array_less(boxes, 1+5e-2)
            # np.testing.assert_array_less(-boxes, 0+1e-5)
            # boxes = torch.from_numpy(boxes)

            # boxes.clamp_(min=0.0, max=1.0)

            # out_dict['boxes'] = boxes

            path = self.featname_to_h5[featname].joinpath(f"{img_id}.jpg")
        
            image = Image.open(path)

            out_dict["image"] = self.transform(image)

        ###### Text #####
        # caption = datum['caption']
        sent = datum['sent']


        input_ids = self.tokenizer.encode(f'{self.args.prompt}{sent}{self.args.post_prompt}', max_length=20, truncation=True)
        question_id = datum['question_id']
        out_dict['question_id'] = question_id

        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191
            answers = []
            scores = []
            for a, s in label.items():
                answers.append(a)
                scores.append(s)

            score_sum = sum(scores)

            if score_sum == 0:
                answer = ''
                score = 0.
            else:
                prob = [score / score_sum for score in scores]
                choice = np.random.multinomial(1, prob).argmax()
                answer = answers[choice]
                score = scores[choice]
                assert len(answer) > 0, (sent, label, choice, answer)

            out_dict['answer'] = answer
            out_dict['score'] = score
            out_dict['all_answers'] = answers

            if sum(scores) > 0:
                best_answers = []
                best_score = max(scores)
                for a, s in label.items():
                    if s == best_score and s > 0:
                        best_answers.append(a)
                out_dict['best_answers_tokenized'] = [self.tokenizer.encode(a) for a in best_answers]
            else:
                out_dict['best_answers_tokenized'] = [[]]

            target_ids = self.tokenizer.encode(answer)

            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # if args.use_vision:
        #     V_L = len(batch[0]['boxes'])
        #     feat_dim = batch[0]['vis_feats'].shape[-1]

        #     boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        #     vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        all_answers_tokenized = []
        best_answers_tokenized = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []

        images = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if args.use_vision:
                # boxes[i] += entry['boxes']
                # vis_feats[i] += entry['vis_feats']
                images.append(entry["image"])
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
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'all_answers_tokenized' in entry:
                all_answers_tokenized.append(entry['all_answers_tokenized'])
            if 'best_answers_tokenized' in entry:
                best_answers_tokenized.append(entry['best_answers_tokenized'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            # batch_entry['boxes'] = boxes
            # batch_entry['vis_feats'] = vis_feats
            batch_entry['images'] = torch.stack(images)
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['all_answers_tokenized'] = all_answers_tokenized
        batch_entry['best_answers_tokenized'] = best_answers_tokenized
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['task'] = 'gqa'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1, verbose=None):

    if verbose is None:
        verbose = (gpu == 0)

    _dset = GQADataset(split, verbose)

    dataset = GQAFineTuneDataset(
        split,
        raw_dataset=_dset,
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

    loader.evaluator = GQAEvaluator(_dset)
    loader.task = 'gqa'

    return loader


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(gqa_dir.joinpath("%s.json" % split))))
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(gqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(open(gqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }
        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)
