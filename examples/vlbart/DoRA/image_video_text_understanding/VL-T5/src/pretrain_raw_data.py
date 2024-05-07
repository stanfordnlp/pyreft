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
from copy import deepcopy


from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from tokenization import VLT5Tokenizer, VLT5TokenizerFast

import preprocess
from qa_answer_table import AnswerTable

from PIL import Image
from vis_encoder import _transform
from vqa_raw_data import augmentation_transform

project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
vg_img_dir = vg_dir.joinpath('VG_100K/')

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
    _sents = []

    args = datum['args']

    if datum['is_train']:
        if 'COCO_train2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_train2014'
        elif 'COCO_val2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_val2014'
        else:
            img_source = 'vgnococo'
    else:
        img_source = 'mscoco_resplit_val'

    for text_source, sents in datum['sentf'].items():
        if datum['caption_only']:
            if text_source not in ['mscoco', 'vg']:
                continue

        if args.coco_only:
            if text_source != 'mscoco':
                continue

        labels = None
        if datum['qa'] and text_source in datum['labelf']:
            labels = datum['labelf'][text_source]

        img_id = datum['img_id']

        for sent_idx, sent in enumerate(sents):

            if ('t5' in datum['backbone'] or 'bart' in datum['backbone']) and len(sent.split()) <= 2:
                continue

            # remove duplicate sentence
            if sent in _sents:
                continue

            new_datum = {
                'uid': make_uid(img_id, text_source, sent_idx),
                'img_id': img_id,
                'img_source': img_source,
                'sent': sent,
                'text_source': text_source
            }

            # Task: QA
            if datum['qa'] and labels is not None:
                label = labels[sent_idx]
                if ('t5' in datum['backbone'] or 'bart' in datum['backbone']) and len(label) == 0:
                    continue
                else:
                    # assert len(label) > 0, (img_id, labels, sent_idx, label)
                    # can have len = 0
                    new_datum = deepcopy(new_datum)
                    new_datum['task'] = 'qa'
                    new_datum['label'] = label
                    data.append(new_datum)

            # Task: Language modeling
            if datum['lm'] and labels is None:
                new_datum = deepcopy(new_datum)
                new_datum['task'] = 'lm'
                new_datum['label'] = None
                data.append(new_datum)

            # Task: Image captioning
            if datum['caption']:
                if args.caption_cocoonly:
                    if text_source == 'mscoco':
                        new_datum = deepcopy(new_datum)
                        new_datum['task'] = 'caption'
                        new_datum['label'] = None
                        data.append(new_datum)
                else:
                    if text_source in ['mscoco', 'vg']:
                        new_datum = deepcopy(new_datum)
                        new_datum['task'] = 'caption'
                        new_datum['label'] = None
                        data.append(new_datum)

            # Task: Image-text matching
            if args.itm_cocoonly:
                caption_sources = ['mscoco']
            else:
                caption_sources = ['mscoco', 'vg']
            if datum['itm'] and text_source in caption_sources:
                new_datum = deepcopy(new_datum)
                new_datum['task'] = 'itm'
                new_datum['label'] = None
                data.append(new_datum)

            _sents.append(sent)
    # Not use grounding tasks for 
    # if datum['ground_caption']:
    #     for i in range(args.ground_upsample):
    #         new_datum = {
    #             'uid': make_uid(img_id, 'ground_caption', i),
    #             'img_id': img_id,
    #             'img_source': img_source,
    #             'task': 'ground_caption',
    #             'text_source': 'ground_caption',
    #             'sent': None,
    #             'label': None,
    #         }
    #         data.append(new_datum)
    # if datum['refer']:
    #     for i in range(args.ground_upsample):
    #         new_datum = {
    #             'uid': make_uid(img_id, 'refer', i),
    #             'img_id': img_id,
    #             'img_source': img_source,
    #             'task': 'refer',
    #             'text_source': 'refer',
    #             'sent': None,
    #             'label': None,
    #         }
    #         data.append(new_datum)

    # for d in data:
    #     assert 'task' in d

    return data



class PretrainDataset(Dataset):
    def __init__(self, split='vg', rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args


        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # Answer Table from LXMERT (Could be removed)
        self.answer_table = AnswerTable()
        if self.verbose:
            print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        losses = args.losses.split(',')

        data = []
        for img_source in self.sources:
            data_info_path = dataset_dir.joinpath(f'lxmert/{img_source}.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    self.img_ids_to_source[datum['img_id']] = img_source
                    # datum['img_source'] = img_source
                    datum['args'] = args
                    datum['is_train'] = is_train
                    datum['caption_only'] = args.caption_only

                    datum['lm'] = 'lm' in losses
                    datum['qa'] = 'qa' in losses
                    datum['ground_caption'] = 'ground_caption' in losses
                    datum['refer'] = 'refer' in losses
                    datum['itm'] = 'itm' in losses
                    datum['caption'] = 'caption' in losses

                    datum['backbone'] = self.args.backbone

                data.extend(_data)

        # Modify the answers
        if 'qa' in args.losses:
            for datum in data:
                labelf = datum['labelf']
                for _qa_source, labels in labelf.items():
                    for label in labels:
                        for ans in list(label.keys()):
                            new_ans = self.answer_table.convert_ans(ans)
                            if self.answer_table.used(new_ans):
                                if ans != new_ans:
                                    label[new_ans] = label.pop(ans)
                            else:
                                label.pop(ans)

        if self.verbose:
            print("# images:", len(data))

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if 'qa' in args.losses:
            self.evaluator = QAEvaluator(data)

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretrainig data examples") for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data) for datum in _data]

        if self.args.itm_cocoonly:
            caption_sources = ['mscoco']
        else:
            caption_sources = ['mscoco', 'vg']
        self.data_captions = [datum for datum in data if datum['text_source'] in caption_sources]
        self.n_data_captions = len(self.data_captions)

        if self.verbose:
            print('# itm data:', self.n_data_captions)

        self.data = data
        self.n_data = len(self.data)

        if self.verbose and is_train:
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
            'mscoco_resplit_train_train2014': coco_img_dir.joinpath(f'train2014'),
            'mscoco_resplit_train_val2014': coco_img_dir.joinpath(f'val2014'),
            'mscoco_resplit_val': coco_img_dir.joinpath(f'val2014'),
            'vgnococo': vg_img_dir,

        }

        self.n_boxes = args.n_boxes

        self.image_size = eval(self.args.image_size)

        if is_train and self.args.use_data_augmentation:
            self.transform = augmentation_transform(self.image_size)
        else:
            self.transform = _transform(self.image_size)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                # self.tokenizer = VLT5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
            else:
                # self.tokenizer = T5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(args.backbone)
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)



    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        ###### Image ######
        img_id = datum['img_id']
        source = datum['img_source']

        # f = self.source_to_h5[source]
        # if isinstance(f, Path):
        #     path = self.source_to_h5[source]
        #     f = h5py.File(path, 'r')
        #     self.source_to_h5[source] = f

        if 't5' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg']

                prefix = "span prediction:"
                sent = datum['sent']
                source_text, target_text = preprocess.corrupt_spans(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

            elif task == 'qa':
                assert text_source in ['vqa', 'gqa', 'visual7w'], (text_source, uid)

                label = datum['label']
                assert len(label) > 0

                keys, values = zip(*label.items())
                # single answer
                if len(keys) == 1:
                    ans = keys[0]
                # multiple answers -> sample one answer
                else:
                    value_sum = sum(values)
                    prob = [value / value_sum for value in values]
                    choice = np.random.multinomial(1, prob).argmax()
                    ans = keys[choice]

                sent = datum['sent']

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {sent}"
                else:
                    source_text = f"{text_source}: {sent}"
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                target_text = ans

            elif task == 'itm':

                assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']

                prefix = "image text match:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

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

            if task == 'refer':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

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
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            path = self.source_to_h5[source].joinpath(f"{img_id}.jpg")
        
            image = Image.open(path)

            out_dict["image"] = self.transform(image)

            # feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            # try:
            #     f[f'{img_id}/features'].read_direct(feats)
            # except KeyError:
            #     print(uid)
            #     print(source)
            #     print(img_id)
            #     exit()

            # feats = torch.from_numpy(feats)
            # out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
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

            return out_dict

        elif 'bart' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg'], (datum, text_source)

                # LM only
                if self.args.losses == 'lm':
                    prefix = None
                else:
                    prefix = "denoise text:"
                sent = datum['sent']
                source_text, target_text = preprocess.corrupt_bart(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

            elif task == 'qa':
                assert text_source in ['vqa', 'gqa',
                    'visual7w'], (text_source, uid)

                label = datum['label']
                assert len(label) > 0
                # for ans in list(label.keys()):
                #     label[self.answer_table.ans2id(ans)] = label.pop(ans)
                keys, values = zip(*label.items())
                # single answer
                if len(keys) == 1:
                    ans = keys[0]
                # multiple answers -> sample one answer
                else:
                    value_sum = sum(values)
                    prob = [value / value_sum for value in values]
                    choice = np.random.multinomial(1, prob).argmax()
                    ans = keys[choice]

                sent = datum['sent']

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {sent}"
                else:
                    source_text = f"{text_source}: {sent}"
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                target_text = ans

            elif task == 'itm':

                assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(
                            0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']

                prefix = "image text match:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

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

            if task == 'refer':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

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
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            path = self.source_to_h5[source].joinpath(f"{img_id}.jpg")
        
            image = Image.open(path)

            out_dict["image"] = self.transform(image)

            # feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            # try:
            #     f[f'{img_id}/features'].read_direct(feats)
            # except KeyError:
            #     print(uid)
            #     print(source)
            #     print(img_id)
            #     exit()

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

            return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        # V_L = len(batch[0]['boxes'])

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        # feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        # vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []
        images = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            # boxes[i] += entry['boxes']
            # vis_feats[i] += entry['vis_feats']

            images.append(entry["image"])

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        # batch_entry['boxes'] = boxes
        # batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        batch_entry['images'] = torch.stack(images)

        return batch_entry


def get_loader(args, split='vgnococo', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):


    verbose = (gpu == 0)
    dataset = PretrainDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
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


class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError
