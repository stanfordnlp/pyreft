# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset

from param import args
from pretrain.qa_answer_table import AnswerTable
from utils import load_obj_tsv
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from lxrt.tokenization import BertTokenizer
from src.tasks.vision_helpers import to_image_list
from src.tools import sharearray
from tqdm import tqdm
import numpy as np
TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000

Split2ImgFeatPath = {
    'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("data/lxmert/%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),

from PIL import Image
import io

def image_to_byte_array(image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format="JPEG")
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def byte_array_to_image(byte):
    imgByteArr = io.BytesIO(byte)
    imgByteArr.seek(0)
    return Image.open(imgByteArr)

from src.tools.lmdb_dataset import TxtLmdb
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter

class ImageReader():
    def __init__(self, args, sources, lmdb_paths):
        self.use_lmdb = args.use_lmdb

        if args.use_lmdb:
            self.lmdb_dataset = {}
            for source in sources:
                self.lmdb_dataset[source] = TxtLmdb(lmdb_paths[source], readonly=True)
            self.lmdb_dataset["mscoco_nominival"] = TxtLmdb(lmdb_paths["mscoco_nominival"], readonly=True)
        
        if args.vqa_style_transform:
            from src.tasks.vision_helpers import Resize, PadToGivenSize

            min_size = args.image_size_min
            max_size = args.image_size_max
            flip_horizontal_prob = 0.0
            flip_vertical_prob = 0.0
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            color_jitter = ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            self.transform = Compose(
                [
                    color_jitter,
                    Resize(min_size, max_size),
                    lambda image: image.convert("RGB"),
                    PadToGivenSize(min_size, max_size) if args.add_zero_padding else lambda image:image,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        
            # calculate image size
        if not os.path.exists("data/mscoco/width_heigths.json"):
            w_h_records = {}
            root_name = "data/vg_raw_images/VG_100K/"
            failed_counter = 0
            for root, dirs, files in os.walk(root_name, topdown=False):
                for file in tqdm(files):
                    try:
                        image = np.asarray(Image.open(os.path.join(root_name, file)))
                        w = image.shape[0]
                        h = image.shape[1]
                        w_h_records[os.path.join(root_name, file)] = (w, h)
                    except:
                        failed_counter += 1
            print("Skipped {} files".format(failed_counter))

            root_name = "data/mscoco/val2014/"
            for root, dirs, files in os.walk(root_name, topdown=False):
                for file in tqdm(files):
                    image = np.asarray(Image.open(os.path.join(root_name, file)))
                    w = image.shape[0]
                    h = image.shape[1]
                    w_h_records[os.path.join(root_name, file)] = (w, h)
            root_name = "data/mscoco/train2014/"
            for root, dirs, files in os.walk(root_name, topdown=False):
                for file in tqdm(files):
                    image = np.asarray(Image.open(os.path.join(root_name, file)))
                    w = image.shape[0]
                    h = image.shape[1]
                    w_h_records[os.path.join(root_name, file)] = (w, h)
            
            root_name = "data/mscoco/test2015/"
            for root, dirs, files in os.walk(root_name, topdown=False):
                for file in tqdm(files):
                    image = np.asarray(Image.open(os.path.join(root_name, file)))
                    w = image.shape[0]
                    h = image.shape[1]
                    w_h_records[os.path.join(root_name, file)] = (w, h)
            
            with open("data/mscoco/width_heigths.json", "w") as f:
                json.dump(w_h_records, f)
            assert(0)
        else:
            with open("data/mscoco/width_heigths.json") as f:
                self.w_h_records = json.load(f)

    def __getitem__(self, img_id):
        if self.use_lmdb:
            if "val2014" in img_id:
                source = 'mscoco_nominival'
            elif "train2014" in img_id:
                source = "mscoco_train"
            else:
                source = "vgnococo"
            
            image = byte_array_to_image(self.lmdb_dataset[source]["{}.jpg".format(img_id)])
        else:
            # img_id : COCO_val2014_000000393267
            # actual image file name: data/mscoco/val2014/COCO_val2014_000000393267.jpg
            if "val2014" in img_id:
                image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
            elif "train2014" in img_id:
                image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
            elif "test2015" in img_id:
                image_file_name = "data/mscoco/test2015/{}.jpg".format(img_id)
            else:
                image_file_name = "data/vg_raw_images/VG_100K/{}.jpg".format(img_id)
            image = Image.open(image_file_name)
        feats = self.transform(image)  # Raw image as a tensor: 3 x 224 x 224
        return feats
"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.name = '_'.join(self.raw_dataset.sources)
        self.task_matched = args.task_matched

        ### Control options
        self.input_raw_images = args.input_raw_images
        self.vqa_style_transform = args.vqa_style_transform
        self.use_h5_file = args.use_h5_file
        self.image_size_min = args.image_size_min
        self.image_size_max = args.image_size_max
        self.dynamic_padding = args.dynamic_padding
        self.add_zero_padding = args.add_zero_padding
        self.disable_obj_mask = args.obj_mask_rate == 0
        self.task_obj_predict = args.task_obj_predict
        self.limit_source = args.limit_source.split(",")
        self.compress_data = args.compress_data
        ###
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        if self.input_raw_images:
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #model, preprocess = clip.load("ViT-B/32", device=device)                
            self.image_reader = ImageReader(args, self.raw_dataset.sources, lmdb_paths = None)
            self.w_h_records = self.image_reader.w_h_records
        
            #used_data = self.raw_dataset.data
            used_data = []
            for datum in self.raw_dataset.data:
                img_id = datum['img_id']
                if "val2014" in img_id:
                    image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
                elif "train2014" in img_id:
                    image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
                elif "test2015" in img_id:
                    image_file_name = "data/mscoco/test2015/{}.jpg".format(img_id)
                else:
                    image_file_name = "data/vg_raw_images/VG_100K/{}.jpg".format(img_id)
                if image_file_name in self.w_h_records:
                    used_data.append(datum)

            # Filter out 

        else:
            # Load the dataset
            img_data = []
            for source in self.raw_dataset.sources:
                img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum

            # Filter out the dataset
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.imgid2img:
                    used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in self.limit_source:
                    continue
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)
        #self.data = self.data[:100]

        if self.compress_data:
            self.compress()
        
        print("Use %d data in torch dataset" % (len(self.data)))

    def compress(self):

        self._img_ids_shared_array, self._img_ids_record_position = self.compress_list_of_strings([i["img_id"] for i in self.data], "data_imonly_img_id_{}".format(self.name))
        self.compress_memory = True

        self._sent_shared_array, self._sent_record_position = self.compress_list_of_strings([i["sent"] for i in self.data], "data_txtonly_sent_{}".format(self.name))

        self._uid_shared_array, self._uid_record_position = self.compress_list_of_strings([i["uid"][0] for i in self.data], "data_txtonly_uid_{}".format(self.name))
        self.data = [i["label"] if "label" in i else None for i in self.data]


    def compress_list_of_strings(self, list_of_string, name):
        # Possible fields in datatum: img_id
        record_position = []
        all_text = []
        current_length = 0
        for index, string in enumerate(list_of_string):
            array = [ord(c) for c in string]
            all_text.extend(array)
            current_length += len(array)
            record_position.append(current_length)

        shared_array = sharearray.cache(name, lambda: np.array(all_text, dtype=np.int32))
        del all_text
        return shared_array, record_position
    
    def decompress_string_index(self, index, shared_array, record_position):
        string_array = shared_array[0 if index == 0 else record_position[index - 1]:record_position[index]]
        return ''.join([chr(c) for c in string_array])
    
    def decompress_getitem__(self, index):
        if self._sent_shared_array is not None:
            sent = self.decompress_string_index(index, self._sent_shared_array, self._sent_record_position)
        else:
            sent = ""
        if self._img_ids_shared_array is not None:
            img_id = self.decompress_string_index(index, self._img_ids_shared_array, self._img_ids_record_position)
        else:
            img_id = None
        
        if self._uid_shared_array is not None:
            uid = self.decompress_string_index(index, self._uid_shared_array, self._uid_record_position)
        else:
            uid = None
        label = self.data[index]
        if label is None:
            return {"sent": sent, "img_id": img_id, "uid": (uid)}
        return {"sent": sent, "img_id": img_id, "uid": (uid), "label": label}

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat
    
    def get_height_and_width(self, index):
        #datum = self.data[index]
        if self.compress_data:
            datum = self.decompress_getitem__(index)
        else:
            datum = self.data[index]
        img_id = datum['img_id']
        if "val2014" in img_id:
            image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
        elif "train2014" in img_id:
            image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
        elif "test2015" in img_id:
            image_file_name = "data/mscoco/test2015/{}.jpg".format(img_id)
        else:
            image_file_name = "data/vg_raw_images/VG_100K/{}.jpg".format(img_id)
        w, h = self.w_h_records[image_file_name]
        return h, w

    def __getitem__(self, item: int):
        if self.input_raw_images:
            return self.getitem_clip(item)
        else:
            return self.getitem_butd(item)
    
    def get_data(self, index):
        if self.compress_data:
            datum = self.decompress_getitem__(index)
        else:
            datum = self.data[index]
        return datum

    def getitem_clip(self, item):
        
        datum = self.get_data(item)
        #datum = self.data[item]
        uid = datum['uid']

        img_id = datum['img_id']
        # img_id : COCO_val2014_000000393267
        # actual image file name: data/mscoco/val2014/COCO_val2014_000000393267.jpg
        '''if "val2014" in img_id:
            image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
        elif "train2014" in img_id:
            image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
        else:
            image_file_name = "data/vg_raw_images/VG_100K/{}.jpg".format(img_id)
        
        feats = self.transform(Image.open(image_file_name))'''  # Raw image as a tensor: 3 x 224 x 224
        feats = self.image_reader[img_id]
        #if feats.size()[1] > feats.size()[2]:
        #    feats = to_image_list([feats], max_size=(3, 1000, 600))[0]
        #else:
        #    feats = to_image_list([feats], max_size=(3, 600, 1000))[0]

        boxes = torch.Tensor([0.0])  # Just being lazy
        
        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.get_data(random.randint(0, len(self.data)-1))
                while other_datum['img_id'] == img_id:
                    other_datum = self.get_data(random.randint(0, len(self.data)-1))
                sent = other_datum['sent']
        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (None, None), (None, None),
            is_matched, label
        )
        return example

        # Provide label (target)
        '''if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques'''
    
    def getitem_butd(self, item: int):
        if self.compress_data:
            datum = self.decompress_getitem__(item)
        else:
            datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        return example

    def collate_fn(self, examples):

        train_features = [self.convert_example_to_features(example, 20, self.tokenizer)
                            for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        # Visual Inputs
        feats = [f.visual_feats[0] for f in train_features]
        if self.input_raw_images:
            if self.dynamic_padding:
                feats = to_image_list(feats)
            else:
                if feats[0].size(1) <= feats[0].size(2):
                    feats = to_image_list(feats, max_size=(3, self.image_size_min, self.image_size_max))
                else:
                    feats = to_image_list(feats, max_size=(3, self.image_size_max, self.image_size_min))
        else:
            feats = torch.stack(feats, dim=0)
        #feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features]))

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long)

        # Visual Prediction
        obj_labels = {}
        if self.task_obj_predict:
            for key in ('obj', 'attr', 'feat'):
                visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features]))
                visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features]))
                assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
                obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_label = torch.tensor([f.is_matched for f in train_features], dtype=torch.long)
        ans = torch.from_numpy(np.stack([f.ans for f in train_features]))

        return {
            "input_ids": input_ids,
            "token_type_ids": segment_ids,
            "attention_mask": input_mask,
            "masked_lm_labels": lm_labels,
            "visual_feats": feats,
            "pos": pos,
            "obj_labels": obj_labels,
            "matched_label": matched_label,
            "ans": ans,
            "uid": [i.uid[0] for i in examples]
        }
    
    def convert_example_to_features(self, example: InputExample, max_seq_length, tokenizer):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        tokens = tokenizer.tokenize(example.sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        # Ge random words
        masked_tokens, masked_label = random_word(tokens, tokenizer)

        # concatenate lm labels and account for CLS, SEP, SEP
        masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

        # Mask & Segment Word
        lm_label_ids = ([-1] + masked_label + [-1])
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length

        feat, boxes = example.visual_feats
        obj_labels, obj_confs = example.obj_labels
        attr_labels, attr_confs = example.attr_labels

        # Mask Image Features:
        if self.disable_obj_mask:
            masked_feat = feat
            feat_mask = None
        else:
            masked_feat, feat_mask = random_feat(feat)

        # QA answer label
        if example.label is None or len(example.label) == 0 or example.is_matched != 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            keys, values = zip(*example.label.items())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = np.random.multinomial(1, prob).argmax()
                ans = keys[choice]

        features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lm_label_ids=lm_label_ids,
            visual_feats=(masked_feat, boxes),
            obj_labels={
                'obj': (obj_labels, obj_confs),
                'attr': (attr_labels, attr_confs),
                'feat': (feat, feat_mask),
            },
            is_matched=example.is_matched,
            ans=ans,
        )
        return features

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < args.obj_mask_rate:
            prob /= args.obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask

class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
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

    def evaluate(self, uid2ans: dict, pprint=False):
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
        try:
            accu = score / cnt
        except:
            accu = 0
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
        raise NotImplemented
