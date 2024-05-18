# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
from tqdm import tqdm
from .vision_helpers import to_image_list
import json

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}

Split2ImgFeatPath = {
    'train': 'data/mscoco_imgfeat/train2014_obj36.h5',
    'valid': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'minival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'nominival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    "test": 'data/mscoco_imgfeat/test2015_obj36.h5',
}

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

from src.pretrain.lxmert_data import ImageReader
"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()

        ### Control options
        self.input_raw_images = args.input_raw_images
        self.vqa_style_transform = args.vqa_style_transform
        self.use_h5_file = args.use_h5_file
        self.image_size_min = args.image_size_min
        self.image_size_max = args.image_size_max
        self.dynamic_padding = args.dynamic_padding
        self.add_zero_padding= args.add_zero_padding
        ###

        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

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
        if self.input_raw_images:
            from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            #model, preprocess = clip.load("ViT-B/32", device=device)        
            new_split = []        
            for split in dataset.splits:
                if "train" in split:
                    new_split.append("mscoco_train")
                elif "nominival" in split:
                    new_split.append("mscoco_nominival")
                else:
                    new_split.append("mscoco_minival")
            print(new_split)
            self.image_reader = ImageReader(args, new_split, lmdb_paths = None)
            self.data = self.raw_dataset.data
        
        elif self.use_h5_file:
            from h5_data import ImageFeatureDataset

            self.image_feature_dataset = ImageFeatureDataset.create(dataset.splits, Split2ImgFeatPath, on_demand=False, on_memory = False)
            self.ids_to_index = self.image_feature_dataset.ids_to_index

            # Screen data
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.ids_to_index:
                    used_data.append(datum)
            self.data = used_data
        else:
            # Loading detection features to img_data
            img_data = []
            for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))

            # Convert img list to dict
            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum

            # Only kept the data with loaded image features
            self.data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.imgid2img:
                    self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        '''with open("snap/vqa/test/test_predict.json") as f:
            predicted = json.load(f)

        if self.raw_dataset.name == "test":
            print(len(self))
            print(len(predicted))
            all_ids = set([str(i["question_id"]) for i in predicted])
            for i in self.data:
                ques_id = i['question_id']
                if str(ques_id) not in all_ids:
                    print(ques_id)
            assert(0),["393139010","393139011","393139012","524265009","524265010","524265011","524265012","393213001","393213002","262142002","218453002"]'''


    def __len__(self):
        return len(self.data)

    def get_height_and_width(self, index):
        datum = self.data[index]
        img_id = datum['img_id']
        if "val2014" in img_id:
            image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
        elif "train2014" in img_id:
            image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
        elif "test2015" in img_id:
            image_file_name = "data/mscoco/test2015/{}.jpg".format(img_id)
        w, h = self.w_h_records[image_file_name]
        return h, w

    def __getitem__(self, item: int):
        if self.input_raw_images:
            return self.getitem_clip(item)
        else:
            return self.getitem_butd(item)
    
    def getitem_clip(self, item):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        # img_id : COCO_val2014_000000393267
        # actual image file name: data/mscoco/val2014/COCO_val2014_000000393267.jpg
        if "val2014" in img_id:
            image_file_name = "data/mscoco/val2014/{}.jpg".format(img_id)
        elif "train2014" in img_id:
            image_file_name = "data/mscoco/train2014/{}.jpg".format(img_id)
        
        #feats = self.transform(Image.open(image_file_name))  # Raw image as a tensor: 3 x 224 x 224
        feats = self.image_reader[img_id]
        #if feats.size()[1] > feats.size()[2]:
        #    feats = to_image_list([feats], max_size=(3, 1000, 600))[0]
        #else:
        #    feats = to_image_list([feats], max_size=(3, 600, 1000))[0]

        boxes = torch.Tensor([0.0]) # Just being lazy

        # Provide label (target)
        
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques
        
    def collate_fn(self, batch):
        if len(batch[0]) == 5:
            ques_id, feats, boxes, ques, target = zip(*batch)
        else:
            ques_id, feats, boxes, ques = zip(*batch)
        if self.input_raw_images and self.vqa_style_transform:
            if self.dynamic_padding:
                feats = to_image_list(feats)
            else:
                if feats[0].size(1) <= feats[0].size(2):
                    feats = to_image_list(feats, max_size=(3, self.image_size_min, self.image_size_max))
                else:
                    feats = to_image_list(feats, max_size=(3, self.image_size_max, self.image_size_min))
        else:
            feats = torch.stack(feats, dim=0)
        boxes = torch.stack(boxes, dim=0)
        #ques_id = torch.LongTensor(ques_id)
        if len(batch[0]) == 5:
            target = torch.stack(target, dim=0)
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

    def getitem_butd(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        if self.use_h5_file:
            image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset[img_id]
        else:
            # Get image info
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            assert obj_num == len(boxes) == len(feats)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        boxes = torch.from_numpy(boxes)
        feats = torch.from_numpy(feats)
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
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
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


