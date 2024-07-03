# coding=utf-8
# Copyleft 2019 project LXRT.

import json

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
from src.pretrain.lxmert_data import ImageReader


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
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
 
        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)
lmdb_paths = {
    'mscoco_train': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/train2014_small.lmdb',
    'mscoco_minival': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014_small.lmdb',
    'mscoco_nominival': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014_small.lmdb',
    'vgnococo': '/local/harold/ubert/clip_vlp/lxmert/data/vg_raw_images/vg_small.lmdb/',
}
Split2ImgFeatPath = {
    'train': '/local/harold/ubert/lxmert/data/mscoco_imgfeat/train2014_obj36.h5',
    'valid': '/local/harold/ubert/lxmert/data/mscoco_imgfeat/val2014_obj36.h5',
    'vgnococo': '/local/harold/ubert/lxmert/data/vg_gqa_imgfeat/vg_gqa_obj36.h5',
}
class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()

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
        
            with open("data/gqa/width_heigths.json") as f:
                self.w_h_records = json.load(f)

    def __getitem__(self, img_id):
        
        # img_id : COCO_val2014_000000393267
        # actual image file name: data/mscoco/val2014/COCO_val2014_000000393267.jpg

        image_file_name = "data/gqa/images/{}.jpg".format(img_id)
        image = Image.open(image_file_name)
        feats = self.transform(image)  # Raw image as a tensor: 3 x 224 x 224
        return feats
"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

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
        if not os.path.exists("data/gqa/width_heigths.json"):
            w_h_records = {}
            root_name = "data/gqa/images/"
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

            with open("data/gqa/width_heigths.json", "w") as f:
                json.dump(w_h_records, f)
            assert(0)
        else:
            with open("data/gqa/width_heigths.json") as f:
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
            self.image_reader = ImageReader(args, new_split, lmdb_paths)
            self.data = self.raw_dataset.data
        elif self.use_h5_file:
            from h5_data import ImageFeatureDataset

            self.image_feature_dataset = ImageFeatureDataset.create(["train", "valid", "vgnococo"], Split2ImgFeatPath, on_demand=False, on_memory = False)
            self.ids_to_index = self.image_feature_dataset.ids_to_index

            # Screen data
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.ids_to_index:
                    used_data.append(datum)
            self.data = used_data
        counter = 0
        new_data = []
        for data in self.data:
            label = data['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            flag = True
            for ans, score in label.items():
                if ans not in self.raw_dataset.ans2label:
                    counter += 1
                    flag = False
            if flag:
                new_data.append(data)
        
        self.data = new_data

        print(" {} answer missing".format(counter))

        print("Use %d data in torch dataset" % (len(self.data)))
        
        print()

    def __len__(self):
        return len(self.data)
    
    def get_height_and_width(self, index):
        datum = self.data[index]
        img_id = datum['img_id']
        image_file_name = "data/gqa/images/{}.jpg".format(img_id)
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
            boxes = img_info['boxes'].copy()
            feats = img_info['features'].copy()
            assert len(boxes) == len(feats) == obj_num

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        boxes = torch.from_numpy(boxes)
        feats = torch.from_numpy(feats)
        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


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


