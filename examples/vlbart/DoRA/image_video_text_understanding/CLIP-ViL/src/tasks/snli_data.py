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
from tqdm import tqdm
import lmdb
from lz4.frame import compress, decompress

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

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

text_db_paths = {
    "valid": "/playpen/home/ylsung/datasets/images/snli-ve/txt_db/ve_dev.db",
    "train": "/playpen/home/ylsung/datasets/images/snli-ve/txt_db/ve_train.db",
    "test": "/playpen/home/ylsung/datasets/images/snli-ve/txt_db/ve_test.db",
}

from src.pretrain.lxmert_data import ImageReader
lmdb_paths = {
    'mscoco_train': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/train2014_small.lmdb',
    'mscoco_minival': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014_small.lmdb',
    'mscoco_nominival': '/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014_small.lmdb',
    'vgnococo': '/local/harold/ubert/clip_vlp/lxmert/data/vg_raw_images/vg_small.lmdb/',
}

Split2ImgFeatPath_h5 = {
    'flickr': '/local/harold/ubert/lxmert/data/flickr30k/fixed36_no_features_split_0_of_1_splits.h5',
}
"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
from src.tools.lmdb_dataset import TxtLmdb
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter

class ImageReader():
    def __init__(self, args):
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
    def __getitem__(self, img_id):
        image_file_name = "/playpen/home/ylsung/datasets/images/snli-ve/flickr30k_images/{}.jpg".format(img_id)
        image = Image.open(image_file_name)
        feats = self.transform(image)  # Raw image as a tensor: 3 x 224 x 224
        return feats

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret
class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

        #if custom_args.get('convert_to_custom_tokenizer', False):
        from lxrt.tokenization import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True)
        self.cls_ = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.mask = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        with open("{}/id2len.json".format(db_dir)) as f:
            self.id2len = json.load(f)
        self.ids = list(self.id2len.keys())
        self.ids.sort()
    
    def convert_to_custom_tokenizer(self, example):
        '''
        {'tokens': ['a', 'group', 'of', 'people', 'stand', 'in', 'the', 'back', 'of', 'a', 'truck', 'filled', 'with', 'cotton'], 'raw': 'A group of people stand in the back of a truck filled with cotton.', 'imgid': 67, 'sentid': 335, 'image_id': 1018148011, 'toked_caption': ['A', 'group', 'of', 'people', 'stand', 'in', 'the', 'back', 'of', 'a', 'truck', 'filled', 'with', 'cotton', '@@.'], 'input_ids': [138, 1372, 1104, 1234, 2484, 1107, 1103, 1171, 1104, 170, 4202, 2709, 1114, 7825, 119], 'img_fname': 'flickr30k_001018148011.npz'}
        '''
        
        '''raw_tokens = example["sentence2"].lower()
        toked_caption = self.tokenizer.tokenize(raw_tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(toked_caption)
        example["toked_caption"] = toked_caption
        example["input_ids"] = input_ids'''
        assert(len(example["target"]["labels"]) == 1)
        example["label"] = example["target"]["labels"][0]
        return example
    
    def __getitem__(self, index):
        id_ = self.ids[index]
        txt_dump = self.db[id_]
        txt_dump = self.convert_to_custom_tokenizer(txt_dump)
        return txt_dump
    
    def __len__(self):
        return len(self.ids)

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return torch.tensor(input_ids)

    @property
    def txt2img(self):
        txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
        return txt2img

    @property
    def img2txts(self):
        img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
        return img2txts

class SNLIDataset(Dataset):
    def __init__(self, split):
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

        self.raw_dataset = TxtTokLmdb(text_db_paths[split])
        self.data = self.raw_dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # calculate image size
        if not os.path.exists("/playpen/home/ylsung/datasets/images/snli-ve/width_heigths.json"):
            w_h_records = {}
            root_name = "/playpen/home/ylsung/datasets/images/snli-ve/flickr30k_images/"
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

            with open("/playpen/home/ylsung/datasets/images/snli-ve/width_heigths.json", "w") as f:
                json.dump(w_h_records, f)
            assert(0)
        else:
            with open("/playpen/home/ylsung/datasets/images/snli-ve/width_heigths.json") as f:
                self.w_h_records = json.load(f)
        if self.input_raw_images:
            from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter
            self.image_reader = ImageReader(args)
        else:
            from h5_data import ImageFeatureDataset
            self.image_feature_dataset = ImageFeatureDataset.create(["flickr"], Split2ImgFeatPath_h5, load_custom_h5_version2=True, on_demand=False, on_memory = False)
        print()

    def __len__(self):
        return len(self.raw_dataset)

    def get_height_and_width(self, index):
        datum = self.data[index]
        img_id = datum['image_id']
        
        image_file_name = "/playpen/home/ylsung/datasets/images/snli-ve/flickr30k_images/{}.jpg".format(img_id)
        w, h = self.w_h_records[image_file_name]
        return h, w

    def __getitem__(self, item: int):
        if self.input_raw_images:
            return self.getitem_clip(item)
        else:
            return self.getitem_butd(item)
    
    def getitem_clip(self, item):
        datum = self.data[item]

        img_id = datum['image_id']
        ques_id = 0
        ques = datum['sentence2'].lower()

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
            return ques_id, feats, boxes, ques, torch.LongTensor([label])
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

        img_id = datum['image_id']
        ques_id = 0
        ques = datum['sentence2'].lower()

        if self.use_h5_file:
            image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset["/local/harold/ubert/lxmert/data/flickr30k/flickr30k_images/{}.jpg".format(img_id)]
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
            
            return ques_id, feats, boxes, ques, torch.LongTensor([label])
        else:
            return ques_id, feats, boxes, ques
