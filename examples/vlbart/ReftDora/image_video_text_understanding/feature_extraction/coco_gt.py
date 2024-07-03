# coding=utf-8

# from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from detectron2_given_box_maxnms import extract, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse

from pycocotools.coco import COCO
import json
import numpy as np


class COCODataset(Dataset):
    def __init__(self, image_dir, box_ann_path, split='val2014'):
        self.image_dir = image_dir

        box_ann_path = str(box_ann_path)

        self.coco = COCO(box_ann_path)

        self.split = split
        with open(box_ann_path) as f:
            box_ann = json.load(f)
        id2name = {}
        for cat2name in box_ann['categories']:
            id2name[cat2name['id']] = cat2name['name']
        self.id2name = id2name

        img_ids = []
        boxes = []
        captions = []
        for img_id, anns in self.coco.imgToAnns.items():
            img_ids.append(img_id)

            boxes.append([ann['bbox'] for ann in anns])
            captions.append([self.id2name[ann['category_id']] for ann in anns])

        assert len(img_ids) == len(boxes)
        assert len(img_ids) == len(captions)

        self.img_ids = img_ids
        self.boxes = boxes
        self.captions = captions

    def __len__(self):
        return len(self.coco.imgToAnns)

    def __getitem__(self, idx):

        image_id = self.img_ids[idx]

        image_name = f'COCO_{self.split}_{str(image_id).zfill(12)}'

        image_path = self.image_dir.joinpath(f'{image_name}.jpg')

        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        H, W, _ = img.shape

        boxes = []
        for box in self.boxes[idx]:
            x, y, width, height = box
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            boxes.append([x1, y1, x2, y2])

        assert len(boxes) > 0

        boxes = np.array(boxes)

        captions = self.captions[idx]

        return {
            'img_id': image_name,
            'img': img,
            'boxes': boxes,
            'captions': captions
        }


def collate_fn(batch):
    img_ids = []
    imgs = []
    boxes = []
    captions = []

    for i, entry in enumerate(batch):
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])
        boxes.append(entry['boxes'])
        captions.append(entry['captions'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    batch_out['boxes'] = boxes

    batch_out['captions'] = captions

    return batch_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--cocoroot', type=str, default='/ssd-playpen/home/jmincho/workspace/datasets/COCO/')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])

    args = parser.parse_args()

    SPLIT2DIR = {
        'train': 'train2014',
        'valid': 'val2014',
        'test': 'test2015',
    }

    coco_dir = Path(args.cocoroot).resolve()
    coco_img_dir = coco_dir.joinpath('images')
    coco_img_split_dir = coco_img_dir.joinpath(SPLIT2DIR[args.split])
    box_ann_path = coco_dir.joinpath('annotations').joinpath(f'instances_{SPLIT2DIR[args.split]}.json')

    dataset_name = 'COCO'

    out_dir = coco_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', coco_img_split_dir)
    print('# Images:', len(list(coco_img_split_dir.iterdir())))

    dataset = COCODataset(coco_img_split_dir, box_ann_path, SPLIT2DIR[args.split])
    print('# Annotated Images:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{SPLIT2DIR[args.split]}_GT.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{SPLIT2DIR[args.split]}_{DIM}'

    extract(output_fname, dataloader, desc)
