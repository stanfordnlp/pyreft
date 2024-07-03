# coding=utf-8

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import json


class VCRDataset(Dataset):
    def __init__(self, vcr_dir, vcr_images_dir, split='val'):

        self.image_dir = vcr_images_dir
        ann_path = vcr_dir.joinpath(f'{split}.jsonl')

        with open(ann_path, 'r') as f:
            _items = [json.loads(s) for s in f]
        print('Load images from', ann_path)

        image_ids = []
        image_paths = []
        items = []
        for item in _items:
            if item['img_id'] not in image_ids:
                items.append(item)
                image_ids.append(item['img_id'])
                image_paths.append(item['img_fn'])

        self.items = items
        self.n_images = len(items)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        item = self.items[idx]
        image_path = item['img_fn']
        image_id = item['img_id']

        image_path = self.image_dir.joinpath(image_path)

        assert Path(image_path).exists()

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--vcrroot', type=str, default='/ssd-playpen/home/jmincho/workspace/datasets/VCR/')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    vcr_dir = Path(args.vcrroot).resolve()
    vcr_images_dir = vcr_dir.joinpath('vcr1images')
    dataset_name = 'VCR'

    out_dir = vcr_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    # print('Load images from', coco_img_split_dir)

    dataset = VCRDataset(vcr_dir, vcr_images_dir, args.split)
    print('# Images:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
