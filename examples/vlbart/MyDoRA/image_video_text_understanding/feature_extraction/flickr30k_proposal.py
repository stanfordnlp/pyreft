# coding=utf-8

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse


class Flickr30KDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--flickrroot', type=str,
                        default='/ssd-playpen/home/jmincho/workspace/datasets/flickr30k/')
    parser.add_argument('--split', type=str, default=None, choices=['trainval', 'test2017', 'test2018'])

    args = parser.parse_args()

    SPLIT2DIR = {
        'trainval': 'flickr30k_images',
        'test2017': 'test_2017_flickr_images',
        'test2018': 'test_2018_flickr_images',
    }

    flickr_dir = Path(args.flickrroot).resolve()
    flickr_img_dir = flickr_dir.joinpath('flickr30k_images/').joinpath(SPLIT2DIR[args.split])

    dataset_name = 'Flickr30K'

    out_dir = flickr_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', flickr_img_dir)
    print('# Images:', len(list(flickr_img_dir.iterdir())))

    dataset = Flickr30KDataset(flickr_img_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
