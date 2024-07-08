# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import h5py
import argparse

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in tqdm(enumerate(reader), ncols=150):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(
                    base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." %
          (len(data), fname, elapsed_time))
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str,
                        default='val2014_obj36.tsv')
    parser.add_argument('--h5_path', type=str,
                        default='val2014_obj36.h5')

    args = parser.parse_args()
    dim = 2048

    print('Load ', args.tsv_path)
    data = load_obj_tsv(args.tsv_path)
    print('# data:', len(data))

    output_fname = args.h5_path
    print('features will be saved at', output_fname)

    with h5py.File(output_fname, 'w') as f:
        for i, datum in tqdm(enumerate(data),
                            ncols=150,):

            img_id = datum['img_id']

            num_boxes = datum['num_boxes']

            grp = f.create_group(img_id)
            grp['features'] = datum['features'].reshape(num_boxes, 2048)
            grp['obj_id'] = datum['objects_id']
            grp['obj_conf'] = datum['objects_conf']
            grp['attr_id'] = datum['attrs_id']
            grp['attr_conf'] = datum['attrs_conf']
            grp['boxes'] = datum['boxes']
            grp['img_w'] = datum['img_w']
            grp['img_h'] = datum['img_h']
