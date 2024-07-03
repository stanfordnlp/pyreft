import h5py
from tqdm import tqdm
import json
import pathlib
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--data_dir', type=str,
                        default='.')

    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir).resolve()
    coco_dir = data_dir.joinpath('COCO')

    with open(data_dir.joinpath('lxmert/mscoco_resplit_val.json'))as f:
        val_data = json.load(f)

    print(len(val_data))

    source_f = h5py.File(coco_dir.joinpath('features/val2014_obj36.h5'), 'r')
    target_f = h5py.File(coco_dir.joinpath('features/resplit_val_obj36.h5'), 'w')

    img_id = val_data[0]['img_id']

    keys = list(source_f[img_id].keys())

    for datum in tqdm(val_data, ncols=50):
        img_id = datum['img_id']

        grp = target_f.create_group(str(img_id))
        for k in keys:
            grp[k] = source_f[f'{img_id}/{k}'][()]
