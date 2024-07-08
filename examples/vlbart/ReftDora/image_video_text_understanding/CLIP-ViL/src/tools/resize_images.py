folder = "/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014/"
root = "/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val_"
out_folder = "/local/harold/ubert/clip_vlp/lxmert/data/mscoco/val2014_small.lmdb"
import torch
import os
import json
from PIL import Image

from tqdm import tqdm
from vlm.vok_utilis import TxtLmdb
import numpy as np
def vokenize_and_cache_dataset(output_path, dataset, vokenizer, tokenizer):
    ## Let's use lmdb

    
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
    for index, batch in enumerate(tqdm(data_loader)):
        top_scores, top_idxs, input_tokens, top_paths = vokenize_batch(batch, tokenizer, vokenizer)
        
        top_paths = top_paths[0]
        top_idxs = top_idxs[0].cpu().numpy().tolist()
        input_tokens = input_tokens[0]
        top_scores = top_scores[0].cpu().numpy().tolist()
        lmdb_dataset[str(index)] = {
            "top_paths": top_paths,
            "top_idxs": top_idxs,
            "input_tokens": input_tokens,
            "top_scores": top_scores
        }

    del lmdb_dataset

from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, ColorJitter
from vision_helpers import Resize, PadToGivenSize

min_size = 384
max_size = 640
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
transform = Compose(
    [
        Resize(min_size, max_size)
        #lambda image: image.convert("RGB"),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)
import copy
import os
import random

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

#class ToyDataset(Dataset):
#    def __init__(self, )
all_image_files = []
for _, dirs, files in os.walk(folder, topdown=False):
        for image_file in tqdm(files):
            if image_file.endswith("jpg"):
                all_image_files.append(image_file)
#with open(root+"image_ids.json", "w") as f:
#    json.dump(all_image_files, f)

#with open("/local/harold/vqa/google_concetual/image_ids.json") as f:
#    all_image_files = json.load(f)

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

from tqdm import tqdm
lmdb_dataset = TxtLmdb(out_folder, readonly=False)
valid_images = {}
skipped = 0
for image in tqdm(all_image_files):
    try:
        feats = transform(Image.open(os.path.join(folder, image)))  # Raw image as a tensor: 3 x 224 x 224
        lmdb_dataset[image] = image_to_byte_array(feats)
        valid_images[image] = feats.size
    except KeyboardInterrupt:
        del lmdb_dataset
        assert (0)
    except:
        skipped += 1
        if skipped % 100 == 0:
            print("{} skipped.".format(skipped))
        pass

with open(root + "image_size.json", "w") as f:
    json.dump(valid_images, f)

'''
all_image_files = []
for root, dirs, files in os.walk(folder, topdown=False):
        for image_file in files:
            if image_file.endswith("jpg"):
                all_image_files.append(image_file)
with open("/local/harold/vqa/google_concetual/image_ids.json", "w") as f:
    json.dump(all_image_files, f)'''