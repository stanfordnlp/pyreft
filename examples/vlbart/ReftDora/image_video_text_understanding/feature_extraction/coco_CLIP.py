"""
Preprocess a raw json dataset into features files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: two folders of features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch import nn

from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_size = 224

preprocess = Compose([
    Resize((image_size, image_size), interpolation=Image.BICUBIC),
    CenterCrop((image_size, image_size)),
    ToTensor()
])


from clip.clip import load
from timm.models.vision_transformer import resize_pos_embed
import timm


class Downsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        """
        output size: list of 1-D size, such as (6, 6), (16, 16)
        """
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, inputs_tuple):
        # inputs (B, L, dim)

        inputs = inputs_tuple
        B, L, dim = inputs.shape

        inputs = inputs.permute(0, 2, 1) # (B, dim, L)

        # restriction: L**0.5 must to be integer
        sqrt_L = int(L ** 0.5)

        inputs = inputs.reshape(B, dim, sqrt_L, sqrt_L)

        outputs = self.pool(inputs) # (B, dim, self.output_size[0], self.output_size[1])
        outputs = outputs.reshape(B, dim, -1)

        outputs = outputs.permute(0, 2, 1) # (B, self.output_size[0]**2, dim)

        return outputs


def main(params):
    if params["model_type"] != 'vit_base_patch32_224_in21k':
        model, transform = load(params["model_type"], jit=False) 
    else:
        model = timm.create_model(params["model_type"], pretrained=True)
        model = model.cuda()

    if params["model_type"] != 'vit_base_patch32_224_in21k':
        save_model_type = params["model_type"].split("-")[0]
        mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").reshape(3, 1, 1)
        std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").reshape(3, 1, 1)

        if "RN" in params["model_type"]:
            num_patches = (image_size // 32) ** 2 # 196 #600 * 1000 // 32 // 32
            pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, model.visual.attnpool.positional_embedding.shape[-1],  device='cuda'),)
            pos_embed.weight = resize_pos_embed(model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
            model.visual.attnpool.positional_embedding = pos_embed

    else:
        save_model_type = 'vit_base'
        mean = torch.Tensor([0.5, 0.5, 0.5]).to("cuda").reshape(3, 1, 1)
        std = torch.Tensor([0.5, 0.5, 0.5]).to("cuda").reshape(3, 1, 1)

        num_patches = 196 #600 * 1000 // 32 // 32
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768,  device='cuda'),)
        pos_embed.weight = resize_pos_embed(model.pos_embed, pos_embed)
        model.pos_embed = pos_embed

    if params["model_type"] == "ViT-B/32":
        num_patches = 196 #600 * 1000 // 32 // 32
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768,  device='cuda'),)
        pos_embed.weight = resize_pos_embed(model.visual.positional_embedding.unsqueeze(0), pos_embed.unsqueeze(0))
        model.visual.positional_embedding = pos_embed


    imgs = os.listdir(params["images_root"])

    # print(imgs)
    N = len(imgs)

    seed(123) # make reproducible

    dir_fc = os.path.join(params['output_dir'], 'data_clip_'+save_model_type+'_fc')
    dir_att = os.path.join(params['output_dir'], 'data_clip_'+save_model_type+'_att')
    if not os.path.isdir(dir_fc):
        os.makedirs(dir_fc)
    if not os.path.isdir(dir_att):
        os.makedirs(dir_att)

    downsample = Downsample((6, 6))

    for i, img in enumerate(imgs):
        # load the image
        img_id = img.split(".")[0]
        output_fname = os.path.join(dir_att, f"{img_id}.h5")

        if os.path.exists(output_fname):
            continue

        with torch.no_grad():
            try:
                image = preprocess(Image.open(os.path.join(params['images_root'], img)).convert("RGB"))
            except (UnidentifiedImageError, OSError):
                print(img)
                continue
            image = torch.tensor(np.stack([image])).cuda()
            image -= mean
            image /= std
            if "RN" in params["model_type"]:
                tmp_att, tmp_fc = model.encode_image(image)
                tmp_att = tmp_att[0].permute(1, 2, 0)
                tmp_fc = tmp_fc[0]
            elif params["model_type"] == 'vit_base_patch32_224_in21k':
                x = model(image)
                tmp_fc = x[0, 0, :]
                tmp_att = x[0, 1:, :].reshape( 14, 14, 768 )
            else:
                x = model.visual.conv1(image.half())  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + model.visual.positional_embedding.to(x.dtype)[:x.shape[1], :]
                x = model.visual.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND

                for layer_idx, layer in enumerate(model.visual.transformer.resblocks):
                    x = layer(x)  

                x = x.permute(1, 0, 2)
                tmp_fc = x[0, 0, :]
                tmp_att = x[0, 1:, :].reshape( 14, 14, 768 )

        output_fname = os.path.join(dir_att, f"{img_id}.h5")
        with h5py.File(output_fname, 'w') as f:
            grp = f.create_group(img_id)

            output = tmp_att.reshape(params["att_size"] ** 2, -1)
            grp['features'] = output.cpu()

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', dir_fc, dir_att)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    # parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='/playpen3/home/ylsung/data/train2014', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')
    parser.add_argument('--model_type', default='RN50', type=str, help='RN50, RN101, RN50x4, ViT-B/32, vit_base_patch32_224_in21k')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
