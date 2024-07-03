# coding=utf-8
# Copyleft 2019 Project LXRT

import argparse
import json
import h5py
from tqdm import tqdm
import os

# import some common libraries
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
# fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data import MetadataCatalog

from torchvision.ops import nms
from detectron2.structures import Boxes, Instances


D2_ROOT = os.path.dirname(os.path.dirname(
    detectron2.__file__))  # Root of detectron2
# DATA_ROOT = os.getenv('COCO_IMG_ROOT', '/ssd-playpen/data/mscoco/images/')
# MIN_BOXES = 36
# MAX_BOXES = 36
# NUM_OBJECTS = 36

# min_n_regions = 20
# min_caption_len = 2
DIM = 2048

# Load VG Classes
data_path = 'demo/data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(D2_ROOT, data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(D2_ROOT, data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())
MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs


def doit(raw_image, raw_boxes, predictor):
        # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])

        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled) and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores=max_attr_prob,
            attr_classes=max_attr_label
        )

        return instances, roi_features

def build_model():
    cfg = get_cfg() # Renew the cfg file
    cfg.merge_from_file(os.path.join(
        D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # cfg.INPUT.MIN_SIZE_TEST = 600
    # cfg.INPUT.MAX_SIZE_TEST = 1000
    # cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"

    detector = DefaultPredictor(cfg)
    return detector


def extract(output_fname, dataloader, desc):
    detector = build_model()

    with h5py.File(output_fname, 'w') as f:
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader),
                                 desc=desc,
                                 ncols=150,
                                 total=len(dataloader)):

                img_ids = batch['img_ids']
                # feat_list, info_list = feature_extractor.get_detectron_features(batch)

                boxes = batch['boxes']
                imgs = batch['imgs']

                assert len(imgs) == 1

                img = imgs[0]
                img_id = img_ids[0]
                boxes = boxes[0]

                instances, features = doit(img, boxes, detector)

                instances = instances.to('cpu')
                features = features.to('cpu')

                num_objects = len(instances)

                # assert num_objects == NUM_OBJECTS
                # assert features.shape == (NUM_OBJECTS, dim)

                grp = f.create_group(img_id)
                grp['features'] = features.numpy()  # [num_features, 2048]
                grp['obj_id'] = instances.pred_classes.numpy()
                grp['obj_conf'] = instances.scores.numpy()
                grp['attr_id'] = instances.attr_classes.numpy()
                grp['attr_conf'] = instances.attr_scores.numpy()
                grp['boxes'] = boxes
                grp['img_w'] = img.shape[1]
                grp['img_h'] = img.shape[0]
                grp['num_objects'] = num_objects

                if 'captions' in batch:
                    captions = batch['captions']
                    grp['captions'] = np.array(captions, dtype=h5py.string_dtype(encoding='utf-8'))
