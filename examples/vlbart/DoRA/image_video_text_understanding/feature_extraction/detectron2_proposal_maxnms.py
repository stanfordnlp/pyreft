# coding=utf-8
# Copyleft 2019 Project LXRT

import os
import numpy as np
import torch
import h5py
from torchvision.ops import nms
from tqdm import tqdm

import detectron2
from detectron2.structures import Boxes, Instances
from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2
# DATA_ROOT = os.getenv('COCO_IMG_ROOT', '/ssd-playpen/data/mscoco/images/')
MIN_BOXES = 36
MAX_BOXES = 36
NUM_OBJECTS = 36
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


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)       # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]     # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep


def doit(raw_image, predictor):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(
            raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(
            images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        # print(instances)

        return instances, roi_features


def build_model():
    cfg = get_cfg()  # Renew the cfg file
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
    # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
    # cfg.MODEL.WEIGHTS = "~/.torch/fvcore_cache/models/faster_rcnn_from_caffe_attr.pkl"
    # Path.home().joinpath('.torch/fvcore_cache/models/faster_rcnn_from_caffe_attr.pkl').exists()
    from pathlib import Path
    cfg.MODEL.WEIGHTS = str(Path.home().joinpath('.torch/fvcore_cache/models/faster_rcnn_from_caffe_attr.pkl'))
    detector = DefaultPredictor(cfg)
    return detector


def collate_fn(batch):
    img_ids = []
    imgs = []

    for i, entry in enumerate(batch):
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    return batch_out

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

                imgs = batch['imgs']

                assert len(imgs) == 1

                img = imgs[0]
                img_id = img_ids[0]

                try:
                    instances, features = doit(img, detector)

                    instances = instances.to('cpu')
                    features = features.to('cpu')

                    num_objects = len(instances)

                    assert num_objects == NUM_OBJECTS, (num_objects, img_id)
                    assert features.shape == (NUM_OBJECTS, DIM)

                    grp = f.create_group(img_id)
                    grp['features'] = features.numpy()  # [num_features, 2048]
                    grp['obj_id'] = instances.pred_classes.numpy()
                    grp['obj_conf'] = instances.scores.numpy()
                    grp['attr_id'] = instances.attr_classes.numpy()
                    grp['attr_conf'] = instances.attr_scores.numpy()
                    grp['boxes'] = instances.pred_boxes.tensor.numpy()
                    grp['img_w'] = img.shape[1]
                    grp['img_h'] = img.shape[0]

                except Exception as e:
                    print(batch)
                    print(e)
                    continue
