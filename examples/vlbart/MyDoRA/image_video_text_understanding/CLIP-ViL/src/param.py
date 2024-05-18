# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self,
                 l_layers=12,
                 x_layers=5,
                 r_layers=0,
                 use_clip=False,
                 visualbert_style=False,
                 freeze_clip=False,
                 clip_model_name="ViT-B/32",
                 drop_boxes=False,
                 vilt_style=False,
                 use_vit=False,
                 reset_pos_embedding=False,
                 sub_sampling=False,
                 sub_feat_num=36,
                 use_positional_embedding=False,
                 use_max_pooling=False,
                 pos_num=25
                 ):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers

        if use_clip and clip_model_name == "ViT-B/32":
            self.visual_feat_dim = 768
        elif use_vit:
            self.visual_feat_dim = 768
        elif use_clip and clip_model_name == "RN50x4":
            self.visual_feat_dim = 2560
        else:
            self.visual_feat_dim = 2048

        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {
            'obj': (self.obj_id_num, 'ce', (-1,), 1/0.15),
            'attr': (self.attr_id_num, 'ce', (-1,), 1/0.15),
            'feat': (self.visual_feat_dim, 'l2', (-1, self.visual_feat_dim), 1/0.15),
        }
        
        self.use_clip = use_clip
        self.visualbert_style = visualbert_style
        self.freeze_clip = freeze_clip
        self.clip_model_name = clip_model_name
        self.drop_boxes = drop_boxes
        self.vilt_style = vilt_style
        self.use_vit = use_vit
        self.reset_pos_embedding = reset_pos_embedding
        self.sub_sampling = sub_sampling
        self.sub_feat_num = sub_feat_num
        self.use_positional_embedding = use_positional_embedding
        self.use_max_pooling = use_max_pooling
        self.pos_num = pos_num

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0, type=int)


    parser.add_argument("--input_raw_images", action='store_true')
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--use_clip", action='store_true')
    parser.add_argument("--visualbert_style", action="store_true")
    parser.add_argument("--report_step", default=200, type=int)
    
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--freeze_clip", action="store_true")
    parser.add_argument("--vqa_style_transform", action="store_true")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_adam_for_visual", action = "store_true")
    parser.add_argument(
        "--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--aspect_ratio_group_factor", type=int, default=0)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    

    parser.add_argument("--use_h5_file", action="store_true")
    parser.add_argument("--drop_boxes", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--image_size_min", default=384, type=int)
    parser.add_argument("--image_size_max", default=640, type=int)
    parser.add_argument("--loss_scale", default=1, type=int)
    parser.add_argument("--dynamic_padding", action="store_true")
    parser.add_argument("--vilt_style", action="store_true")
    parser.add_argument("--add_zero_padding", action='store_true')

    parser.add_argument("--warmup_ratio", default=0.1, type=float)

    parser.add_argument("--use_vit", action="store_true")
    parser.add_argument("--limit_source", default="", type=str)

    parser.add_argument("--save_step", default=-1, type=int)
    parser.add_argument("--reset_pos_embedding", action="store_true")

    parser.add_argument("--sub_sampling", action="store_true")
    parser.add_argument("--sub_feat_num", default=36, type=int)

    parser.add_argument("--use_separate_optimizer_for_visual", action="store_true")

    parser.add_argument("--sgd_lr", default=0.001, type=float)
    parser.add_argument("--sgd_momentum", default=0.0, type=float)
    parser.add_argument("--sgd_weight_decay", default=0.0004, type=float)
    parser.add_argument("--schedule", default="2,5", type=str)

    parser.add_argument("--compress_data", action="store_true")
    parser.add_argument("--use_lmdb", action="store_true")

    parser.add_argument("--start_epoch", default=0, type=int)

    parser.add_argument("--use_positional_embedding",action="store_true")
    parser.add_argument("--use_max_pooling", action="store_true")

    parser.add_argument("--pos_num", default = 25, type = int)
    parser.add_argument("--not_load_scheduler", action="store_true")
    parser.add_argument("--pad_square", action="store_true")
    parser.add_argument("--not_load_adam_optimizer", action="store_true")
    parser.add_argument("--separate_image", action="store_true")

    # for adapters
    parser.add_argument('--use_adapter', action="store_true")
    parser.add_argument('--reduction_factor', type=int, default=16)
    parser.add_argument('--use_vis_adapter', action="store_true")
    parser.add_argument('--use_bn', action="store_true")
    parser.add_argument('--use_gate', action="store_true")

    # for side network
    parser.add_argument('--use_side_transformers', action="store_true")
    parser.add_argument('--load_side_pretrained_weights', default="")
    parser.add_argument('--samples_for_fisher', type=int, default=1024)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.schedule = [int(i) for i in args.schedule.split(",")]
    assert(not args.use_lmdb)
    assert(args.add_zero_padding)
    return args


args = parse_args()
VISUAL_CONFIG=VisualConfig(
    use_clip=args.use_clip,
    visualbert_style=args.visualbert_style,
    freeze_clip=args.freeze_clip,
    clip_model_name=args.clip_model_name,
    drop_boxes=args.drop_boxes,
    vilt_style=args.vilt_style,
    use_vit=args.use_vit,
    reset_pos_embedding=args.reset_pos_embedding,
    sub_sampling=args.sub_sampling,
    sub_feat_num=args.sub_feat_num,
    use_positional_embedding=args.use_positional_embedding,
    use_max_pooling=args.use_max_pooling,
    pos_num=args.pos_num)