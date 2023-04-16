import argparse
import os
import os.path as osp

import torch
import torchvision.transforms as transforms

import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device


MMSEG_CONFIG = "/home/btrabucc/mmsegmentation/work_dirs/\
segformer_mit-b0_224x224_160k_thor/segformer_mit-b0_224x224_160k_thor.py"
MMSEG_CHECKPOINT = "/home/btrabucc/mmsegmentation/\
work_dirs/segformer_mit-b0_224x224_160k_thor/iter_320000.pth"


def load_segformer(gpu_id=0):

    cfg = mmcv.Config.fromfile(MMSEG_CONFIG)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.gpu_ids = [gpu_id]
    cfg.model.train_cfg = None

    model = build_segmentor(
        cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)

    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(
        model, MMSEG_CHECKPOINT, map_location='cpu')

    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    cfg.device = get_device()
    model = revert_sync_batchnorm(model)

    model = build_dp(model, cfg.device, 
                     device_ids=cfg.gpu_ids)

    model.eval()
    
    transform = torch.nn.Sequential(
        transforms.Resize((224, 224)),
        transforms.Normalize(
            [123.675 / 255, 
             116.28 / 255, 
             103.53 / 255], 
            [58.395 / 255, 
             57.12 / 255, 
             57.375 / 255]))

    return model, transform