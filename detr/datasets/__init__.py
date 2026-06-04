# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .coco import build_voc, build_oi, build_bdd


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset == 'voc' and 'ood' not in image_set:
        return build_voc(image_set, args)
    if args.dataset == 'bdd' and 'ood' not in image_set:
        return build_bdd(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if 'val_ood_coco' in image_set:
        return build_coco('val', args)
    if 'val_ood_oi' in image_set:
        return build_oi('val', args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


