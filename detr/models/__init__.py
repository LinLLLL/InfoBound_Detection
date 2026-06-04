# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import nn
from .backbone import build_backbone
from .deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess
from .detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from .deformable_detr import SetInfoBoundCriterion as DefSetCriterion_InfoBound
from .def_matcher import build_matcher as build_def_matcher
from .detr_matcher import build_matcher as build_detr_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .transformer import build_transformer


def build_model(args):
    if args.dataset_file == 'coco':
        if args.dataset == 'coco_ood_val' or args.dataset == 'openimages_ood_val':
            if args.eval_bdd:
                num_classes = 10
            else:
                num_classes = 20
        elif args.dataset == 'coco_ood_val_bdd':
            num_classes = 10
        else:
            num_classes = 90
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    elif args.dataset_file == 'airbus':
        num_classes = 1
    elif args.dataset_file == 'bdd':
        num_classes = 10
    else:
        num_classes = 20
    # num_classes += 1
    print("-----------num of classes:{}-----------".format(num_classes))
    device = torch.device(args.device)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    if args.mm_energy:
        weight_dict['loss_mm'] = args.mm_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef



    weight_dict['loss_vmf'] = 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})

        # only in def detr impl.
        if args.model == 'deformable_detr':
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.object_embedding_loss:
        losses.append('object_embedding')
        weight_dict['loss_object_embedding'] = args.object_embedding_coef
    if args.masks:
        losses += ["masks"]
    if args.mm_energy:
        losses += ['mm_energy']

    backbone = build_backbone(args)

    if args.model == 'deformable_detr':
        transformer = build_deforamble_transformer(args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head,
            # objectness = args.objectness,
            args=args
        )
        matcher = build_def_matcher(args)

        # criterion = DefSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, args=args)
        criterion = DefSetCriterion_InfoBound(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, args=args)
        print('Using the InfoBound regularization loss!')
        # postprocessors = {'bbox': DefPostProcess()}
        postprocessors = {'bbox': PostProcessOAGDINO()}

    elif args.model == 'detr':
        transformer = build_transformer(args)
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        matcher = build_detr_matcher(args)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, args.eos_coef,
                                     losses, object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': DETRPostProcess()}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


############################# PostProcess for OOD eval #######################################
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class PostProcessOAGDINO(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # out_pen_features = outputs['pen_features'] # object features before the center_project
        out_pen_features = outputs['bbox_embd']


        out_project_features = None
        out_sampling_cls = None
        out_godin_h = None
        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_indexes = torch.nonzero(prob.reshape(-1) > 0.1).view(1, -1)
        scores = prob.reshape(-1)[topk_indexes[0]].view(1,-1)
        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])

        # topk_values, topk_indexes = torch.topk(out_logits.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        boxes = box_cxcywh_to_xyxy(out_bbox)  #
        boxes = boxes.reshape(1, -1, out_bbox.shape[-1])
        out_bbox = out_bbox.reshape(1, -1, out_bbox.shape[-1])
        out_logits = out_logits.reshape(1, -1, out_logits.shape[-1])
        out_pen_features = out_pen_features.reshape(1, -1, out_pen_features.shape[-1])
        # print(topk_boxes.unsqueeze(-1).repeat(1,1,4).shape, boxes.shape, out_logits.shape, out_bbox.shape)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        new_ped_logits = torch.gather(out_logits, 1,
                                      topk_boxes.unsqueeze(-1).repeat(1, 1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1, 1, out_pen_features.shape[2]))

        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_project_features.shape[2]))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes1 * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob,
                    "logits_for_ood_eval": new_ped_logits, 'boxes_for_ood_eval': boxes,
                    "pen_features": out_pen_features,  # the bbox_embd for visualization and ood detection by KNN and vMF
                    "project_features": out_project_features,
                    "sampling_cls": out_sampling_cls,
                    'godin_h': out_godin_h,
                    "am_for_ood": None} for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]

        return results

    @torch.no_grad()
    def forward_maha(self, outputs, targets, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # out_pen_features = outputs['pen_features']
        out_pen_features = outputs['bbox_embd']
        # enhanced_text_features = outputs['enhanced_text_features']
        out_project_features = None
        out_sampling_cls = None
        out_godin_h = None

        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']


        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), len(targets[0]['labels']), dim=1)
        scores = topk_values

        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        new_ped_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1,1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1, 1, out_pen_features.shape[2]))

        out_pen_features = torch.cat([out_pen_features, labels.unsqueeze(2)], -1)

        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1,1, out_project_features.shape[2]))
            out_project_features = torch.cat([out_project_features, labels.unsqueeze(2)], -1)


        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes1 * scale_fct[:, None, :]


        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob,
                    "logits_for_ood_eval": new_ped_logits, 'boxes_for_ood_eval': boxes,
                    "pen_features": out_pen_features,  # the bbox_embd for visualization and ood detection by KNN and vMF
                    "project_features": out_project_features,
                    # "enhanced_text_features": enhanced_text_features,  # the prototype of the ID classes for ood detection
                    "sampling_cls": out_sampling_cls,
                    'godin_h': out_godin_h,
                    "am_for_ood": None} \
                for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]

        return results



