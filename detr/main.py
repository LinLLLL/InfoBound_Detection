# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_infobound import evaluate, evaluate_ood_id, evaluate_ood_ood, evaluate_uda, train_one_epoch, viz
from models import build_model
from models.backbone import build_swav_backbone, build_swav_backbone_old
from util.default_args import set_model_defaults, get_args_parser

PRETRAINING_DATASETS = ['imagenet', 'imagenet100', 'coco_pretrain', 'airbus_pretrain']


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.random_seed:
        args.seed = np.random.randint(0, 1000000)

    if args.resume:
        checkpoint_args = torch.load(args.resume, map_location='cpu')['args']
        args.seed = checkpoint_args.seed
        print("Loaded random seed from checkpoint:", checkpoint_args.seed)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")
    swav_model = None
    if args.dataset in PRETRAINING_DATASETS:
        if args.obj_embedding_head == 'head':
            swav_model = build_swav_backbone(args, device)
        elif args.obj_embedding_head == 'intermediate':
            swav_model = build_swav_backbone_old(args, device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train, dataset_val, dataset_val_ood = get_datasets(args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(
                dataset_val, shuffle=False)
            sampler_val_ood = samplers.NodeDistributedSampler(
                dataset_val_ood, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(
                dataset_val, shuffle=False)
            sampler_val_ood = samplers.DistributedSampler(
                dataset_val_ood, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_val_ood = torch.utils.data.SequentialSampler(dataset_val_ood)
    coco_evaluator = None
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    data_loader_val_ood = DataLoader(dataset_val_ood, args.batch_size, sampler=sampler_val_ood,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    if args.dataset_file == 'voc':
        idx = np.array(range(len(dataset_train)))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        index = idx[:10000]
        dataset_train_subsample = torch.utils.data.Subset(dataset_train, index)
        sampler = torch.utils.data.SequentialSampler(dataset_train_subsample)
        data_loader_train = DataLoader(dataset_train, 1, sampler=sampler,
                                     drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]

    # sample few-shot uda data from both the covariate-shifted ood val and the semantic-shifted ood val
    ####################################################################################
    ####################################################################################
    ####################################################################################
    rng = np.random.default_rng(seed)
    idx = np.array(range(len(dataset_val)))
    rng.shuffle(idx)
    uda_len = 20
    index = idx[:uda_len]
    dataset_val_uda = torch.utils.data.Subset(dataset_val, index)
    sampler_val_uda = torch.utils.data.SequentialSampler(dataset_val_uda)
    data_loader_val_uda = DataLoader(dataset_val_uda, 1, sampler=sampler_val_uda,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                     pin_memory=True)

    idx = np.array(range(len(dataset_val_ood)))
    rng.shuffle(idx)
    uda_len = 20
    index = idx[:uda_len]
    dataset_val_ood_uda = torch.utils.data.Subset(dataset_val_ood, index)
    sampler_val_ood_uda = torch.utils.data.SequentialSampler(dataset_val_ood_uda)
    data_loader_val_ood_uda = DataLoader(dataset_val_ood_uda, 1, sampler=sampler_val_ood_uda,
                                         drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                         pin_memory=True)
    all_uda = []
    in_uda = []
    out_uda = []
    for samples, targets in data_loader_val_uda:
        all_uda.append(samples)
        in_uda.append(samples)
    for samples, targets in data_loader_val_ood_uda:
        all_uda.append(samples)
        out_uda.append(samples)

    if args.mixture_ood:
        samples_uda = [data_loader_val_uda, data_loader_val_ood_uda]
    else:
        samples_uda = [data_loader_val_ood_uda]  # only semantic-out ood samples are used

    ##################################################################################
    ##################################################################################
    ##################################################################################

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not \
                 match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset_file == "coco" or args.dataset_file == "airbus" or args.dataset_file == "bdd":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # remove useless class embed
            if 'class_embed' in k:
                del state_dict[k]
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(msg)


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (
            k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            ##################### fine-tuning setting  ############################
            params_ft = []
            for name, param in model_without_ddp.named_parameters():
                param.requires_grad = False
                if name.split(".")[0] in ['class_embed', 'bbox_embed', 'transformer'] and 'encoder' not in name:
                    param.requires_grad = True
                    params_ft.append(param)
                print(name, param.requires_grad)
            param_dicts = [{"params": params_ft, "lr": args.lr}]

            if args.sgd:
                optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                            weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                              weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            ##################### fine-tuning setting  ############################

            # todo: this is a hack for doing experiment that resume from checkpoint and
            #  also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        # check the resumed model
        # if (not args.eval and not args.viz and args.dataset in ['bdd', 'voc']):
        #     test_stats, coco_evaluator = evaluate_ood_id(args, model, criterion, postprocessors, data_loader_val,
        #                     base_ds, device, args.output_dir, args.output_dir, args.dataset,
        #                     args.viz_prediction_results, 0)

    if args.eval and not args.viz:
        #################################################################
        #################################################################
        #################################################################
        if args.dataset_file == 'bdd':
            threshold = -1
        elif args.ood_data == 'coco':
            threshold = -1
        else:
            threshold = -1
        print('We set the threshold as {}.'.format(threshold))

        all_logits_uda, pen_features_uda, pred_boxes_uda = evaluate_uda(model, criterion, postprocessors, all_uda,
                                                                        device)
        if args.dataset_file == 'bdd':
            all_logits_uda = all_logits_uda.reshape(-1, 10)
        else:
            all_logits_uda = all_logits_uda.reshape(-1, 20)
        energy_uda = -torch.logsumexp(all_logits_uda, dim=1)

        uda_labels = (1 * (energy_uda <= threshold))
        uda_labels = [uda_labels[:len(data_loader_val_uda)], uda_labels[len(data_loader_val_uda):]]
        ##################################################################
        #################################################################
        #################################################################
        epochN = int(args.resume.split('checkpoint00')[1].split('.pth')[0]) if '00' in args.resume else 0
        
        if args.evaluate_mode == "id":
            evaluate_ood_id(args, model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args.output_dir, args.dataset, args.viz_prediction_results, epochN)
        else:
            evaluate_ood_ood(model, criterion, postprocessors, data_loader_val_ood, base_ds, device, args.output_dir, args.output_dir, args.dataset, args.viz_prediction_results, epochN)
        return
        
    if args.viz:
        viz(model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if epoch < -1:
            args.mm_energy = False
            uda_labels = None
            pred_boxes_uda = None
            threshold = None
        else:
            args.mm_energy = True
            ##################################################################
            if args.dataset_file == 'bdd':
                if args.ood_data == 'coco':
                    threshold = -1
                else:
                    threshold = -1.5
            elif args.ood_data == 'coco': # for voc to coco
                threshold = -1
            else:
                threshold = -1  # for voc to oi
            print('We set the threshold as {}.'.format(threshold))
            all_logits_in_uda, _, pred_boxes_in_uda = evaluate_uda(model, criterion, postprocessors, in_uda,
                                                                            device)
            all_logits_out_uda, _, pred_boxes_out_uda = evaluate_uda(model, criterion, postprocessors, out_uda,
                                                                            device)                                  
            all_logits_uda = torch.vstack([all_logits_in_uda, all_logits_out_uda])    
            pred_boxes_uda = torch.vstack([pred_boxes_in_uda, pred_boxes_out_uda])    
            
            if args.dataset_file == 'bdd':
                all_logits_uda = all_logits_uda.reshape(-1, 10)
            else:
                all_logits_uda = all_logits_uda.reshape(-1, 20)
            energy_uda = -torch.logsumexp(all_logits_uda, dim=1)

            uda_labels = (1 * (energy_uda <= threshold))
            print('-------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------')
            
            print('pseudo labels for semantic-in and semantic-out unlabeled smapels:')
            print("{}/{}, {}/{}".format(uda_labels[:all_logits_in_uda.shape[0]].sum().item(), all_logits_in_uda.shape[0], uda_labels[all_logits_in_uda.shape[0]:].sum().item(), all_logits_out_uda.shape[0]))
            print('-------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------')
            uda_labels = [uda_labels[:len(data_loader_val_uda)], uda_labels[len(data_loader_val_uda):]]
            ##################################################################

        train_stats = train_one_epoch(model, swav_model, criterion, data_loader_train,
                                      optimizer, device, epoch, args.clip_max_norm,
                                      postprocessors, samples_uda, uda_labels, pred_boxes_uda, threshold, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 4 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        if args.dataset in ['coco', 'voc', 'bdd'] and (epoch % args.eval_every == 0 or (epoch + 1) % 4 == 0):
            # test_stats, coco_evaluator = evaluate_ood_id(args, model, criterion, postprocessors, data_loader_val,
            #                                              base_ds, device, args.output_dir, args.output_dir,
            #                                              args.dataset, args.viz_prediction_results, epoch)
            # evaluate_ood_ood(model, criterion, postprocessors,
            #                  data_loader_val_ood, base_ds, device, args.output_dir,
            #                  args.output_dir, args.dataset, args.viz_prediction_results, epoch)
            test_stats = {}
        else:
            test_stats = {}

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if 'imagenet' not in args.dataset and coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_datasets(args):
    if args.dataset == 'coco':
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'coco_pretrain':
        from datasets.selfdet import build_selfdet
        dataset_train = build_selfdet(
            'train', args=args, p=os.path.join(args.coco_path, 'train2017'))
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'airbus':
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'airbus_pretrain':
        from datasets.selfdet import build_selfdet
        dataset_train = build_selfdet(
            'train', args=args, p=os.path.join(args.airbus_path, 'train_v2'))
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'imagenet':
        from datasets.selfdet import build_selfdet
        dataset_train = build_selfdet(
            'train', args=args, p=os.path.join(args.imagenet_path, 'train'))
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'imagenet100':
        from datasets.selfdet import build_selfdet
        dataset_train = build_selfdet(
            'train', args=args, p=os.path.join(args.imagenet100_path, 'train'))
        dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'voc':
        from datasets.torchvision_datasets.voc import VOCDetection
        from datasets.coco import make_coco_transforms
        if not args.maha_train:
            dataset_train = VOCDetection(args.voc_path, ["2007", "2012"], image_sets=['trainval', 'trainval'],
                                         transforms=make_coco_transforms('train'), filter_pct=args.filter_pct)
            dataset_val = VOCDetection(args.voc_path, ["2007"], image_sets=[
                                       'test'], transforms=make_coco_transforms('val'))
        else:
            dataset_train = VOCDetection(args.voc_path, ["2007", "2012"], image_sets=['trainval', 'trainval'],
                                         transforms=make_coco_transforms('train'), filter_pct=args.filter_pct)
            dataset_val = VOCDetection(args.voc_path, ["2007"], image_sets=[
                'test'], transforms=make_coco_transforms('val'))
        # dataset_train = build_dataset(image_set='train', args=args)
        # dataset_val = build_dataset(image_set='val', args=args)
    elif args.dataset == 'bdd':
        if not args.maha_train:
            dataset_train = build_dataset(image_set='train', args=args)
            dataset_val = build_dataset(image_set='val', args=args)
        else:
            dataset_train = build_dataset(image_set='val', args=args)
            dataset_val = build_dataset(image_set='train', args=args)
    else:
        raise ValueError(f"Wrong dataset name: {args.dataset}")

    if args.ood_data in ['coco',]:
        dataset_val_ood = build_dataset(image_set='val_ood_coco', args=args)
        print('we have a few-shot uda ood samples form coco...')
    else:
        dataset_val_ood = build_dataset(image_set='val_ood_oi', args=args)
        print('we have a few-shot uda ood samples form openimages...')

    return dataset_train, dataset_val, dataset_val_ood


def set_dataset_path(args):
    args.data_root = 'datasets/COCO_DATASET_ROOT/'
    args.bdd_root = 'datasets/BDD_DATASET_ROOT/'
    args.bdd_ann_root_train = 'datasets/BDD_DATASET_ROOT/train_bdd_converted.json'
    args.bdd_ann_root_test = 'datasets/BDD_DATASET_ROOT/val_bdd_converted.json'
    args.open_root = 'datasets/OpenImages/OpenImages/ood_classes_rm_overlap/images'
    args.open_ann_root = 'datasets/OpenImages/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json'
    args.coco_path = os.path.join(args.data_root)
    args.airbus_path = os.path.join(args.data_root, 'airbus-ship-detection')
    args.imagenet_path = os.path.join(args.data_root, 'ilsvrc')
    args.imagenet100_path = os.path.join(args.data_root, 'ilsvrc100')
    args.voc_path = 'datasets/VOC_DATASET_ROOT/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_dataset_path(args)
    set_model_defaults(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
