import argparse
import datetime
import logging
import math
import os
import random
import sys
import time
import warnings
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.distillation.distillation import calculate_rpn_distillation_loss
from maskrcnn_benchmark.distillation.finetune_distillation_all import (
    soften_proposales_iou_targets,
    calculate_roi_scores_distillation_losses_old_raw,
    calculate_roi_scores_distillation_losses_new_raw,
)
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.pseudo_labels import merge_pseudo_labels
from maskrcnn_benchmark.solver import make_optimizer, make_lr_scheduler
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

warnings.filterwarnings("ignore", category=UserWarning)


def do_train(
        model_source,
        model_finetune,
        model_target,
        data_loader,
        optimizer,
        scheduler,
        checkpointer_target,
        device,
        checkpoint_period,
        arguments_target,
        summary_writer,
        cfg,
):
    logger = logging.getLogger("maskrcnn_benchmark_target_model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments_target["iteration"]
    print(start_iter)
    model_target.train()
    model_finetune.eval()
    start_training_time = time.time()
    end = time.time()
    average_distillation_loss = 0
    average_faster_rcnn_loss = 0

    for iteration, (images, targets, _, img_id, _) in tqdm(enumerate(data_loader, start_iter), total=max_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments_target["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.no_grad():
            soften_result, _, soften_proposal, _, _, _, rpn_output_source, _ = (
                model_source.generate_soften_proposal(images)
            )
            feature_finetune, rpn_output_finetune = (
                model_finetune.generate_features_rpn_output(images)
            )
            finetune_result, _, _ = model_finetune.forward(
                images, features=feature_finetune, proposals=soften_proposal
            )
            if cfg.PSEUDO_LABELS.ENABLE:
                pseudo_targets, prev_features, _ = model_source.generate_pseudo_targets(
                    images
                )

        ###### BRIDGE THE PAST ######
        merged_target = merge_pseudo_labels(
            pseudo_targets,
            targets,
            cfg.IOU_LOW,
            cfg.IOU_HIGH,
            cfg.LOW_WEIGHT,
            cfg.HIGH_WEIGHT,
        )
        #############################

        loss_dict_target, feature_target, rpn_output_target = model_target(
            images,
            merged_target,
            pseudo_targets=pseudo_targets,
            rpn_output_source=rpn_output_source,
        )

        faster_rcnn_losses = sum(loss for loss in loss_dict_target.values())

        target_result, _, _ = model_target.forward(
            images, targets, features=feature_target, proposals=soften_proposal
        )

        distillation_losses = torch.tensor(0.0).to(device)

        ###### DISTILLATION WITH FUTURE ######
        if cfg.DIST.CLS > 0:
            class_distillation_loss = torch.tensor([]).to(device)
            bbox_distillation_loss = torch.tensor([]).to(device)
            soften_indexes, finetune_indexes = soften_proposales_iou_targets(
                soften_proposal, targets
            )
            delta = 0
            for img_idx in range(len(soften_proposal)):
                soften_indexes[img_idx] += delta
                finetune_indexes[img_idx] += delta
                delta += len(soften_proposal[img_idx])
            soften_indexes = torch.cat(soften_indexes, dim=0)
            finetune_indexes = torch.cat(finetune_indexes, dim=0)

            if len(soften_indexes) > 0:
                dis_soften_result = (
                    soften_result[0][soften_indexes],
                    soften_result[1][soften_indexes],
                )
                dis_target_soften_result = (
                    target_result[0][soften_indexes],
                    target_result[1][soften_indexes],
                )
                dis_finetune_soften_result = (
                    finetune_result[0][soften_indexes],
                    finetune_result[1][soften_indexes],
                )

                (
                    soften_class_distillation_loss_raw,
                    soften_bbox_distillation_loss_raw,
                ) = calculate_roi_scores_distillation_losses_old_raw(
                    dis_soften_result,
                    dis_finetune_soften_result,
                    dis_target_soften_result,
                )
                class_distillation_loss = torch.cat(
                    [class_distillation_loss, soften_class_distillation_loss_raw], dim=0
                )
                bbox_distillation_loss = torch.cat(
                    [bbox_distillation_loss, soften_bbox_distillation_loss_raw], dim=0
                )

            if len(finetune_indexes) > 0:
                dis_finetune_result = (
                    finetune_result[0][finetune_indexes],
                    finetune_result[1][finetune_indexes],
                )
                dis_soften_finetune_result = (
                    soften_result[0][finetune_indexes],
                    soften_result[1][finetune_indexes],
                )
                dis_target_finetune_result = (
                    target_result[0][finetune_indexes],
                    target_result[1][finetune_indexes],
                )
                (
                    finetune_class_distillation_loss_raw,
                    finetune_bbox_distillation_loss_raw,
                ) = calculate_roi_scores_distillation_losses_new_raw(
                    dis_soften_finetune_result,
                    dis_finetune_result,
                    dis_target_finetune_result,
                )
                class_distillation_loss = torch.cat(
                    [class_distillation_loss, finetune_class_distillation_loss_raw],
                    dim=0,
                )
                bbox_distillation_loss = torch.cat(
                    [bbox_distillation_loss, finetune_bbox_distillation_loss_raw], dim=0
                )

            class_distillation_loss = class_distillation_loss.mean()
            bbox_distillation_loss = bbox_distillation_loss.mean()
            distillation_losses += cfg.DIST.CLS * (
                    class_distillation_loss + bbox_distillation_loss
            )
        ###################################

        if cfg.DIST.RPN:
            rpn_distillation_losses = calculate_rpn_distillation_loss(
                rpn_output_source,
                rpn_output_target,
                cls_loss="filtered_l2",
                bbox_loss="l2",
                bbox_threshold=0.1,
            )
            distillation_losses += rpn_distillation_losses

        distillation_dict = {}
        distillation_dict["distillation_loss"] = distillation_losses.clone().detach()
        loss_dict_target.update(distillation_dict)

        losses = faster_rcnn_losses + distillation_losses

        loss_dict_reduced = reduce_loss_dict(loss_dict_target)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if not math.isfinite(loss := losses_reduced.item()):
            print(f"Loss is {loss}, stop training")
            sys.exit(1)

        if (iteration - 1) > 0:
            average_distillation_loss = (
                                                average_distillation_loss * (iteration - 1) + distillation_losses
                                        ) / iteration
            average_faster_rcnn_loss = (
                                               average_faster_rcnn_loss * (iteration - 1) + faster_rcnn_losses
                                       ) / iteration
        else:
            average_distillation_loss = distillation_losses
            average_faster_rcnn_loss = faster_rcnn_losses

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            loss_global_avg = meters.loss.global_avg
            loss_median = meters.loss.median
            summary_writer.add_scalar(
                "train_loss_global_avg", loss_global_avg, iteration
            )
            summary_writer.add_scalar("train_loss_median", loss_median, iteration)
            summary_writer.add_scalar("train_loss_raw", losses_reduced, iteration)
            summary_writer.add_scalar(
                "distillation_losses_raw", distillation_losses, iteration
            )
            summary_writer.add_scalar(
                "faster_rcnn_losses_raw", faster_rcnn_losses, iteration
            )
            summary_writer.add_scalar(
                "distillation_losses_avg", average_distillation_loss, iteration
            )
            summary_writer.add_scalar(
                "faster_rcnn_losses_avg", average_faster_rcnn_loss, iteration
            )

        if iteration % checkpoint_period == 0:
            # checkpointer_target.save("model_last", **arguments_target)
            checkpointer_target.save(
                "model_{:07d}".format(iteration), **arguments_target
            )
        if iteration == max_iter:
            checkpointer_target.save("model_final", **arguments_target)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iter
        )
    )


def initalizeTargetCls_MiB(cfg, model_source, model_target):
    n_old_classes = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
    cls_score_source = model_source.roi_heads.box.predictor.cls_score
    with torch.no_grad():
        model_target.roi_heads.box.predictor.cls_score.weight[n_old_classes + 1:] = (
            cls_score_source.weight[0]
        )
        model_target.roi_heads.box.predictor.cls_score.bias[n_old_classes + 1:] = (
                cls_score_source.bias[0]
                - torch.log(torch.Tensor([n_old_classes]).to(cls_score_source.bias.device))
        )
    return model_target


def train(
        cfg_source,
        cfg_finetune,
        cfg_target,
        logger_target,
):
    device = torch.device(cfg_source.MODEL.DEVICE)

    model_source = build_detection_model(cfg_source)
    model_finetune = build_detection_model(cfg_finetune)
    model_target = build_detection_model(cfg_target)
    model_target.to(device)
    model_finetune.to(device)
    model_source.to(device)

    optimizer = make_optimizer(cfg_target, model_target)
    scheduler = make_lr_scheduler(cfg_target, optimizer)

    arguments_target = {"iteration": 0}
    arguments_source = {"iteration": 0}
    arguments_finetune = {"iteration": 0}

    output_dir_target = cfg_target.OUTPUT_DIR
    output_dir_finetune = cfg_finetune.OUTPUT_DIR
    output_dir_source = cfg_source.OUTPUT_DIR
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)

    save_to_disk = get_rank() == 0
    checkpointer_source = DetectronCheckpointer(
        cfg_source,
        model_source,
        optimizer=None,
        scheduler=None,
        save_dir=output_dir_source,
        save_to_disk=save_to_disk,
    )
    extra_checkpoint_data_source = checkpointer_source.load(
        cfg_source.MODEL.SOURCE_WEIGHT
    )
    print("cfg_source.MODEL.SOURCE_WEIGHT:", cfg_source.MODEL.SOURCE_WEIGHT)

    checkpointer_finetune = DetectronCheckpointer(
        cfg_finetune,
        model_finetune,
        optimizer=None,
        scheduler=None,
        save_dir=output_dir_finetune,
        save_to_disk=save_to_disk,
    )
    extra_checkpoint_data_finetune = checkpointer_finetune.load(
        cfg_finetune.MODEL.FINETUNE_WEIGHT
    )
    print("cfg_finetune.MODEL.FINETUNE_WEIGHT:", cfg_finetune.MODEL.FINETUNE_WEIGHT)

    checkpointer_target = DetectronCheckpointer(
        cfg_target,
        model_target,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=output_dir_target,
        save_to_disk=save_to_disk,
        logger=logger_target,
    )
    extra_checkpoint_data_target = checkpointer_target.load(cfg_target.MODEL.WEIGHT)
    print("cfg_target.MODEL.WEIGHT:", cfg_target.MODEL.WEIGHT)
    arguments_source.update(extra_checkpoint_data_source)
    arguments_finetune.update(extra_checkpoint_data_finetune)
    arguments_target.update(extra_checkpoint_data_target)

    # Parameter initialization
    if cfg_target.DIST.INIT:
        model_target = initalizeTargetCls_MiB(cfg_target, model_source, model_target)

    data_loader = make_data_loader(
        cfg_target,
        is_train=True,
        start_iter=arguments_target["iteration"],
    )
    print("Finish loading data")
    checkpoint_period = cfg_target.SOLVER.CHECKPOINT_PERIOD

    del (
        checkpointer_source,
        arguments_source,
        extra_checkpoint_data_source,
        extra_checkpoint_data_target,
    )
    do_train(
        model_source,
        model_finetune,
        model_target,
        data_loader,
        optimizer,
        scheduler,
        checkpointer_target,
        device,
        checkpoint_period,
        arguments_target,
        summary_writer,
        cfg_target,
    )

    checkpointer_target.save("model_trimmed", trim=True, **arguments_target)

    return model_target


def test(cfg):
    if get_rank() != 0:
        return
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    print("#### The model will be saved at {} in test phase.".format(output_dir))
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    print("#### The model weight used in test phase is: {}.".format(cfg.MODEL.WEIGHT))
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    summary_writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            alphabetical_order=cfg.TEST.COCO_ALPHABETICAL_ORDER,
            summary_writer=summary_writer,
        )
        # pdb.set_trace()
        len_old = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        len_new = len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
        assert len(result) == (len_old + len_new), \
                "The length of result is not equal to the number of classes."
        ap_old = result[:len_old].mean()
        ap_new = result[len_old:].mean()
        ap_all = result.mean()
        with open(os.path.join("log", f"result.txt"), "a") as fid:
            fid.write(f"{cfg.DATASET} {cfg.NAME} Task {cfg.TASK} Step {cfg.STEP}\n")
            fid.write(f"mAP Old: {ap_old:.2f}, mAP New: {ap_new:.2f}, mAP All: {ap_all:.2f}\n\n")


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(random.randint(1, 1000))

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("-n", "--name", default="EXP", type=str)
    parser.add_argument("-t", "--task", type=str, default="15-5")
    parser.add_argument("-s", "--step", default=1, type=int)
    parser.add_argument("-e", "--eval_only", default=False, type=bool)

    parser.add_argument("--ist", default=False, action="store_true")
    parser.add_argument("--rpn", default=False, action="store_true")
    parser.add_argument("--feat", default="no", type=str, choices=["no", "std", "align", "att", "ard"])
    parser.add_argument("--uce", default=False, action="store_true")
    parser.add_argument("--init", default=False, action="store_true")
    parser.add_argument("--bg", default=False, action="store_true")
    parser.add_argument("--inv", default=False, action="store_true")
    parser.add_argument("--mask",default=1.0,type=float,)
    parser.add_argument("--cls",default=1.0,type=float,)
    parser.add_argument("--alpha",default=1.0,type=float,)

    parser.add_argument("--beta",default=1.0,type=float)
    parser.add_argument("--gamma",default=1.0,type=float,)
    parser.add_argument("--dist_type",default="l2",type=str,choices=["uce", "ce", "ce_ada", "ce_all", "l2", "none"],)
    parser.add_argument("-l", "--iou_low", default=0.4, type=float)
    parser.add_argument("-high", "--iou_high", default=0.7, type=float)
    parser.add_argument("-lw", "--low_weight", default=1.0, type=float)
    parser.add_argument("-hw", "--high_weight", default=0.3, type=float)
    parser.add_argument("-lr", "--LR", default=-1, type=float)
    args = parser.parse_args()

    target_model_config_file = (f"configs/{args.dataset}/{args.task}/target.yaml")
    full_name = f"{args.name}/STEP{args.step}"  # if args.step > 1 else args.name

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(target_model_config_file)
    cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = (
            len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
    )
    cfg_source.OUTPUT_DIR += "/" + args.dataset + "/" + args.task + "/" + full_name + "/SRC"
    cfg_source.TENSORBOARD_DIR += "/" + args.dataset + "/" + args.task + "/" + full_name
    cfg_source.freeze()

    cfg_finetune = cfg.clone()
    cfg_finetune.merge_from_file(target_model_config_file)
    cfg_finetune.MODEL.ROI_BOX_HEAD.NUM_CLASSES = (
            len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES) + 1
    )
    cfg_finetune.OUTPUT_DIR += "/" + args.dataset + "/" + args.task + "/" + full_name + "/FINETUNE"
    cfg_finetune.freeze()

    # LOAD THEN MODIFY PARS FROM CLI
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)
    # if args.step == 2:
    #     cfg_target.MODEL.WEIGHT = f"output/{args.name}/model_trimmed.pth"
    if args.step >= 2:
        base = "log" if not args.ist else "mask_out"
        cfg_target.MODEL.WEIGHT = (
            f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_trimmed.pth"
        )
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES = (
                len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        )
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES += args.step * cfg_target.CLS_PER_STEP
        print(cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES += (
            cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                : (args.step - 1) * cfg_target.CLS_PER_STEP
            ]
        )
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = (
            cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                args.step * cfg_source.CLS_PER_STEP:
            ]
        )
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES = (
            cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                (args.step - 1)
                * cfg_target.CLS_PER_STEP: args.step
                                           * cfg_source.CLS_PER_STEP
            ]
        )
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)

    cfg_target.DIST.MASK = args.mask if args.ist else 0.0
    cfg_target.DIST.RPN = args.rpn
    cfg_target.DIST.INV_CLS = args.inv
    cfg_target.DIST.FEAT = args.feat
    if args.cls != -1:
        cfg_target.DIST.CLS = args.cls
    else:
        cfg_target.DIST.CLS = (
                len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
                / cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        )
    cfg_target.DIST.TYPE = args.dist_type
    cfg_target.DIST.INIT = args.init
    cfg_target.DIST.ALPHA = args.alpha
    cfg_target.DIST.BETA = args.beta
    cfg_target.DIST.GAMMA = args.gamma
    cfg_target.DIST.BG = args.bg

    cfg_target.OUTPUT_DIR += "/" + args.dataset + "/" + args.task + "/" + full_name
    cfg_target.INCREMENTAL = args.uce
    cfg_target.TENSORBOARD_DIR += "/" + args.dataset + "/" + args.task + "/" + full_name
    cfg_target.TASK = args.task
    cfg_target.STEP = args.step
    cfg_target.NAME = args.name
    cfg_target.IOU_LOW = args.iou_low
    cfg_target.IOU_HIGH = args.iou_high
    cfg_target.LOW_WEIGHT = args.low_weight
    cfg_target.HIGH_WEIGHT = args.high_weight
    if args.LR > 0:
        cfg_target.SOLVER.BASE_LR = args.LR
    # if args.weight is not "NONE":
    #     cfg_target.MODEL.WEIGHT = args.weight

    cfg_target.freeze()

    output_dir_target = cfg_target.OUTPUT_DIR
    if output_dir_target:
        mkdir(output_dir_target)
    output_dir_source = cfg_source.OUTPUT_DIR
    if output_dir_source:
        mkdir(output_dir_source)
    tensorboard_dir = cfg_target.TENSORBOARD_DIR
    if tensorboard_dir:
        mkdir(tensorboard_dir)

    logger_target = setup_logger("maskrcnn_benchmark_target_model", output_dir_target, get_rank())
    logger_target.info("config yaml file for target model: {}".format(target_model_config_file))

    if not args.eval_only:
        train(
            cfg_source,
            cfg_finetune,
            cfg_target,
            logger_target,
        )
    test(cfg_target)


if __name__ == "__main__":
    main()
