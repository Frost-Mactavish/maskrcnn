import argparse
import datetime
import logging
import math
import numpy as np
import os
import random
import sys
import time
import torch
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.distillation.attentive_distillation import (
    calculate_attentive_distillation_losses
)
from maskrcnn_benchmark.distillation.distillation import (
    calculate_feature_distillation_loss,
    calculate_roi_align_distillation,
    calculate_roi_distillation_losses,
    calculate_rpn_distillation_loss
)
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

warnings.filterwarnings("ignore", category=UserWarning)


def do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_target,
             device, checkpoint_period, arguments_target, summary_writer, cfg):
    logger = logging.getLogger("maskrcnn_benchmark_target_model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments_target["iteration"]
    model_target.train()
    model_source.eval()
    start_training_time = time.time()
    end = time.time()
    average_distillation_loss = 0
    average_faster_rcnn_loss = 0

    for iteration, (images, targets, _, idx) in tqdm(enumerate(data_loader, start_iter), total=max_iter):

        data_time = time.time() - end
        iteration = iteration + 1
        arguments_target["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        dist_type = cfg.DIST.TYPE

        with torch.no_grad():
            soften_result, soften_mask_logits, soften_proposal, feature_source, _, _, rpn_output_source, roi_align_features_source = \
                model_source.generate_soften_proposal(images)
        
        loss_dict_target, feature_target, _, _, rpn_output_target, target_proposals, _, target_soften_results \
            = model_target(images, targets, rpn_output_source=rpn_output_source)
        faster_rcnn_losses = sum(loss for loss in loss_dict_target.values())

        target_result, target_mask_logits, roi_align_features_target = model_target.forward(images, targets,
                                                                                            features=feature_target,
                                                                                            proposals=soften_proposal)

        if cfg.DIST.CLS > 0:
            distillation_losses = cfg.DIST.CLS * calculate_roi_distillation_losses(soften_result, target_result,
                                                                                   dist=dist_type)
        else:
            distillation_losses = torch.tensor(0.).to(device)

        if cfg.DIST.RPN:
            rpn_distillation_losses = calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target,
                                                                      cls_loss="filtered_l2", bbox_loss="l2",
                                                                      bbox_threshold=0.1)
            distillation_losses += rpn_distillation_losses
        if cfg.DIST.FEAT == "align":
            feature_distillation_losses = calculate_roi_align_distillation(roi_align_features_source,
                                                                           roi_align_features_target)
            distillation_losses += feature_distillation_losses
        elif cfg.DIST.FEAT == "std":
            feature_distillation_losses = calculate_feature_distillation_loss(feature_source, feature_target,
                                                                              loss="normalized_filtered_l1")
            distillation_losses += feature_distillation_losses
        elif cfg.DIST.FEAT == "att":
            feature_distillation_losses = calculate_attentive_distillation_losses(feature_source, feature_target)
            distillation_losses += 0.1 * feature_distillation_losses

        distillation_dict = {}
        distillation_dict["distillation_loss"] = distillation_losses.clone().detach()
        loss_dict_target.update(distillation_dict)

        losses = faster_rcnn_losses + distillation_losses

        loss_dict_reduced = reduce_loss_dict(loss_dict_target)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if not math.isfinite(loss := losses_reduced.item()):
            logger.info(f"Loss is {loss}, stop training")
            sys.exit(1)

        if (iteration - 1) > 0:
            average_distillation_loss = (average_distillation_loss * (iteration - 1) + distillation_losses) / iteration
            average_faster_rcnn_loss = (average_faster_rcnn_loss * (iteration - 1) + faster_rcnn_losses) / iteration
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
                meters.delimiter.join(["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}"
                                       ]).format(eta=eta_string, iter=iteration, meters=str(meters),
                                                 lr=optimizer.param_groups[0]["lr"],
                                                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
            loss_global_avg = meters.loss.global_avg
            loss_median = meters.loss.median
            summary_writer.add_scalar("train_loss_global_avg", loss_global_avg, iteration)
            summary_writer.add_scalar("train_loss_median", loss_median, iteration)
            summary_writer.add_scalar("train_loss_raw", losses_reduced, iteration)
            summary_writer.add_scalar("distillation_losses_raw", distillation_losses, iteration)
            summary_writer.add_scalar("faster_rcnn_losses_raw", faster_rcnn_losses, iteration)
            summary_writer.add_scalar("distillation_losses_avg", average_distillation_loss, iteration)
            summary_writer.add_scalar("faster_rcnn_losses_avg", average_faster_rcnn_loss, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer_target.save("model_last", **arguments_target)
        if iteration == max_iter:
            checkpointer_target.save("model_final", **arguments_target)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))


def train(cfg_source, cfg_target, logger_target):
    device = torch.device(cfg_source.MODEL.DEVICE)

    model_source = build_detection_model(cfg_source)
    model_target = build_detection_model(cfg_target)
    model_target.to(device)
    model_source.to(device)

    optimizer = make_optimizer(cfg_target, model_target)
    scheduler = make_lr_scheduler(cfg_target, optimizer)

    arguments_target = {"iteration": 0}
    arguments_source = {"iteration": 0}

    output_dir_target = cfg_target.OUTPUT_DIR
    output_dir_source = cfg_source.OUTPUT_DIR
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)

    save_to_disk = get_rank() == 0
    checkpointer_source = DetectronCheckpointer(cfg_source, model_source, optimizer=None, scheduler=None,
                                                save_dir=output_dir_source,
                                                save_to_disk=save_to_disk)
    extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)

    checkpointer_target = DetectronCheckpointer(cfg_target, model_target, optimizer=optimizer, scheduler=scheduler,
                                                save_dir=output_dir_target,
                                                save_to_disk=save_to_disk, logger=logger_target)
    extra_checkpoint_data_target = checkpointer_target.load(cfg_target.MODEL.WEIGHT)

    arguments_source.update(extra_checkpoint_data_source)
    arguments_target.update(extra_checkpoint_data_target)

    print("Start iteration: {0}".format(arguments_target["iteration"]))

    data_loader = make_data_loader(cfg_target, is_train=True, start_iter=arguments_target["iteration"])
    print("Finish loading data")
    checkpoint_period = cfg_target.SOLVER.CHECKPOINT_PERIOD

    do_train(model_source, model_target, data_loader,
             optimizer, scheduler, checkpointer_target,
             device, checkpoint_period, arguments_target,
             summary_writer, cfg_target)

    checkpointer_target.save("model_trimmed", trim=True, **arguments_target)

    return model_target


def test(cfg, cfg_target, model):
    iou_types = ("bbox",)
    dataset_names = cfg_target.DATASETS.TEST
    output_folders = [cfg_target.OUTPUT_DIR] * len(dataset_names) if cfg_target.OUTPUT_DIR else [None] * len(dataset_names)
    data_loaders_val = make_data_loader(cfg_target, is_train=False, is_distributed=False)
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg_target.MODEL.RETINANET_ON else cfg_target.MODEL.RPN_ONLY,
            device=cfg_target.MODEL.DEVICE,
            expected_results=cfg_target.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg_target.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            alphabetical_order=cfg_target.TEST.COCO_ALPHABETICAL_ORDER,
            summary_writer=summary_writer,
            cfg=cfg_target
        )

        if len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES) == 0:
            len_old = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
            ap_old = result[:len_old].mean() * 100
            ap_new = result[len_old:].mean() * 100
            ap_all = result.mean() * 100
            with open(os.path.join("log", f"result.txt"), "a") as fid:
                fid.write(f"{cfg_target.DATASET} {cfg_target.NAME} Task {cfg_target.TASK} Step {cfg_target.STEP}\n")
                fid.write(f"mAP Old: {ap_old:.1f}, mAP New: {ap_new:.1f}, mAP All: {ap_all:.1f}\n\n")


def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("-n", "--name", default="MMA", type=str)
    parser.add_argument("-d", "--dataset", default="DIOR", type=str)
    parser.add_argument("-t", "--task", default="19-1", type=str)
    parser.add_argument("-s", "--step", default=1, type=int)
    parser.add_argument("--rpn", default=False, action="store_true")
    parser.add_argument("--feat", default="no", choices=["no", "std", "align", "att"], type=str)
    parser.add_argument("--uce", default=False, action="store_true")
    parser.add_argument("--cls", default=1., type=float)
    parser.add_argument("--dist_type", default="l2", type=str,
                         choices=["uce", "ce", "ce_ada", "ce_all", "l2", "none"])
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    config_file = f"configs/{args.dataset}/{args.task}/target.yaml"
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    full_name = f"log/{args.dataset}/{args.task}/{args.name}"

    cfg_source = cfg.clone()
    cfg_target = cfg.clone()

    if args.step >= 2:
        model_weight = f"{full_name}/STEP{args.step - 1}/model_trimmed.pth"
        cfg_source.MODEL.WEIGHT = cfg_target.MODEL.WEIGHT = model_weight
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES += (args.step - 1) * cfg_source.CLS_PER_STEP
    else:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES += args.step * cfg_target.CLS_PER_STEP
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES += cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
            :(args.step - 1) * cfg_target.CLS_PER_STEP]
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
            args.step * cfg_source.CLS_PER_STEP:]
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES = cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
            (args.step - 1) * cfg_target.CLS_PER_STEP:
            args.step * cfg_source.CLS_PER_STEP]

    cfg_source.OUTPUT_DIR = f"{full_name}/STEP{args.step}/SRC"
    cfg_target.OUTPUT_DIR = f"{full_name}/STEP{args.step}"
    cfg_target.TENSORBOARD_DIR = f"{full_name}/STEP{args.step}/tb_logs"

    cfg_target.NAME = args.name
    cfg_target.DATASET = args.dataset
    cfg_target.TASK = args.task
    cfg_target.STEP = args.step

    cfg_target.DIST.RPN = args.rpn
    cfg_target.DIST.FEAT = args.feat
    if args.cls != -1:
        cfg_target.DIST.CLS = args.cls
    else:
        cfg_target.DIST.CLS = len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) / \
                              cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    cfg_target.INCREMENTAL = args.uce
    cfg_target.DIST.TYPE = args.dist_type

    cfg_source.freeze()
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

    old_classes = cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES
    new_classes = cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES
    excluded_classes = cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES
    num_classes = cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1
    logger_target = setup_logger("maskrcnn_benchmark_target_model", output_dir_target, get_rank())
    logger_target.info(f"All: {num_classes}, Old: {len(old_classes)}, New: {len(new_classes)}, Excluded: {len(excluded_classes)}")
    logger_target.info(args)
    logger_target.info("config yaml file for target model: {}".format(config_file))

    model = train(cfg_source, cfg_target, logger_target)
    test(cfg, cfg_target, model)


if __name__ == "__main__":
    main()
