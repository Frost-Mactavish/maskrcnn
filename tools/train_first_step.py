import argparse
import os

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def train(cfg):
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # create a parameter dictionary and initialize the iteration number to 0
    arguments = {}
    arguments["iteration"] = 0

    # path to store the trained parameter value
    output_dir = cfg.OUTPUT_DIR

    # create check pointer
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, True)

    # load the pre-trained model parameter to current model
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

    # dict updating method to update the parameter dictionary
    arguments.update(extra_checkpoint_data)

    # load training data
    # type of data_loader is list, type of its inside elements is torch.utils.data.DataLoader
    # When is_train=True, make sure cfg.DATASETS.TRAIN is a list
    # it has to point to one or multiple annotation files
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # number of iteration to store parameter value in pth file

    # train the model: call function ./maskrcnn_benchmark/engine/trainer.py do_train() function
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model):
    torch.cuda.empty_cache()

    iou_types = ("bbox",)

    # according to number of test dataset decides number of output folder
    output_folders = [None] * len(cfg.DATASETS.TEST)

    # create folder to store test result
    # output result is stored in: cfg.OUTPUT_DIR/inference/cfg.DATASETS.TEST
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    # load testing data
    data_loaders_val = make_data_loader(
        cfg,
        is_train=False,
    )

    # test
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/data/my_code/filod/configs/e2e_faster_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir)
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))

    # open and read the input yaml file, store it on config_str and display on the screen
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg)

    if not args.skip_test:
        run_test(cfg, model)


if __name__ == "__main__":
    main()
