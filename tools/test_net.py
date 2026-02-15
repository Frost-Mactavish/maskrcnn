import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", "-c", default="", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = cfg.OUTPUT_DIR
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info(args)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    dataset_names = cfg.DATASETS.TEST
    output_folders = [cfg.OUTPUT_DIR] * len(dataset_names) if cfg.OUTPUT_DIR else [None] * len(dataset_names)
    data_loaders_val = make_data_loader(cfg, is_train=False)
    summary_writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR)
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
            alphabetical_order=cfg.TEST.COCO_ALPHABETICAL_ORDER,
            summary_writer=summary_writer,
            cfg=cfg
        )


if __name__ == "__main__":
    main()
