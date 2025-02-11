import logging

from .voc_eval import do_voc_evaluation
from .voc_eval_inst import do_voc_evaluation_inst


def voc_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    return do_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )


def voc_evaluation_inst(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    return do_voc_evaluation_inst(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
