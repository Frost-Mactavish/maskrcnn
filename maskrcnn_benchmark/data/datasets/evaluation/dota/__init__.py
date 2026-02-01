import logging

from .dota_eval import do_dota_evaluation

def dota_evaluation(dataset, predictions, output_folder, box_only, file_list, cfg, **_):
    logger = logging.getLogger("maskrcnn_benchmark_target_model.inference")
    if box_only:
        logger.warning("dota evaluation doesn't support box_only, ignored.")
    logger.info("performing dota evaluation, ignored iou_types.")
    return do_dota_evaluation(
        dataset=dataset,
        predictions=predictions,
        file_list=file_list,
        output_folder=output_folder,
        logger=logger,
        cfg=cfg
    )