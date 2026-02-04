import argparse
import datetime
import logging
import os
import time
import warnings

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.build import make_bbox_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tools.extract_memory import Mem
warnings.filterwarnings("ignore", category=UserWarning)


def extract_bboxes_and_features(model_source, data_loader, device, cfg):
    old_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES
    new_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES

    logger = logging.getLogger("maskrcnn_benchmark_last_model.trainer")
    logger.info("Start sampling")
    meters = MetricLogger(delimiter="  ")

    #################################################################
    # extract the feature maps for each boxes per classes in images #
    #################################################################

    # Start extracting!
    max_iter = len(data_loader)
    data_iter = iter(data_loader)
    model_source.eval()
    start_training_time = time.time()
    end = time.time()

    all_bboxes_info = [[] for _ in range(len(new_classes))]

    for i in range(len(data_loader)):
        try:
            images, targets, original_targets, idx = next(data_iter)
        except StopIteration:
            break
        # print("Current load data is {0}/{1}".format(i, len(data_loader)))
        data_time = time.time() - end

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # extract the features for each rois
        with torch.no_grad():
            (target_scores, _), _, _, _, roi_align_features = \
                model_source.generate_feature_logits_by_targets(images, targets)

        target_scores.tolist()  # [9, 16]
        roi_align_features = torch.mean(roi_align_features.cpu(), dim=1).tolist()  # [9, 1024, 7, 7]

        # print(target_scores.shape)
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        bbox_index = 0

        for img_n in range(len(idx)):

            target = original_targets[img_n]

            ######## visulation ########
            # img_id = ids_[img_n][0]
            # img = Image.open("data/voc07/VOCdevkit/VOC2007/JPEGImages/{0}.jpg".format(img_id)).convert("RGB")
            # from PIL import ImageDraw
            # from PIL import ImageFont
            # a = ImageDraw.ImageDraw(img)
            # ttf = ImageFont.load_default()
            # for g in range(target.__len__()):
            #     gt_ = target.bbox.tolist()[g]
            #     label_ = target.extra_fields['labels'].tolist()[g]
            #     a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline='blue', width=4)
            #     a.text((gt_[0]+5, gt_[1]+6), str(label_), font=ttf, fill=(0,0,255))
            # img.save('output/box_rehearsal/current_image_{}.jpg'.format(img_id))

            for ind in range(target.__len__()):
                bboxes = target.bbox[ind].cpu().tolist()

                bbox_index += 1
                # Delete too small boxes.
                if (bboxes[2] - bboxes[0]) <= 70 and (bboxes[3] - bboxes[1]) <= 70:
                    continue
                else:
                    # print(len(roi_align_features))
                    # print(bbox_index-1)
                    all_bboxes_info[target.extra_fields["labels"][ind].item() - len(old_classes) - 1].append(
                        {'feature': roi_align_features[bbox_index - 1],  # [1024, 7, 7]
                         'logits': target_scores[img_n + ind].cpu(),  # [16]
                         'image_path': idx[img_n],  # list
                         'box_class': target.extra_fields["labels"][ind].cpu().item(),  # []
                         'box': bboxes,  # []
                         'mode': target.mode})  # str

    # Saving the old bbox and image information through the frozen model
    # Under the default continuous learning settings, all information should not be saved and is only used for debugging！！
    # with open(old_bbox_information_file, 'wb') as obf:
    #     pickle.dump(all_bboxes_info, obf, pickle.HIGHEST_PROTOCOL)
    #     print("The old_bbox_information size is: {}".format(len(all_bboxes_info)))
    #     print("All_information is saved in: {}".format(old_bbox_information_file))

    # Display the total used training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total sampling time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    return all_bboxes_info


def selector(cfg_source):
    current_mem_file = f"{cfg_source.MEM_TYPE}_{cfg_source.MEM_BUFF}"

    # creat or load the memory file
    # if cfg_source.STEP == 0:
    #     # Creat a memory buffer for first task (first task is the same for different setting)
    #     current_mem_path = os.path.split(cfg_source.MODEL.SOURCE_WEIGHT)[0] + current_mem_file
    #     if not os.path.exists(current_mem_path):
    #         os.mkdir(current_mem_path)
    # else:
    #     # Create the corresponding memory buffer for incremental steps
    current_mem_path = os.path.join(cfg_source.OUTPUT_DIR, current_mem_file)
    if not os.path.exists(current_mem_path):
        os.mkdir(current_mem_path)

    print('-- PBS REPORT-- The current Box Reahersal path is {0}'.format(current_mem_path))

    # Select the prototype box image
    num_file_classes = len(os.listdir(current_mem_path))

    if cfg_source.STEP == 0 and num_file_classes >= int(cfg_source.MEM_BUFF):
        # The prototype boxes have exsisted for current step!
        print("The prototype box images for first step have existed!!")
        all_bboxes_info = None
    else:  # Update the prototype boxes for current classes
        model_source = build_detection_model(cfg_source)
        device = torch.device(cfg_source.MODEL.DEVICE)
        model_source.to(device)

        arguments_source = {}
        arguments_source["iteration"] = 0

        output_dir_source = cfg_source.OUTPUT_DIR + f"STEP{cfg_source.STEP}"

        checkpointer_source = DetectronCheckpointer(cfg_source, model_source, save_dir=output_dir_source)
        extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)
        arguments_source.update(extra_checkpoint_data_source)

        bbox_loader = make_bbox_loader(cfg_source, is_train=False, rank=get_rank())

        # get the memory from the model
        all_bboxes_info = extract_bboxes_and_features(model_source, bbox_loader, device, cfg_source)

    ##############################################################################
    # create or update memory
    ##############################################################################

    Exemplar = Mem(cfg_source, cfg_source.STEP, current_mem_path)
    Exemplar.update_memory(all_bboxes_info)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-n", "--name",
        default="ABR",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="DIOR"
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="10-10"
    )
    parser.add_argument(
        "-s", "--step",
        default=0, type=int,
    )
    parser.add_argument(
        "-mb", "--memory_buffer",
        default=2000, type=int,
    )
    parser.add_argument(
        "-mt", "--memory_type",
        default="mean", type=str,
        choices=['random', 'herding', 'mean']
    )
    parser.add_argument(
        "-iss", "--is_sample",
        default=True,
        action='store_true',
    )

    args = parser.parse_args()

    if args.step == 0:
        source_model_config_file = f"configs/{args.dataset}/{args.task}/base.yaml"
    else:
        source_model_config_file = f"configs/{args.dataset}/{args.task}/RB_target.yaml"

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(source_model_config_file)

    base = 'log'
    if args.step == 0:
        cfg_source.MODEL.SOURCE_WEIGHT = f"{base}/{args.dataset}/{args.task}/{args.name}/STEP{args.step}/model_trimmed.pth"
        cfg_source.MODEL.WEIGHT = cfg_source.MODEL.SOURCE_WEIGHT
    elif args.step >= 1:
        cfg_source.MODEL.WEIGHT = f"{base}/{args.dataset}/{args.task}/{args.name}/STEP{args.step}/model_trimmed.pth"
        cfg_source.OUTPUT_DIR = f"{base}/{args.dataset}/{args.task}/{args.name}"
        if cfg_source.OUTPUT_DIR:
            mkdir(cfg_source.OUTPUT_DIR)

    # setting the output head 
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES += args.step * cfg_source.CLS_PER_STEP
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES += cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
            :(args.step - 1) * cfg_source.CLS_PER_STEP]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
            args.step * cfg_source.CLS_PER_STEP:]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES)
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES = cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[(args.step - 1) * cfg_source.CLS_PER_STEP: args.step * cfg_source.CLS_PER_STEP]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
    else:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES) + 1

    cfg_source.TASK = args.task
    cfg_source.STEP = args.step
    cfg_source.NAME = args.name
    cfg_source.IS_FATHER = False
    cfg_source.IS_SAMPLE = args.is_sample
    cfg_source.TEST.IMS_PER_BATCH = 8
    cfg_source.DATASET = args.dataset

    cfg_source.MEM_BUFF = args.memory_buffer
    cfg_source.MEM_TYPE = args.memory_type
    cfg_source.freeze()

    # use current model to select the prototype boxes
    selector(cfg_source)


if __name__ == "__main__":
    main()
