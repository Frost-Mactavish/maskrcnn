import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss=None, bbox_loss=None, bbox_threshold=None):
    rpn_objectness_source, rpn_bbox_regression_source = rpn_output_source
    rpn_objectness_target, rpn_bbox_regression_target = rpn_output_target

    # calculate rpn classification loss
    num_source_rpn_objectness = len(rpn_objectness_source)
    num_target_rpn_objectness = len(rpn_objectness_target)
    final_rpn_cls_distillation_loss = []
    objectness_difference = []

    if num_source_rpn_objectness == num_target_rpn_objectness:
        for i in range(num_target_rpn_objectness):
            current_source_rpn_objectness = rpn_objectness_source[i]
            current_target_rpn_objectness = rpn_objectness_target[i]
            if cls_loss == 'filtered_l1':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_distillation_loss = torch.max(rpn_objectness_difference, filter)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'filtered_l2':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'normalized_filtered_l2':
                avrage_source_rpn_objectness = torch.mean(current_source_rpn_objectness)
                average_target_rpn_objectness = torch.mean(current_target_rpn_objectness)
                normalized_source_rpn_objectness = current_source_rpn_objectness - avrage_source_rpn_objectness
                normalized_target_rpn_objectness = current_target_rpn_objectness - average_target_rpn_objectness
                rpn_objectness_difference = normalized_source_rpn_objectness - normalized_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            elif cls_loss == 'masked_filtered_l2':
                source_mask = current_source_rpn_objectness.clone()
                source_mask[current_source_rpn_objectness >= 0.7] = 1  # rpn threshold for foreground
                source_mask[current_source_rpn_objectness < 0.7] = 0
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                masked_rpn_objectness_difference = rpn_objectness_difference * source_mask
                objectness_difference.append(masked_rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(masked_rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for rpn classification distillation")
    else:
        raise ValueError("Wrong rpn objectness output")
    final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss) / num_source_rpn_objectness
    # a = objectness_difference > 0

    # calculate rpn bounding box regression loss
    num_source_rpn_bbox = len(rpn_bbox_regression_source)
    num_target_rpn_bbox = len(rpn_bbox_regression_target)
    final_rpn_bbs_distillation_loss = []
    l2_loss = nn.MSELoss(size_average=False, reduce=False)

    if num_source_rpn_bbox == num_target_rpn_bbox:
        for i in range(num_target_rpn_bbox):
            current_source_rpn_bbox = rpn_bbox_regression_source[i]
            current_target_rpn_bbox = rpn_bbox_regression_target[i]
            current_objectness_difference = objectness_difference[i]
            [N, A, H,
             W] = current_objectness_difference.size()  # second dimention contains location shifting information for each anchor
            current_objectness_difference = permute_and_flatten(current_objectness_difference, N, A, 1, H, W)
            current_source_rpn_bbox = permute_and_flatten(current_source_rpn_bbox, N, A, 4, H, W)
            current_target_rpn_bbox = permute_and_flatten(current_target_rpn_bbox, N, A, 4, H, W)
            current_objectness_mask = current_objectness_difference.clone()
            current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
            current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
            masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
            masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
            if bbox_loss == 'l2':
                current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(
                    torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'l1':
                current_bbox_distillation_loss = torch.abs(masked_source_rpn_bbox - masked_source_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(
                    torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'None':
                final_rpn_bbs_distillation_loss.append(0)
            else:
                raise ValueError('Wrong loss function for rpn bounding box regression distillation')
    else:
        raise ValueError('Wrong RPN bounding box regression output')
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss) / num_source_rpn_bbox

    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    final_rpn_loss.to('cuda')

    return final_rpn_loss


def soften_proposales_iou_targets(soften_proposals, targets):
    # calculate the iou with gt and check out
    soften_proposals_indexes = []
    finetune_proposals_indexes = []
    for per_soften_proposal, per_target in zip(soften_proposals, targets):
        per_match_quality_matrix = boxlist_iou(per_target, per_soften_proposal).t()

        # find the index of proposals for the old classes (iou < 0.5 for all GT)
        per_iou_less_than_05_mask = per_match_quality_matrix <= 0.5
        per_iou_less_than_05_indices = torch.nonzero(per_iou_less_than_05_mask.all(dim=1)).squeeze(1)
        soften_proposals_indexes.append(per_iou_less_than_05_indices)

        # find the index of proposals for the new classes (iou > 0.5 for all GT)
        per_iou_greater_than_05_mask = per_match_quality_matrix > 0.5
        per_iou_greater_than_05_indices = torch.nonzero(per_iou_greater_than_05_mask.any(dim=1)).squeeze(1)
        finetune_proposals_indexes.append(per_iou_greater_than_05_indices)

    return soften_proposals_indexes, finetune_proposals_indexes


def calculate_roi_scores_distillation_losses_old_raw(soften_results, finetune_results, target_results):
    # soften: [num_proposal, 11],   finetune: [num_proposal, 11],   target: [num_proposal, 21]
    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    finetune_scores, finetune_bboxes = finetune_results

    num_of_distillation_categories = soften_scores.size()[1]  # [11]
    tot_classes = target_scores.size()[1]  # [21]

    soften_labels = torch.softmax(soften_scores, dim=1)
    finetune_labels = torch.softmax(finetune_scores, dim=1)
    modified_target_scores = F.log_softmax(target_scores[:, :], dim=1)

    soften_bg_probability = soften_labels[:, 0].unsqueeze(1)
    finetune_sums = finetune_labels.sum(dim=1, keepdim=True)
    scale_factors = soften_bg_probability / finetune_sums
    scaled_finetune_labels = finetune_labels * scale_factors

    distillation_labels = torch.cat(
        [torch.cat([scaled_finetune_labels[:, 0].unsqueeze(1), soften_labels[:, 1:]], dim=1),
         scaled_finetune_labels[:, 1:]], dim=1)
    class_distillation_loss = - distillation_labels * modified_target_scores
    class_distillation_loss_raw = torch.mean(class_distillation_loss, dim=1)  # average towards categories and proposals

    # compute distillation bbox loss
    modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox

    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
    bbox_distillation_loss_raw = torch.mean(torch.sum(bbox_distillation_loss, dim=2),
                                            dim=1)  # average towards categories and proposals

    return class_distillation_loss_raw, bbox_distillation_loss_raw


def calculate_roi_scores_distillation_losses_new_raw(soften_results, finetune_results, target_results):
    # soften: [num_proposal, 11],   finetune: [num_proposal, 11],   target: [num_proposal, 21]
    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    finetune_scores, finetune_bboxes = finetune_results

    num_of_distillation_categories = soften_scores.size()[1]  # [11]
    tot_classes = target_scores.size()[1]  # [21]

    soften_labels = torch.softmax(soften_scores, dim=1)
    finetune_labels = torch.softmax(finetune_scores, dim=1)
    modified_target_scores = F.log_softmax(target_scores[:, :], dim=1)

    finetune_bg_probability = finetune_labels[:, 0].unsqueeze(1)
    soften_sums = soften_labels.sum(dim=1, keepdim=True)
    scale_factors = finetune_bg_probability / soften_sums
    scaled_soften_labels = soften_labels * scale_factors

    distillation_labels = torch.cat([scaled_soften_labels, finetune_labels[:, 1:]], dim=1)
    class_distillation_loss = - distillation_labels * modified_target_scores
    class_distillation_loss_raw = torch.mean(class_distillation_loss, dim=1)  # average towards categories and proposals

    # compute distillation bbox loss
    modified_finetune_boxes = finetune_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, num_of_distillation_categories:, :]  # exclude background bbox

    l2_loss = nn.MSELoss(size_average=False, reduce=False)
    bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_finetune_boxes)
    bbox_distillation_loss_raw = torch.mean(torch.sum(bbox_distillation_loss, dim=2),
                                            dim=1)  # average towards categories and proposals

    return class_distillation_loss_raw, bbox_distillation_loss_raw
