# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .mask_head.mask_head import build_roi_mask_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}

        if self.training:
            x, detections, soft_res, loss_box, roi_align_features = self.box(features, proposals, targets)
            losses.update(loss_box)
        else:
            x, detections, results_background = self.box(features, proposals, targets)
            # return x, detections, results_background

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                    self.training
                    and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        if not self.training:
            return x, detections, results_background, []
        else:
            return x, detections, soft_res, losses, roi_align_features

    def calculate_soften_label(self, features, proposals, targets=None):
        soften_score, soften_bbox, proposal_features, roi_align_features = self.box.calculate_soften_label(features,
                                                                                                           proposals,
                                                                                                           targets)
        if self.cfg.MODEL.MASK_ON:
            mask_logits = self.mask.calculate_soften_label(proposal_features)
        else:
            mask_logits = None

        return soften_score, soften_bbox, mask_logits, roi_align_features

    def feature_distillation(self, features, proposals):
        # print('roi_heads.py | feature_distillation')
        class_logits = self.box.feature_distillation(features, proposals)
        return class_logits


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards

    print('roi_heads.py | build_roi_heads | in_channels: {0}'.format(in_channels))

    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
