# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x, roi_align_features = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result, results_background = self.post_processor((class_logits, box_regression), proposals)
            return x, result, results_background

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])

        return x, proposals, (class_logits, box_regression.view([-1, class_logits.size()[1], 4])), \
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg), roi_align_features

    def calculate_soften_label(self, features, proposals, targets=None):
        # Extract features that will be fed to the final classifier.
        # The feature_extractor generally corresponds to the pooler + heads
        x, roi_align_features = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        # print('box_head.py | calculate_soften_label | class_logits size : {0}, box_regression size : {1}'.format(class_logits.size(), box_regression.size()))
        # print('box_head.py | calculate_soften_label | class_logits : {0}, box_regression : {1}'.format(class_logits, box_regression))

        soften_scores = class_logits
        # print('box_head.py | calculate_soften_label | soften_score size: {0}'.format(soften_scores.size()))
        # print('box_head.py | calculate_soften_label | soften_score : {0}'.format(soften_scores))

        soften_bboxes = box_regression.view([-1, soften_scores.size()[1], 4])
        # print('box_head.py | calculate_soften_label | soften_bbox size: {0}'.format(soften_bboxes.size()))
        # print('box_head.py | calculate_soften_label | soften_bbox : {0}'.format(soften_bboxes))

        return soften_scores, soften_bboxes, x, roi_align_features

    def feature_distillation(self, features, proposals):
        # print('box_head.py | feature_distillation')
        x = self.feature_extractor.feature_distillation(features, proposals)
        class_logits, box_regression = self.predictor(x)
        # print('box_head.py | class_logits size: {0}'.format(class_logits.size()))
        # print('box_head.py | box_regression size: {0}'.format(box_regression.size()))
        return class_logits


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
