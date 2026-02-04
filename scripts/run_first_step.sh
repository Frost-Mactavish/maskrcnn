#!/bin/bash

# FIRST STEP

# first 10
python tools/train_first_step.py -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml
sleep 5
python tools/trim_detectron_model.py --name "10-10/LR005_BS4_BF_att1_rpn1"


# first 15
python tools/train_first_step.py -c configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml
sleep 5
python tools/trim_detectron_model.py --name "15-5/LR005_BS4_BF_att1_rpn1"

