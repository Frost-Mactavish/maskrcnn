#!/bin/bash

# FIRST STEP

# first 10
python tools/train_first_step.py -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml
sleep 5
python tools/trim_detectron_model.py --name "10-10/LR005_BS4_BF_att1_rpn1"


# SECONDE STEP
# sec 10
python tools/train_first_step.py -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_sec10.yaml


# THIRD STEP
# 10-10
alias exp="python tools/train_incremental_finetune_all.py"
shopt -s expand_aliases

task=10-10
exp -t ${task} -n test --cls 0.15 -l 0.4 -high 0.7 -lw 1.0 -hw 0.3
