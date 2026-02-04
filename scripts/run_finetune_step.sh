#!/bin/bash

# SECONDE STEP

# sec 10
python tools/train_first_step.py -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_sec10.yaml


# sec 5
python tools/train_first_step.py -c configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_sec5.yaml
