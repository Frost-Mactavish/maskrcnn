#!/bin/bash

#  Choose the corresponding first task #
# 19 classes in first task #
# task=19-1

# 15 classes in first task #
# task=15-5

# 10 classes in first task #
# task=10-10

# 5 classes in first task #
# task=5-15

task=$1

#### 1. Train the First Task ####
python tools/train_first_step.py -c configs/voc/${task}/e2e_faster_rcnn_R_50_C4_4x.yaml

#### 2. Prototype Box Selection (PBR) ####
step=0
name=ABR # The corrensponding name for the incremental method
python tools/prototype_box_selection.py \
    -cvd 0 \
    -n ${name} -t ${task} -s ${step} \
    -mb 2000 \
    -mt mean \
    -iss