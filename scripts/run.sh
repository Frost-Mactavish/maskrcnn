#!/bin/bash

alias exp="python tools/train_incremental.py"
shopt -s expand_aliases

# FIRST STEP
# python tools/train_first_step.py -c configs/OD_cfg/e2e_faster_rcnn_R_50_C4_4x.yaml

# INCREMENTAL STEPS
task=19-1
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --rpn --feat std --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

task=15-5
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5

task=10-10
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1

task=5-15
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1


alias exp_dota="exp -d DOTA"
task=14-1
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --rpn --feat std --cls 1.
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

task=10-5
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1.
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5

task=8-7
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1.
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1

task=5-10
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1.
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1