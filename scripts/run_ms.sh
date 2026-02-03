#!/bin/bash

alias exp="python tools/train_incremental.py"
shopt -s expand_aliases

# FIRST STEP
# python tools/train_first_step.py -c configs/OD_cfg/e2e_faster_rcnn_R_50_C4_4x.yaml

# INCREMENTAL STEPS
task=10-2
for s in {1..5}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
  echo Done
done

task=15-1
for s in {1..5}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
  echo Done
done

task=10-5
for s in 1 2; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
  echo Done
done

task=5-5
for s in {1..3}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
  echo Done
done


alias exp_dota="exp -d DOTA"

task=10-1
for s in {1..5}; do
  exp_dota -t ${task} -n ILOD -s $s
  exp_dota -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
  echo Done
done

task=5-5
for s in 1 2; do
  exp_dota -t ${task} -n ILOD -s $s
  exp_dota -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
  echo Done
done