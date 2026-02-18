#!/bin/bash

alias exp="python tools/train_incremental.py"

task=10-2
for s in {1..5}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
done

task=15-1
for s in {1..5}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
done

task=10-5
for s in {1..2}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
done

task=5-5
for s in {1..3}; do
  exp -t ${task} -n ILOD -s $s
  exp -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
done


alias exp_dota="exp -d DOTA"

task=10-1
for s in {1..5}; do
  exp_dota -t ${task} -n ILOD -s $s
  exp_dota -t ${task} -n FILOD -s $s --rpn --feat std --cls 1.
  exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1 -s $s
done

task=5-5
for s in {1..2}; do
  exp_dota -t ${task} -n ILOD -s $s
  exp_dota -t ${task} -n FILOD -s $s --rpn --feat std --cls 1. SOLVER.BASE_LR 0.0005
  exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5 -s $s
done