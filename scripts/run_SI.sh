#!/bin/bash

alias exp="python tools/train_incremental.py"

task=19-1
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --rpn --feat std --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

task=15-5
exp -t ${task} -n ILOD
exp -t ${task} -n FILOD --feat std --rpn --cls 1.
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5

task=10-10
exp -t ${task} -n ILOD SOLVER.BASE_LR 0.001
exp -t ${task} -n FILOD --feat std --rpn --cls 1. SOLVER.BASE_LR 0.001
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1

task=5-15
exp -t ${task} -n ILOD SOLVER.BASE_LR 0.001
exp -t ${task} -n FILOD --feat std --rpn --cls 1. SOLVER.BASE_LR 0.001
exp -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1


alias exp_dota="exp -d DOTA"
task=14-1
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --rpn --feat std --cls 1.
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 1

task=10-5
exp_dota -t ${task} -n ILOD
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1. SOLVER.BASE_LR 0.0005
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.5

task=8-7
exp_dota -t ${task} -n ILOD SOLVER.BASE_LR 0.001
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1. SOLVER.BASE_LR 0.001
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1

task=5-10
exp_dota -t ${task} -n ILOD SOLVER.BASE_LR 0.001
exp_dota -t ${task} -n FILOD --feat std --rpn --cls 1. SOLVER.BASE_LR 0.001
exp_dota -t ${task} -n MMA --rpn --uce --dist_type uce --cls 0.1