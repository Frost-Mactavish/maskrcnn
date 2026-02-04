#!/bin/bash

alias exp="python tools/train_incremental_finetune_all.py"
shopt -s expand_aliases

task=10-10
exp -t ${task} -n test --cls 0.15 -l 0.4 -high 0.7 -lw 1.0 -hw 0.3
