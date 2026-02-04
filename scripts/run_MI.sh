#!/bin/bash

alias exp="python tools/train_incremental.py -n ABR --feat ard --dist_type id -mt mean"
alias box="python tools/prototype_box_selection.py -n ABR -mt mean -iss"
shopt -s expand_aliases


# INCREMENTAL STEPS
alias exp_dior="exp -d DIOR -mb 2000"
alias box_dior="box -d DIOR -mb 2000"

task=10-5
for s in {1..2}; do
  exp_dior -t ${task} -s $s -gamma 1.0 -alpha 1.0 -beta 1.0
  box_dior -t ${task} -s $s
  echo Done
done

task=5-5
for s in {1..3}; do
  exp_dior -t ${task} -s $s -gamma 1.0 -alpha 0.5 -beta 1.0
  box_dior -t ${task} -s $s
  echo Done
done

task=10-2
for s in {1..5}; do
  exp_dior -t ${task} -s $s -gamma 0.5 -alpha 1.0 -beta 1.0
  box_dior -t ${task} -s $s
  echo Done
done

task=15-1
for s in {1..5}; do
  exp_dior -t ${task} -s $s -gamma 1.0 -alpha 1.0 -beta 1.0
  box_dior -t ${task} -s $s
  echo Done
done


alias exp_dota="exp -d DOTA"
alias box_dota="box -d DOTA"

task=10-1
for s in {1..5}; do
  exp_dota -t ${task} -s $s -mb 1000 -gamma 1.0 -alpha 1.0 -beta 1.0
  box_dota -t ${task} -s $s -mb 500
  echo Done
done

task=5-5
for s in {1..2}; do
  exp_dota -t ${task} -s $s -mb 500 -gamma 1.0 -alpha 0.5 -beta 1.0
  box_dota -t ${task} -s $s -mb 500
  echo Done
done