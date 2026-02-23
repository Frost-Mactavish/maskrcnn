#!/bin/bash

alias exp="python tools/train_incremental.py -n BPF -il 0.4 -ih 0.7 -lw 1.0 -hw 0.3"
alias ft="python tools/train_first_step.py -ft"

task=10-5
for s in {1..2}; do
    ft -fs $s -c configs/DIOR/${task}/ft.yaml
    exp -d DIOR -t ${task} -s $s --cls 0.5
done

task=5-5
for s in {1..3}; do
    ft -fs $s -c configs/DIOR/${task}/ft.yaml
    exp -d DIOR -t ${task} -s $s --cls 0.5
done

task=10-2
for s in {1..5}; do
    ft -fs $s -c configs/DIOR/${task}/ft.yaml
    exp -d DIOR -t ${task} -s $s --cls 1.
done

task=15-1
for s in {1..5}; do
    ft -fs $s -c configs/DIOR/${task}/ft.yaml
    exp -d DIOR -t ${task} -s $s --cls 1.
done

task=10-1
for s in {1..5}; do
    ft -fs $s -c configs/DOTA/${task}/ft.yaml
    exp -d DOTA -t ${task} -s $s --cls 1.
done

task=5-5
for s in {1..2}; do
    ft -fs $s -c configs/DOTA/${task}/ft.yaml
    exp -d DOTA -t ${task} -s $s --cls 0.5
done