#!/bin/bash

alias exp="python tools/train_incremental.py -n BPF -il 0.4 -ih 0.7 -lw 1.0 -hw 0.3"
alias ft="python tools/train_first_step.py -fs 1"

ft -c configs/DIOR/19-1/ft.yaml
exp -d DIOR -t 19-1 --cls 1.

ft -c configs/DIOR/15-5/ft.yaml
exp -d DIOR -t 15-5 --cls 0.5

ft -c configs/DIOR/10-10/ft.yaml
exp -d DIOR -t 10-10 --cls 0.1

ft -c configs/DIOR/5-15/ft.yaml
exp -d DIOR -t 5-15 --cls 0.1

ft -c configs/DOTA/14-1/ft.yaml
exp -d DOTA -t 14-1 --cls 1.

ft -c configs/DOTA/10-5/ft.yaml
exp -d DOTA -t 10-5 --cls 0.5

ft -c configs/DOTA/8-7/ft.yaml
exp -d DOTA -t 8-7 --cls 0.1

ft -c configs/DOTA/5-10/ft.yaml
exp -d DOTA -t 5-10 --cls 0.1