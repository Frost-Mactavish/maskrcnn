#!/bin/bash

alias exp="python tools/train_incremental.py -n BPF -il 0.4 -ih 0.7 -lw 1.0 -hw 0.3"

exp -d DIOR -t 19-1 --cls 1.
exp -d DIOR -t 15-5 --cls 0.5
exp -d DIOR -t 10-10 --cls 0.1
exp -d DIOR -t 5-15 --cls 0.1

exp -d DOTA -t 14-1 --cls 1.
exp -d DOTA -t 10-5 --cls 0.5
exp -d DOTA -t 8-7 --cls 0.1
exp -d DOTA -t 5-10 --cls 0.1