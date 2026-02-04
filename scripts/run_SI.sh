#!/bin/bash

alias exp="python tools/train_incremental.py -n ABR -s 1 --feat ard --dist_type id -mt mean"
shopt -s expand_aliases

alias exp_dior="exp -d DIOR"
exp_dior -t 19-1 -gamma 5.0 -alpha 1.0 -beta 1.0 -mb 2000
exp_dior -t 15-5 -gamma 1.0 -alpha 0.5 -beta 1.0 -mb 2000
exp_dior -t 10-10 -gamma 1.0 -alpha 0.1 -beta 0.5 -mb 2000
exp_dior -t 5-15 -gamma 1.0 -alpha 0.1 -beta 0.5 -mb 2000

alias exp_dota="exp -d DOTA"
exp_dota -t 14-1 -gamma 5.0 -alpha 1.0 -beta 1.0 -mb 1000
exp_dota -t 10-5 -gamma 1.0 -alpha 0.5 -beta 1.0 -mb 1000
exp_dota -t 8-7 -gamma 1.0 -alpha 0.1 -beta 0.5 -mb 1000
exp_dota -t 5-10 -gamma 1.0 -alpha 0.1 -beta 0.5 -mb 500