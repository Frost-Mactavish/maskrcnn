#!/bin/bash

alias base="python tools/train_first_step.py"

base -d DIOR -c configs/DIOR/19-1/base.yaml
base -d DIOR -c configs/DIOR/15-5/base.yaml
base -d DIOR -c configs/DIOR/10-10/base.yaml
base -d DIOR -c configs/DIOR/5-15/base.yaml

base -d DOTA -c configs/DOTA/14-1/base.yaml
base -d DOTA -c configs/DOTA/10-5/base.yaml
base -d DOTA -c configs/DOTA/8-7/base.yaml
base -d DOTA -c configs/DOTA/5-10/base.yaml


alias box="python tools/prototype_box_selection.py -n ABR -s 0 -mt mean -iss"

box -d DIOR -mb 2000 -t 19-1
box -d DIOR -mb 2000 -t 15-5
box -d DIOR -mb 2000 -t 10-10
box -d DIOR -mb 2000 -t 5-15

box -d DOTA -mb 1000 -t 14-1
box -d DOTA -mb 1000 -t 10-5
box -d DOTA -mb 1000 -t 8-7
box -d DOTA -mb 600 -t 5-10