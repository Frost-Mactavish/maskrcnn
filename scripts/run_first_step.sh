#!/bin/bash

alias base="python tools/train_first_step.py"

base -c configs/DIOR/19-1/base.yaml
base -c configs/DIOR/15-5/base.yaml
base -c configs/DIOR/10-10/base.yaml
base -c configs/DIOR/5-15/base.yaml

base -c configs/DOTA/14-1/base.yaml
base -c configs/DOTA/10-5/base.yaml
base -c configs/DOTA/8-7/base.yaml
base -c configs/DOTA/5-10/base.yaml