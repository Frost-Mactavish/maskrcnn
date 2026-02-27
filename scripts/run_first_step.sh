#!/bin/bash

alias exp="python tools/train_first_step.py"

exp -c configs/DIOR/19-1/base.yaml
exp -c configs/DIOR/15-5/base.yaml
exp -c configs/DIOR/10-10/base.yaml
exp -c configs/DIOR/5-15/base.yaml

exp -c configs/DOTA/14-1/base.yaml
exp -c configs/DOTA/10-5/base.yaml
exp -c configs/DOTA/8-7/base.yaml
exp -c configs/DOTA/5-10/base.yaml