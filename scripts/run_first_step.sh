#!/bin/bash

alias exp="python tools/train_first_step.py"

exp -c configs/DIOR/19-1/base.yaml
exp -c configs/DIOR/19-1/ft.yaml -ft

exp -c configs/DIOR/15-5/base.yaml
exp -c configs/DIOR/15-5/ft.yaml -ft

exp -c configs/DIOR/10-10/base.yaml
exp -c configs/DIOR/10-10/ft.yaml -ft

exp -c configs/DIOR/5-15/base.yaml
exp -c configs/DIOR/5-15/ft.yaml -ft

exp -c configs/DOTA/14-1/base.yaml
exp -c configs/DOTA/14-1/ft.yaml -ft

exp -c configs/DOTA/10-5/base.yaml
exp -c configs/DOTA/10-5/ft.yaml -ft

exp -c configs/DOTA/8-7/base.yaml
exp -c configs/DOTA/8-7/ft.yaml -ft

exp -c configs/DOTA/5-10/base.yaml
exp -c configs/DOTA/5-10/ft.yaml -ft