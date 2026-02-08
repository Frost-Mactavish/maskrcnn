import argparse
import os

import torch

parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument("--path", "-p", default="", help="Weight path", type=str)
args = parser.parse_args()

args.save_path = f"{os.path.dirname(args.path)}/model_trimmed.pth"
print('pretrained model path: {}'.format(args.path))

pretrained_weights = torch.load(args.path, map_location='cpu')['model']
new_dict = {k: v for k, v in pretrained_weights.items()}

torch.save(new_dict, args.save_path)
print('saved to {}.'.format(args.save_path))
