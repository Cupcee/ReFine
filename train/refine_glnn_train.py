import os
import time
import random
import json
import argparse

import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

from utils.dataset import get_datasets
from explainers import *
from torch_geometric.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--dataset', type=str, default='ba3',
                        choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])
    parser.add_argument('--result_dir', type=str, default="results/",
                        help='Result directory.')
    parser.add_argument('--lr', type=float, default=2*1e-4,
                        help='Fine-tuning learning rate.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Fine-tuning rpoch.')
    parser.add_argument('--simple', help='use Simplified ReFine', action='store_true')
    parser.add_argument('--no_relu', help='use ReFineNoReLU', action='store_true')
    parser.add_argument('--random_seed', help='use model trained with random_seed', type=str, default=None)
    return parser.parse_args()

args = parse_args()
results = {}
if args.dataset == 'ba3':
    ground_truth = True
else:
    ground_truth = False

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)

graph_mask_path = f'param/filtered/{args.dataset}_idx_test.pt'
print(f"Loading graph mask from {graph_mask_path}")
graph_mask = torch.load(graph_mask_path)
test_loader = DataLoader(test_dataset[graph_mask], batch_size=1, shuffle=False, drop_last=False)
ratios = [0.1 *i for i in range(1,11)]

refine_model_path = f'param/refine/{args.dataset}.pt'
print(f"Loading model from {refine_model_path}")
refine = torch.load(refine_model_path)
refine.remap_device(device)

#---------------------------------------------------
print("Evaluate ReFine w.r.t. ACC-AUC...")
for i, r in enumerate(ratios):
    acc_logger = []
    for g in tqdm(iter(test_loader), total=len(test_loader)):
        g.to(device)
        y = refine.explain_graph(g, fine_tune=True, ratio=r, lr=args.lr, epoch=args.epoch)
        print(y)
        raise Exception("lol")

os.makedirs(args.result_dir, exist_ok=True)

with open(os.path.join(args.result_dir, f"{args.dataset}_results.json"), "w") as f:
    json.dump(results, f, indent=4)
