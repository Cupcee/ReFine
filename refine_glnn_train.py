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
from gnns import *
from glnns.glnn import GLNN
from datasets.ba3motif_dataset import BA3Motif
from torch_scatter import scatter
from torch_geometric.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ba3",
        choices=["mutag", "ba3", "graphsst2", "mnist", "vg", "reddit5k"],
    )
    parser.add_argument(
        "--result_dir", type=str, default="results/", help="Result directory."
    )
    parser.add_argument(
        "--lr", type=float, default=2 * 1e-4, help="Fine-tuning learning rate."
    )
    parser.add_argument("--epoch", type=int, default=20, help="Fine-tuning rpoch.")
    parser.add_argument("--simple", help="use Simplified ReFine", action="store_true")
    parser.add_argument("--no_relu", help="use ReFineNoReLU", action="store_true")
    parser.add_argument(
        "--random_seed",
        help="use model trained with random_seed",
        type=str,
        default=None,
    )
    return parser.parse_args()


args = parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

#test_dataset = BA3Motif(args.data_path, mode='testing')
#val_dataset = BA3Motif(args.data_path, mode='evaluation')
train_dataset = BA3Motif('data/BA3', mode='training')

#test_loader = DataLoader(test_dataset,
#                         batch_size=args.batch_size,
#                         shuffle=False
#                         )
#val_loader = DataLoader(val_dataset,
#                        batch_size=args.batch_size,
#                        shuffle=False
#                        )
train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True
                          )
model_path = f"param/gnns/{args.dataset}_net.pt"
model = torch.load(model_path, map_location=device)

in_features = torch.flatten(train_dataset[0].x, 1, -1).size(1)
mlp_model = GLNN(device=device, in_features=in_features, out_features=3)
loss_label = torch.nn.CrossEntropyLoss()
loss_teacher = torch.nn.KLDivLoss()
loss_lambda = 0.5
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)
for epoch in range(10):
    running_loss = 0.0
    it = 0
    for g in tqdm(iter(train_loader), total=len(train_loader)):
        g.to(device)
        with torch.no_grad():
            z = model(g.x, g.edge_index, g.edge_attr, g.batch)

        optimizer.zero_grad()
        mlp_in = scatter(g.x, g.batch, dim=0, dim_size=int(g.batch.max().item() + 1), reduce='add')
        y_hat = mlp_model(mlp_in)
        #print(f"g.x.shape: {g.x.shape}")
        #print(f"g.batch.shape: {g.batch.shape}")
        #print(f"g.y.shape: {g.y.shape}")
        #print(f"y_hat.shape: {y_hat.shape}")
        #print(f"z.shape: {z.shape}")
        #g.x.shape: torch.Size([2737, 4])
        #g.batch.shape: torch.Size([2737])
        #g.y.shape: torch.Size([128])
        #y_hat.shape: torch.Size([128, 3])
        #z.shape: torch.Size([128, 3])
        loss = loss_lambda * loss_label(y_hat, g.y) \
                + (1 - loss_lambda) * loss_teacher(y_hat, z)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if it % 500 == 499:
            print("Loss after mini-batch %5d: %.3f" % (it + 1, running_loss / 500))
            running_loss = 0.0
        it += 1

# os.makedirs(args.result_dir, exist_ok=True)
#
# with open(os.path.join(args.result_dir, f"{args.dataset}_results.json"), "w") as f:
#    json.dump(results, f, indent=4)
