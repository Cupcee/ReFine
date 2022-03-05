import copy
import math
import numpy as np

import torch.nn as nn


class GLNN(nn.Module):
    def __init__(self, device, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        ).to(device)

    def forward(self, x):
        return self.layers(x)
