#src/model.py

import torch.nn as nn

# Construct the network
def make_mlp(in_dim=10, h1=32, h2=16, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h2, 1)
    )