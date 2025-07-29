import torch.nn as nn
from src.custom_relu import MyReLUModule


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            MyReLUModule(),  # <‑‑ use the module wrapper
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # flatten NCHW -> N x 784
        return self.net(x.view(x.size(0), -1))
