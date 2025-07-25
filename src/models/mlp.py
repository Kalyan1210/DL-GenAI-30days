import torch.nn as nn


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 256, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # flatten NCHW -> N x 784
        return self.net(x.view(x.size(0), -1))
