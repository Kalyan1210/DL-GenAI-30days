#!/usr/bin/env python
"""
Smith LR-range test.
Saves lr_finder.png in the working directory.
Run:  python scripts/lr_finder.py
"""
import math
import torch
import src.models.mlp as mlp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.mlp import TwoLayerMLP

BATCH = 256
LR_MIN = 1e-5
LR_MAX = 1.0
STEPS = 100  # one epoch on MNIST ≈ 235, so 100 is enough

device = "cpu"


def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = TwoLayerMLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR_MIN)
    loss_fn = torch.nn.CrossEntropyLoss()

    lrs, losses, best_loss = [], [], float("inf")
    mult = (LR_MAX / LR_MIN) ** (1 / (STEPS - 1))

    iterator = iter(dl)
    for step in tqdm(range(STEPS)):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(dl)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()

        # Record & update LR
        lr = LR_MIN * mult**step
        for g in opt.param_groups:
            g["lr"] = lr

        lrs.append(lr)
        losses.append(loss.item())
        best_loss = min(best_loss, loss.item())

        if loss.item() > 4 * best_loss:
            break

    # Plot
    plt.semilogx(lrs, losses)
    plt.xlabel("Learning rate")
    plt.ylabel("Smoothed loss")
    plt.title("LR range test")
    plt.savefig("lr_finder.png", dpi=120)
    print("Saved plot as lr_finder.png")


if __name__ == "__main__":
    main()
