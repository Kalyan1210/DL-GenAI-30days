import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.mlp import TwoLayerMLP

device = (
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)


def get_data(batch):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(test_ds, batch_size=batch),
    )


def train(args):
    train_dl, test_dl = get_data(args.batch)
    model = TwoLayerMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # quick val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                preds = model(xb.to(device)).argmax(1)
                correct += (preds.cpu() == yb).sum().item()
                total += yb.size(0)
        acc = 100 * correct / total
        print(f"Epoch {epoch}: val accuracy {acc:.2f}%")

    torch.save(model.state_dict(), "mlp_mnist.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
