"""MLP model and train / test routines for RNA-seq classification."""

import torch
import torch.nn as nn

from fl_rnaseq.dataset import NUM_CLASSES, NUM_FEATURES


def _norm_layer(kind: str, dim: int) -> nn.Module:
    """Normalisation layer selector.

    ``"batch"`` is the default, but BatchNorm's running buffers are weight-averaged
    by FedAvg along with the parameters — including the int64 ``num_batches_tracked``
    counter — which is a known failure mode under non-IID data (cf. FedBN).
    ``"layer"`` keeps normalisation client-local and is the ablation against it.
    """
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm {kind!r}; expected 'batch', 'layer', or 'none'.")


class MLP(nn.Module):
    """Three-layer MLP with normalisation and Dropout for high-dimensional tabular data.

    20264 → 512 → 128 → 5 (cancer type)

    ``num_features`` is a constructor argument rather than a hard-wired constant so
    the gene-subset ablation is a one-line change. Note that `nn.LazyLinear` is not
    an option here: server_app builds ``ArrayRecord(MLP().state_dict())`` before any
    forward pass, and an uninitialised lazy layer has no materialised parameters.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
        norm: str = "batch",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 512),
            _norm_layer(norm, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            _norm_layer(norm, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    net: MLP,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train for `epochs` local epochs. Returns average cross-entropy loss."""
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    running_loss = 0.0
    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / (epochs * len(trainloader))


def test(
    net: MLP,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model. Returns (avg loss, accuracy)."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            logits = net(X)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(testloader), correct / total
