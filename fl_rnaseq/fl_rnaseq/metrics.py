"""Evaluation metrics and figures shared by every experiment.

`task.test()` returns only (loss, accuracy), which is not enough: with an
imbalance ratio of 3.85 and clients that may never see a class, accuracy hides
exactly the failure this project is trying to demonstrate. This module adds the
prediction primitive and macro-averaged reporting on top, without touching
`task.test()` — that signature is the federated contract.

Two details here are load-bearing and easy to get wrong:

  - ``confusion_matrix(..., labels=range(NUM_CLASSES))`` so a class a client
    never saw produces a zero row rather than a smaller matrix.
  - ``f1_score(..., zero_division=0)`` so a never-predicted class scores 0.0
    rather than raising or being silently dropped.

Get either wrong and non-IID collapse becomes invisible in the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from fl_rnaseq.dataset import CLASSES, NUM_CLASSES

# Categorical hues in fixed order, one per class — never cycled, never reassigned
# by rank. Validated for CVD separation (worst adjacent pair ΔE 11.0 deutan) and
# normal-vision separation (ΔE 16.4) against a light surface.
CLASS_COLORS = {
    "BRCA": "#0072B2",
    "COAD": "#E69F00",
    "KIRC": "#009E73",
    "LUAD": "#D55E00",
    "PRAD": "#CC79A7",
}
# Method colors for cross-experiment comparison figures, same fixed-order rule.
METHOD_COLORS = {
    "centralized": "#0072B2",
    "federated": "#009E73",
    "local_only": "#D55E00",
}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"


# ---------------------------------------------------------------------------
# Prediction and scoring
# ---------------------------------------------------------------------------

def predict(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (y_true, y_pred, y_prob)."""
    net.to(device)
    net.eval()
    trues, preds, probs = [], [], []

    with torch.no_grad():
        for X, y in loader:
            logits = net(X.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(y.numpy())

    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Full metric set for one evaluation, JSON-ready."""
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # balanced_accuracy_score builds its own confusion matrix without a `labels`
    # argument, so on a non-IID local validation set holding a single class it
    # warns about matrix shape and returns a degenerate value. Report None there
    # rather than a misleading number — every other metric already passes
    # labels=range(NUM_CLASSES) explicitly.
    n_classes_present = len(np.unique(y_true))
    balanced = (
        float(balanced_accuracy_score(y_true, y_pred)) if n_classes_present > 1 else None
    )

    report = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": balanced,
        "per_class": {
            CLASSES[c]: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in labels
        },
        # labels=... is what keeps this 5x5 even when the client never saw a class,
        # so an absent class shows as a zero row instead of silently shrinking the
        # matrix and shifting every other class's position.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes_present_in_truth": sorted({int(c) for c in np.unique(y_true)}),
        "classes_predicted": sorted({int(c) for c in np.unique(y_pred)}),
    }

    # AUROC needs every class represented in y_true; a non-IID local validation
    # set often has only two or three, in which case it is simply not defined.
    if y_prob is not None and len(np.unique(y_true)) == NUM_CLASSES:
        try:
            report["macro_auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            report["macro_auroc"] = None
    else:
        report["macro_auroc"] = None

    return report


def evaluate_model(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate and return the full report, plus a sample-weighted loss.

    `task.test()` averages cross-entropy per batch, which is slightly biased when
    the last batch is ragged. Both are reported; they differ only marginally.
    """
    y_true, y_pred, y_prob = predict(net, loader, device)
    report = classification_report_dict(y_true, y_pred, y_prob)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(net(X), y).item()
            total_n += y.size(0)
    report["loss"] = total_loss / max(total_n, 1)
    return report


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    """Recessive grid and axes so the data carries the ink."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def plot_confusion_matrix(
    cm: list | np.ndarray,
    path: str | Path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> Path:
    """Confusion matrix heatmap.

    Sequential magnitude, so a single hue light->dark — never a rainbow map.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(cm, dtype=float)
    # Color by per-class rate so rows are comparable despite class imbalance,
    # but annotate with raw counts: on this dataset most off-diagonal cells are
    # genuinely empty, and an integer "0" says that unambiguously where a
    # normalised "0.00" reads like a formatting bug.
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        cm = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    else:
        cm = counts

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else max(counts.max(), 1))

    ax.set_xticks(range(NUM_CLASSES), CLASSES)
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title(title, color=INK, fontsize=11, pad=10)

    # Direct-label every cell: a 5x5 grid is small enough that the numbers are
    # the point, and it doubles as the table view for the contrast requirement.
    # Counts on top, per-class rate underneath on the diagonal and on any cell
    # that actually captured errors.
    threshold = cm.max() * 0.6
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            n = int(counts[i, j])
            ink = "white" if cm[i, j] > threshold else INK
            ax.text(j, i - (0.12 if normalize and n else 0.0), str(n),
                    ha="center", va="center", fontsize=9, color=ink)
            if normalize and n:
                ax.text(j, i + 0.20, f"{cm[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color=ink, alpha=0.75)

    ax.set_xticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    # 2px surface gap between adjacent cells.
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, length=0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_curves(
    history: list[dict],
    keys: list[str],
    path: str | Path,
    x_key: str = "round",
    title: str = "",
    xlabel: str = "Round",
) -> Path:
    """Line chart of one or more metrics over rounds/epochs.

    One y-axis only — never a second scale. Metrics on different scales belong
    in separate figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = [h[x_key] for h in history]
    series_colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    for i, key in enumerate(keys):
        ys = [h.get(key) for h in history]
        ax.plot(
            x, ys, linewidth=2, color=series_colors[i % len(series_colors)],
            marker="o", markersize=4, label=key.replace("_", " "),
        )
        # Selective direct label at the final point — not a number on every point.
        if ys and ys[-1] is not None:
            ax.annotate(
                f"{ys[-1]:.3f}", (x[-1], ys[-1]), textcoords="offset points",
                xytext=(6, 0), va="center", fontsize=8, color=INK,
            )

    ax.set_xlabel(xlabel, color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=11, pad=10)
    _style_axes(ax)
    # A legend is always present for >= 2 series; a single series is named by the title.
    if len(keys) >= 2:
        leg = ax.legend(frameon=False, labelcolor=INK, fontsize=9)
        leg.set_title(None)
    elif not title:
        ax.set_ylabel(keys[0].replace("_", " "), color=INK)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_client_label_heatmap(
    counts: np.ndarray,
    path: str | Path,
    title: str = "Samples per class, by client",
) -> Path:
    """Client x class count matrix — the picture of how non-IID a split is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(counts)
    n_clients = counts.shape[0]

    fig, ax = plt.subplots(figsize=(5.6, 0.55 * n_clients + 2.0))
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(counts.max(), 1))

    ax.set_xticks(range(NUM_CLASSES), CLASSES)
    ax.set_yticks(range(n_clients), [f"client {i}" for i in range(n_clients)])
    ax.set_title(title, color=INK, fontsize=11, pad=10)

    threshold = counts.max() * 0.6
    for i in range(n_clients):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(int(counts[i, j])), ha="center", va="center", fontsize=8,
                color="white" if counts[i, j] > threshold else INK,
            )

    ax.set_xticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_clients, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, length=0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
