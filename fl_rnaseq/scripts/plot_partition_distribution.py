"""Characterise how non-IID each partitioning configuration actually is.

Produces, per configuration, a client x class count matrix and a scalar
heterogeneity index (mean Jensen-Shannon divergence between each client's label
distribution and the global one). The scalar is what turns the alpha sweep into
a real experiment: accuracy can be plotted against measured heterogeneity as a
continuous curve rather than against three anecdotal alpha values.

    & "C:\\Program Files\\python311\\python.exe" -m scripts.plot_partition_distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fl_rnaseq.dataset import CLASSES, NUM_CLASSES, load_arrays
from fl_rnaseq.metrics import plot_client_label_heatmap
from fl_rnaseq.partitioning import (
    get_partitions, heterogeneity_index, label_distribution,
)
from fl_rnaseq.utils import run_slug, write_result

DEFAULT_DATA_DIR = "../gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531"

# (strategy, alpha) pairs. "dirichlet" entries need flwr-datasets installed.
DEFAULT_CONFIGS = [
    ("iid", None),
    ("dirichlet", 1.0),
    ("dirichlet", 0.5),
    ("dirichlet", 0.1),
    ("dirichlet-legacy", 0.5),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-partition-size", type=int, default=32)
    args = ap.parse_args()

    y_train = load_arrays(args.data_dir)["y_train"]
    out_dir = Path(args.results_dir) / "partitions"
    rows = []

    print(f"\n{'configuration':<28}{'JS':>8}  shard sizes")
    print("-" * 72)

    for strategy, alpha in DEFAULT_CONFIGS:
        slug = run_slug(strategy, alpha, args.seed)
        try:
            parts = get_partitions(
                strategy=strategy,
                y_train=y_train,
                num_partitions=args.num_clients,
                alpha=alpha if alpha is not None else 0.5,
                seed=args.seed,
                min_partition_size=args.min_partition_size,
            )
        except ImportError:
            print(f"{slug:<28}{'—':>8}  skipped (flwr-datasets not installed)")
            continue
        except ValueError as exc:
            # At alpha=0.1 with only 640 samples the partitioner may be unable to
            # satisfy min_partition_size. That infeasibility is itself a finding
            # about small-cohort federated medical data — record it, don't crash.
            print(f"{slug:<28}{'—':>8}  INFEASIBLE: {exc}")
            rows.append({
                "slug": slug, "strategy": strategy, "alpha": alpha,
                "status": "infeasible", "error": str(exc),
            })
            continue

        counts = label_distribution(parts, y_train, NUM_CLASSES)
        hetero = heterogeneity_index(counts)
        sizes = [int(len(p)) for p in parts]
        classes_seen = [int((counts[c] > 0).sum()) for c in range(args.num_clients)]

        print(f"{slug:<28}{hetero:>8.4f}  {sizes}  classes/client={classes_seen}")

        rows.append({
            "slug": slug,
            "strategy": strategy,
            "alpha": alpha,
            "status": "ok",
            "seed": args.seed,
            "heterogeneity_js": hetero,
            "shard_sizes": sizes,
            "classes_per_client": classes_seen,
            "label_counts": {
                f"client_{c}": dict(zip(CLASSES, counts[c].tolist()))
                for c in range(args.num_clients)
            },
        })

        plot_client_label_heatmap(
            counts, out_dir / f"{slug}.png",
            title=f"{slug}   (JS divergence {hetero:.3f})",
        )

    write_result({"partitions": rows}, out_dir, name="partitions.json")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        _plot_heterogeneity(ok, out_dir / "heterogeneity.png")
    print(f"\nWritten to {out_dir}")
    return 0


def _plot_heterogeneity(rows: list[dict], path: Path) -> None:
    """Measured heterogeneity per configuration, ordered least to most skewed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fl_rnaseq.metrics import GRID, INK, INK_MUTED

    rows = sorted(rows, key=lambda r: r["heterogeneity_js"])
    labels = [r["slug"].replace("_seed42", "") for r in rows]
    vals = [r["heterogeneity_js"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(rows) + 1.8))
    bars = ax.barh(np.arange(len(rows)), vals, height=0.6, color="#0072B2")
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8, color=INK)

    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xlabel("mean Jensen-Shannon divergence from the global label distribution",
                  color=INK, fontsize=9)
    ax.set_title("Measured heterogeneity by partitioning configuration",
                 color=INK, fontsize=11, pad=10)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
