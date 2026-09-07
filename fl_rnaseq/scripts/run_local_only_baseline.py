"""Local-only baseline — the lower bound federation must beat.

Each client trains a fresh MLP on *only its own shard*, with no aggregation and
no communication, then is scored on the same global test set the federated model
uses. This answers "what would a hospital get if it never joined the federation?"

Each client is evaluated twice, and the gap between the two is the argument for
federating: on its own local validation split a client holding mostly one tumour
type looks excellent, while on the global test set the same model is useless for
the classes it never saw.

Compute is matched to the federated run: 10 rounds x 3 local epochs = 30 epochs
over the same shard, so any difference is attributable to aggregation rather
than to training budget.

Needs no Flower installation:

    & "C:\\Program Files\\python311\\python.exe" -m scripts.run_local_only_baseline --strategy iid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fl_rnaseq.dataset import (
    CLASSES, NUM_CLASSES, load_arrays, load_data, load_test_data, split_fingerprint,
)
from fl_rnaseq.metrics import (
    evaluate_model, plot_client_label_heatmap, plot_confusion_matrix,
    INK, INK_MUTED, GRID,
)
from fl_rnaseq.partitioning import (
    get_partitions, heterogeneity_index, label_distribution,
)
from fl_rnaseq.task import MLP, train
from fl_rnaseq.utils import result_envelope, run_slug, set_seed, write_result

DEFAULT_DATA_DIR = "../gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="iid",
                    choices=["iid", "dirichlet", "dirichlet-legacy"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30,
                    help="matched to num-server-rounds x local-epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-partition-size", type=int, default=32)
    ap.add_argument("--self-balancing", action="store_true")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = None if args.strategy == "iid" else args.alpha
    slug = run_slug(args.strategy, alpha, args.seed)
    out_dir = Path(args.results_dir) / "local_only" / slug

    arrays = load_arrays(args.data_dir)
    fp = split_fingerprint(args.data_dir)
    test_loader = load_test_data(args.data_dir, args.batch_size)

    # Same partition the ClientApp will use, so the comparison is like-for-like.
    shards = get_partitions(
        strategy=args.strategy,
        y_train=arrays["y_train"],
        num_partitions=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        min_partition_size=args.min_partition_size,
        self_balancing=args.self_balancing,
    )
    counts = label_distribution(shards, arrays["y_train"], NUM_CLASSES)
    hetero = heterogeneity_index(counts)

    print(f"\nLocal-only baseline — {slug}")
    print(f"  split {fp['test_idx_sha1']}, {fp['n_test']} global test samples")
    print(f"  heterogeneity (mean JS divergence): {hetero:.4f}")
    print(f"\n  {'client':<9}{'n':>6}  classes seen           label counts")
    for cid in range(args.num_clients):
        seen = [CLASSES[c] for c in range(NUM_CLASSES) if counts[cid, c] > 0]
        print(
            f"  {cid:<9}{counts[cid].sum():>6}  {len(seen)}/5 {','.join(seen):<18}"
            f" {dict(zip(CLASSES, counts[cid].tolist()))}"
        )

    per_client, confusions = [], []
    print(f"\n  training {args.num_clients} isolated models, {args.epochs} epochs each...")

    for cid in range(args.num_clients):
        # Independent but deterministic init per client.
        set_seed(args.seed * 100 + cid)

        try:
            trainloader, valloader = load_data(
                partition_id=cid,
                num_partitions=args.num_clients,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                partition_strategy=args.strategy,
                dirichlet_alpha=args.alpha,
                seed=args.seed,
                min_partition_size=args.min_partition_size,
                self_balancing=args.self_balancing,
            )
        except ValueError as exc:
            # A degenerate shard is a finding, not a reason to abort the sweep.
            print(f"    client {cid}: SKIPPED — {exc}")
            per_client.append({"client_id": cid, "status": "degenerate_shard",
                               "error": str(exc)})
            continue

        model = MLP()
        t0 = time.time()
        train_loss = train(model, trainloader, args.epochs, args.lr, device)
        elapsed = time.time() - t0

        global_rep = evaluate_model(model, test_loader, device)
        local_rep = evaluate_model(model, valloader, device)
        confusions.append((cid, global_rep["confusion_matrix"]))

        n_classes_seen = int((counts[cid] > 0).sum())
        per_client.append({
            "client_id": cid,
            "status": "ok",
            "n_train": len(trainloader.dataset),
            "n_val": len(valloader.dataset),
            "n_classes_seen": n_classes_seen,
            "label_counts": dict(zip(CLASSES, counts[cid].tolist())),
            "train_loss": train_loss,
            "global": global_rep,
            "local_val": local_rep,
            "seconds": round(elapsed, 1),
        })

        print(
            f"    client {cid}: global acc={global_rep['accuracy']:.4f} "
            f"macroF1={global_rep['macro_f1']:.4f}  |  "
            f"local-val acc={local_rep['accuracy']:.4f}  "
            f"({n_classes_seen}/5 classes, {elapsed:.0f}s)"
        )

    ok = [c for c in per_client if c["status"] == "ok"]
    if not ok:
        print("\n  Every client had a degenerate shard — nothing to aggregate.")
        return 1

    def agg(path: str, key: str) -> dict:
        vals = [c[path][key] for c in ok]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    # Mean per-class recall across clients. COAD is the minority class (78 of 801),
    # so its recall is the most sensitive indicator of non-IID collapse: a client
    # that never saw COAD scores 0.0 here regardless of how good its accuracy looks.
    per_class_mean = {
        cls: {
            "recall": float(np.mean([c["global"]["per_class"][cls]["recall"] for c in ok])),
            "f1": float(np.mean([c["global"]["per_class"][cls]["f1"] for c in ok])),
            "support": ok[0]["global"]["per_class"][cls]["support"],
        }
        for cls in CLASSES
    }

    metrics = {
        "global_accuracy": agg("global", "accuracy"),
        "global_macro_f1": agg("global", "macro_f1"),
        "local_val_accuracy": agg("local_val", "accuracy"),
        "heterogeneity_js": hetero,
        "n_clients_ok": len(ok),
        "n_clients_degenerate": len(per_client) - len(ok),
        # The headline numbers for the write-up: what one isolated site achieves.
        "accuracy": agg("global", "accuracy")["mean"],
        "macro_f1": agg("global", "macro_f1")["mean"],
        "per_class": per_class_mean,
    }

    payload = result_envelope(
        experiment="local_only",
        slug=slug,
        config={
            "strategy": args.strategy, "alpha": alpha, "num_clients": args.num_clients,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "seed": args.seed, "min_partition_size": args.min_partition_size,
            "self_balancing": args.self_balancing, "data_dir": args.data_dir,
        },
        split_fingerprint=fp,
        model_info={"model": "MLP", "isolated": True},
        metrics=metrics,
        per_client=per_client,
    )
    write_result(payload, out_dir)

    plot_client_label_heatmap(
        counts, out_dir / "label_distribution.png",
        title=f"Training samples per class, by client ({slug})",
    )
    _plot_per_client(ok, out_dir / "per_client_performance.png", slug)
    for cid, cm in confusions:
        plot_confusion_matrix(
            cm, out_dir / f"confusion_client{cid}.png",
            title=f"Client {cid} alone, on global test set",
        )

    print("\n" + "=" * 68)
    print(f"  global accuracy   {metrics['global_accuracy']['mean']:.4f} "
          f"+/- {metrics['global_accuracy']['std']:.4f}  "
          f"[{metrics['global_accuracy']['min']:.4f}, {metrics['global_accuracy']['max']:.4f}]")
    print(f"  global macro-F1   {metrics['global_macro_f1']['mean']:.4f} "
          f"+/- {metrics['global_macro_f1']['std']:.4f}")
    print(f"  local-val acc     {metrics['local_val_accuracy']['mean']:.4f} "
          f"+/- {metrics['local_val_accuracy']['std']:.4f}   <- what a site sees locally")
    print("=" * 68)
    gap = metrics["local_val_accuracy"]["mean"] - metrics["global_accuracy"]["mean"]
    print(f"\n  Local-vs-global optimism gap: {gap:+.4f}")
    print(f"  Results written to {out_dir}")
    return 0


def _plot_per_client(ok: list[dict], path: Path, slug: str) -> None:
    """Grouped bars: accuracy and macro-F1 per client, on the global test set.

    Plotting both is the point — where they diverge, the client is scoring well
    on the majority class while failing entirely on classes it never saw.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [c["client_id"] for c in ok]
    acc = [c["global"]["accuracy"] for c in ok]
    f1 = [c["global"]["macro_f1"] for c in ok]
    local = [c["local_val"]["accuracy"] for c in ok]

    x = np.arange(len(ids))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset, vals, color, label in (
        (-width, local, "#E69F00", "local-val accuracy"),
        (0.0, acc, "#0072B2", "global accuracy"),
        (width, f1, "#D55E00", "global macro-F1"),
    ):
        bars = ax.bar(x + offset, vals, width * 0.92, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=7.5, color=INK)

    ax.set_xticks(x, [f"client {i}\n({c['n_classes_seen']}/5 classes)" for i, c in zip(ids, ok)])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score", color=INK)
    ax.set_title(f"Isolated clients — {slug}", color=INK, fontsize=11, pad=10)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
