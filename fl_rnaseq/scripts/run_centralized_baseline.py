"""Centralized baseline — the upper bound the federated model is measured against.

Trains on the *pooled* training set (as if every client had shared its data) and
evaluates on the same held-out test set the federated run uses. Three models:

  1. L2-regularized multinomial logistic regression, C tuned by 5-fold CV.
     The textbook estimator for p >> n (20264 genes, 640 samples) and the
     standard reference in the RNA-seq literature.
  2. L1 / saga variant — yields a sparse gene signature, the one place this
     project can say something biological.
  3. The same MLP the federation trains, on pooled data. This separates "cost of
     federation" from "cost of model choice": if the centralized MLP already
     underperforms logistic regression, the architecture is the weak link, not
     FedAvg.

Needs no Flower installation. Run with the system interpreter:

    & "C:\\Program Files\\python311\\python.exe" -m scripts.run_centralized_baseline
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from fl_rnaseq.dataset import (
    CLASSES, NUM_CLASSES, NUM_FEATURES, load_arrays, split_fingerprint,
)
from fl_rnaseq.metrics import (
    classification_report_dict, evaluate_model, plot_confusion_matrix,
    plot_metric_curves, INK, INK_MUTED, GRID,
)
from fl_rnaseq.task import MLP, train
from fl_rnaseq.utils import (
    communication_cost, result_envelope, set_seed, write_result,
)

DEFAULT_DATA_DIR = "../gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531"


# ---------------------------------------------------------------------------
# 1. Logistic regression
# ---------------------------------------------------------------------------

def run_logreg(arrays: dict, out_dir: Path, seed: int, fp: dict, data_dir: str) -> dict:
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    # `multi_class` is deliberately omitted: it is deprecated from sklearn 1.5,
    # and lbfgs already uses the multinomial (softmax) formulation for 5 classes.
    grid = GridSearchCV(
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000),
        {"C": np.logspace(-4, 4, 9)},
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=True,
    )

    print("  fitting L2 logistic regression (9 values of C x 5 folds)...")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        grid.fit(X_train, y_train)
        n_convergence_warnings = sum(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
    elapsed = time.time() - t0

    clf = grid.best_estimator_
    assert clf.classes_.tolist() == list(range(NUM_CLASSES)), "unexpected class order"

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    report = classification_report_dict(y_test, y_pred, y_prob)

    n_iter = int(np.max(clf.n_iter_))
    print(
        f"  best C = {grid.best_params_['C']:.4g}  "
        f"CV macro-F1 = {grid.best_score_:.4f}  "
        f"test acc = {report['accuracy']:.4f}  macro-F1 = {report['macro_f1']:.4f}  "
        f"({elapsed:.1f}s, n_iter={n_iter})"
    )
    if n_iter >= 5000:
        print("  WARNING: hit max_iter — the reported model may not have converged.")

    cv = grid.cv_results_
    _plot_cv_curve(cv, out_dir / "cv_curve.png", grid.best_params_["C"])
    plot_confusion_matrix(
        report["confusion_matrix"], out_dir / "confusion_matrix.png",
        title="Centralized L2 logistic regression",
    )

    payload = result_envelope(
        experiment="centralized_logreg",
        slug="logreg",
        config={
            "model": "LogisticRegression(l2, lbfgs)",
            "C_grid": np.logspace(-4, 4, 9).tolist(),
            "best_C": float(grid.best_params_["C"]),
            "cv_folds": 5,
            "seed": seed,
            "data_dir": data_dir,
        },
        split_fingerprint=fp,
        model_info={
            "num_params": int(clf.coef_.size + clf.intercept_.size),
            "n_iter": n_iter,
            "convergence_warnings": n_convergence_warnings,
            "fit_seconds": round(elapsed, 1),
        },
        metrics=report,
        history=[
            {
                "C": float(c),
                "cv_macro_f1": float(m),
                "cv_std": float(s),
                "train_macro_f1": float(t),
            }
            for c, m, s, t in zip(
                cv["param_C"].data, cv["mean_test_score"],
                cv["std_test_score"], cv["mean_train_score"],
            )
        ],
    )
    write_result(payload, out_dir)
    return report


def _plot_cv_curve(cv: dict, path: Path, best_c: float) -> None:
    """Macro-F1 vs regularization strength, with a +/-1 sigma band.

    With p >> n this plateau is more informative than the single accuracy number:
    it shows how much regularization the problem actually needs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = np.array([float(c) for c in cv["param_C"].data])
    mean, std = cv["mean_test_score"], cv["std_test_score"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(cs, mean - std, mean + std, color="#0072B2", alpha=0.15, linewidth=0)
    ax.plot(cs, mean, color="#0072B2", linewidth=2, marker="o", markersize=5)
    ax.axvline(best_c, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"best C = {best_c:.4g}", (best_c, ax.get_ylim()[0]),
        textcoords="offset points", xytext=(6, 8), fontsize=8, color=INK,
    )

    ax.set_xscale("log")
    ax.set_xlabel("C  (inverse regularization strength)", color=INK)
    ax.set_ylabel("CV macro-F1", color=INK)
    ax.set_title("Regularization path, 5-fold CV", color=INK, fontsize=11, pad=10)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. L1 sparse gene signature
# ---------------------------------------------------------------------------

def run_logreg_l1(arrays: dict, out_dir: Path, seed: int, fp: dict, data_dir: str) -> dict:
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]
    gene_names = arrays.get("gene_names")

    grid = GridSearchCV(
        LogisticRegression(penalty="l1", solver="saga", max_iter=2000, tol=1e-3),
        {"C": np.logspace(-3, 1, 5)},
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        scoring="f1_macro",
        n_jobs=-1,
    )

    print("  fitting L1/saga (slow at 20k features — a few minutes)...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    clf = grid.best_estimator_
    report = classification_report_dict(
        y_test, clf.predict(X_test), clf.predict_proba(X_test)
    )

    nonzero_per_class = (np.abs(clf.coef_) > 0).sum(axis=1).tolist()
    signature = {}
    for c, cls in enumerate(CLASSES):
        top = np.argsort(np.abs(clf.coef_[c]))[::-1][:20]
        signature[cls] = [
            {
                "gene": gene_names[i] if gene_names is not None else f"gene_{i}",
                "coef": float(clf.coef_[c, i]),
            }
            for i in top
            if clf.coef_[c, i] != 0
        ]

    print(
        f"  best C = {grid.best_params_['C']:.4g}  "
        f"test acc = {report['accuracy']:.4f}  macro-F1 = {report['macro_f1']:.4f}  "
        f"non-zero genes/class = {nonzero_per_class}  ({elapsed:.0f}s)"
    )

    payload = result_envelope(
        experiment="centralized_logreg_l1",
        slug="logreg_l1",
        config={
            "model": "LogisticRegression(l1, saga)",
            "best_C": float(grid.best_params_["C"]),
            "seed": seed,
            "data_dir": data_dir,
        },
        split_fingerprint=fp,
        model_info={
            "nonzero_coefs_per_class": nonzero_per_class,
            "fit_seconds": round(elapsed, 1),
        },
        metrics=report,
    )
    write_result(payload, out_dir)
    write_result(signature, out_dir, name="selected_genes.json")
    plot_confusion_matrix(
        report["confusion_matrix"], out_dir / "confusion_matrix.png",
        title="Centralized L1 logistic regression",
    )
    return report


# ---------------------------------------------------------------------------
# 3. Centralized MLP
# ---------------------------------------------------------------------------

def run_centralized_mlp(
    arrays: dict, out_dir: Path, seed: int, fp: dict, data_dir: str,
    epochs: int, lr: float, batch_size: int,
) -> dict:
    """Same architecture as the federated model, trained on pooled data.

    Two epoch budgets are run by the caller: `epochs=30` matches the federated
    client's epoch count (10 rounds x 3 local epochs), while `epochs=6` matches
    its *gradient-step* count, since 640 pooled samples give ~5x the steps of a
    128-sample shard. Reporting both forecloses the objection that the
    comparison was tilted either way.
    """
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, y_train = arrays["X_train"], arrays["y_train"]
    # A stratified slice for epoch-wise monitoring only; the reported numbers
    # still come from the untouched global test set.
    idx_fit, idx_mon = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=seed, stratify=y_train
    )

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X_train[idx]), torch.from_numpy(y_train[idx]))
        drop_last = shuffle and len(ds) > batch_size and len(ds) % batch_size == 1
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    fit_loader, mon_loader = loader(idx_fit, True), loader(idx_mon, False)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays["X_test"]), torch.from_numpy(arrays["y_test"])),
        batch_size=batch_size,
    )

    model = MLP()
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train(model, fit_loader, epochs=1, lr=lr, device=device)
        mon = evaluate_model(model, mon_loader, device)
        history.append({
            "round": epoch,
            "train_loss": train_loss,
            "val_loss": mon["loss"],
            "val_macro_f1": mon["macro_f1"],
        })
    elapsed = time.time() - t0

    report = evaluate_model(model, test_loader, device)
    print(
        f"  {epochs} epochs: test acc = {report['accuracy']:.4f}  "
        f"macro-F1 = {report['macro_f1']:.4f}  ({elapsed:.0f}s)"
    )

    payload = result_envelope(
        experiment=f"centralized_mlp_{epochs}ep",
        slug=f"mlp_{epochs}ep",
        config={
            "model": "MLP", "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "seed": seed, "data_dir": data_dir,
        },
        split_fingerprint=fp,
        model_info=communication_cost(model, num_clients=1, num_rounds=1),
        metrics=report,
        history=history,
    )
    write_result(payload, out_dir)
    plot_metric_curves(
        history, ["train_loss", "val_loss"], out_dir / "loss_curve.png",
        x_key="round", title=f"Centralized MLP, {epochs} epochs", xlabel="Epoch",
    )
    plot_confusion_matrix(
        report["confusion_matrix"], out_dir / "confusion_matrix.png",
        title=f"Centralized MLP ({epochs} epochs)",
    )
    return report


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-l1", action="store_true", help="skip the slow saga fit")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.results_dir) / "centralized"

    print(f"\nLoading {args.data_dir} ...")
    arrays = load_arrays(args.data_dir)
    fp = split_fingerprint(args.data_dir)
    print(
        f"  {len(arrays['y_train'])} train / {len(arrays['y_test'])} test, "
        f"{NUM_FEATURES} genes, split {fp['test_idx_sha1']}"
    )

    summary = {}

    print("\n[1/3] L2 logistic regression")
    summary["logreg"] = run_logreg(arrays, root / "logreg", args.seed, fp, args.data_dir)

    if args.skip_l1:
        print("\n[2/3] L1 logistic regression — skipped")
    else:
        print("\n[2/3] L1 logistic regression (sparse gene signature)")
        summary["logreg_l1"] = run_logreg_l1(
            arrays, root / "logreg_l1", args.seed, fp, args.data_dir
        )

    if args.skip_mlp:
        print("\n[3/3] Centralized MLP — skipped")
    else:
        print("\n[3/3] Centralized MLP")
        for epochs in (30, 6):
            summary[f"mlp_{epochs}ep"] = run_centralized_mlp(
                arrays, root / f"mlp_{epochs}ep", args.seed, fp, args.data_dir,
                epochs=epochs, lr=args.lr, batch_size=args.batch_size,
            )

    print("\n" + "=" * 62)
    print(f"{'model':<22}{'accuracy':>12}{'macro-F1':>12}{'COAD recall':>14}")
    print("-" * 62)
    for name, rep in summary.items():
        print(
            f"{name:<22}{rep['accuracy']:>12.4f}{rep['macro_f1']:>12.4f}"
            f"{rep['per_class']['COAD']['recall']:>14.4f}"
        )
    print("=" * 62)

    # V6 tripwire: this dataset is near-linearly separable, so anything below
    # 0.95 means the labels or the scaling are wrong, not that the model is weak.
    acc = summary["logreg"]["accuracy"]
    if acc < 0.95:
        print(f"\nV6 FAILED: logreg accuracy {acc:.4f} < 0.95.")
        print("Suspect label encoding or scaling — debug before running anything else.")
        return 1

    print(f"\nResults written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
