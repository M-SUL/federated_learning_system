"""Pre-flight checks. Run this before any experiment.

Covers V0-V4 from the plan: interpreter, package availability, data shape, label
encoding, split determinism and stratification, scaler leakage, and partition
invariants. Exits non-zero on the first hard failure.

    & "C:\\Program Files\\python311\\python.exe" -m scripts.verify_env
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = "../gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531"

_failures: list[str] = []
_warnings: list[str] = []


def check(label: str, ok: bool, detail: str = "", hard: bool = True) -> bool:
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{mark}] {label:<38} {detail}")
    if not ok:
        (_failures if hard else _warnings).append(label)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    print("\nInterpreter")
    print(f"  {sys.executable}")
    print(f"  Python {sys.version.split()[0]}")
    if ".virtualenvs" in sys.executable and "code-" in sys.executable:
        print(
            "\n  ERROR: this is the stray empty venv, not a project interpreter.\n"
            '  Use: & "C:\\Program Files\\python311\\python.exe" -m scripts.verify_env\n'
        )
        return 2

    print("\nPackages")
    from fl_rnaseq.utils import env_fingerprint

    env = env_fingerprint()
    for pkg in ("torch", "numpy", "pandas", "scikit_learn"):
        check(pkg, env[pkg] is not None, str(env[pkg]))
    for pkg in ("flwr", "flwr_datasets"):
        check(
            pkg, env[pkg] is not None,
            str(env[pkg]) if env[pkg] else "not installed (needed from Phase 5 on)",
            hard=False,
        )
    print(f"  [INFO] cuda_available                    {env['cuda_available']}")

    data_dir = Path(args.data_dir)
    if not (data_dir / "data_pruned.csv").exists():
        print(f"\n  FAIL: {data_dir / 'data_pruned.csv'} not found.")
        print("  Generate it by running notebooks/01_eda.ipynb through the")
        print("  'Prune constant genes' cell. See RUNNING.md.")
        return 2

    print("\nData and split  (parsing ~200 MB, takes a few seconds)")
    from fl_rnaseq.dataset import (
        CLASSES, NUM_CLASSES, NUM_FEATURES, load_arrays, split_fingerprint,
    )

    arrays = load_arrays(str(data_dir))
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    n_train, n_test = len(y_train), len(y_test)

    check("CLASSES is sorted", CLASSES == sorted(CLASSES), str(CLASSES))
    check("feature count", X_train.shape[1] == NUM_FEATURES, f"{X_train.shape[1]} genes")
    check(
        "all 801 samples accounted for",
        n_train + n_test == 801,
        f"{n_train} train + {n_test} test = {n_train + n_test}",
    )
    # StratifiedShuffleSplit rounds per class, so the test set lands near but not
    # exactly on 20% of 801 (160.2). Anything within a couple of samples is correct.
    check("test set is ~20%", abs(n_test - 0.2 * 801) <= 2, f"{n_test} ({n_test / 801:.1%})")
    check(
        "train/test disjoint",
        len(np.intersect1d(arrays["train_idx"], arrays["test_idx"])) == 0,
        "no shared indices",
    )
    check(
        "scaler fit on train only (V3)",
        arrays["scaler_n_samples_seen"] == n_train,
        f"n_samples_seen_ = {arrays['scaler_n_samples_seen']} (== n_train)",
    )

    fp = split_fingerprint(str(data_dir))
    counts = fp["test_class_counts"]
    full = {c: int((y_train == i).sum() + (y_test == i).sum()) for i, c in enumerate(CLASSES)}
    worst = max(abs(counts[c] - 0.2 * full[c]) for c in CLASSES)
    check(
        "test set is stratified (V2)",
        worst <= 1.0,
        f"{counts}  max deviation from 20% = {worst:.1f}",
    )

    # Determinism: a second load must produce the same split. Clear the cache so
    # this re-derives rather than returning the memoised dict.
    import fl_rnaseq.dataset as ds_mod

    first_sha = fp["test_idx_sha1"]
    ds_mod._cache.clear()
    check(
        "split is deterministic (V2)",
        split_fingerprint(str(data_dir))["test_idx_sha1"] == first_sha,
        f"test_idx_sha1 = {first_sha}",
    )

    print("\nModel")
    import torch
    from fl_rnaseq.task import MLP
    from fl_rnaseq.utils import communication_cost, count_params

    model = MLP()
    out = model(torch.randn(4, NUM_FEATURES))
    check("forward pass (V0)", tuple(out.shape) == (4, NUM_CLASSES), str(tuple(out.shape)))
    for norm in ("layer", "none"):
        m = MLP(norm=norm)
        check(f"norm={norm!r} builds", tuple(m(torch.randn(4, NUM_FEATURES)).shape) == (4, 5))

    cost = communication_cost(model, num_clients=5, num_rounds=10)
    print(
        f"  [INFO] params {count_params(model):,}  "
        f"{cost['mb_per_model']} MB/model  {cost['total_gb']} GB total traffic"
    )

    print("\nPartitioning (V4)")
    from fl_rnaseq.partitioning import (
        VALID_STRATEGIES, get_partitions, heterogeneity_index, label_distribution,
    )

    for strategy in ("iid", "dirichlet-legacy"):
        try:
            parts = get_partitions(strategy, y_train, num_partitions=5, alpha=0.5, seed=42)
            sizes = [len(p) for p in parts]
            hetero = heterogeneity_index(label_distribution(parts, y_train, NUM_CLASSES))
            check(strategy, True, f"sizes={sizes}  JS={hetero:.3f}")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole gate
            check(strategy, False, f"{type(exc).__name__}: {exc}")

    if env["flwr_datasets"]:
        import inspect
        from flwr_datasets.partitioner import DirichletPartitioner

        print(f"  [INFO] DirichletPartitioner{inspect.signature(DirichletPartitioner.__init__)}")
        try:
            parts = get_partitions("dirichlet", y_train, 5, alpha=0.5, seed=42)
            check("dirichlet (flwr-datasets)", True, f"sizes={[len(p) for p in parts]}")
        except Exception as exc:  # noqa: BLE001
            check("dirichlet (flwr-datasets)", False, f"{type(exc).__name__}: {exc}")
    else:
        check(
            "dirichlet (flwr-datasets)", False,
            "skipped — flwr-datasets not installed", hard=False,
        )

    check(
        "unknown strategy rejected",
        _raises(lambda: get_partitions("dirchlet", y_train, 5)),
        f"valid: {', '.join(VALID_STRATEGIES)}",
    )

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    if _warnings:
        print(f"OK, with {len(_warnings)} warning(s): {', '.join(_warnings)}")
    else:
        print("All checks passed.")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
