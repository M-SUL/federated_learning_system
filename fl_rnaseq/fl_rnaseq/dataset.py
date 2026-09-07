"""Data loading, preprocessing, and partitioning for TCGA PANCAN RNA-seq.

Reads the pruned expression matrix produced by ``notebooks/01_eda.ipynb``
(267 constant genes dropped from the raw 20531). Partitioning across clients is
delegated to :mod:`fl_rnaseq.partitioning`.

The global train/test split is carved out *before* any partitioning and is
deliberately fixed: ``SPLIT_SEED`` is not parameterisable, so every experiment
-- centralized, local-only, and federated -- is scored on an identical test set.
``split_fingerprint()`` exposes a hash that proves it.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fl_rnaseq.partitioning import get_partitions

# Fixed class order — deterministic label encoding across all workers.
# Must stay sorted: `le.classes_ = np.array(CLASSES)` below assumes the same
# ordering LabelEncoder would produce itself. Reordering this list without
# reordering the encoder would silently permute every label in the project.
CLASSES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
assert CLASSES == sorted(CLASSES), "CLASSES must be sorted — see comment above."

NUM_SAMPLES = 801
# data_pruned.csv: 20531 raw genes − 267 constant genes (notebooks/01_eda.ipynb).
NUM_FEATURES = 20264
NUM_CLASSES = len(CLASSES)

# 20 % of all 801 samples are held out as a global test set before any partitioning.
GLOBAL_TEST_FRACTION = 0.2

# Fixed by design — see module docstring. Do not thread a parameter through this.
SPLIT_SEED = 42

# Per-data-dir cache: each simulation worker process loads once per lifetime.
_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_preprocess(data_dir: str) -> dict:
    key = str(Path(data_dir).resolve())
    if key in _cache:
        return _cache[key]

    root = Path(data_dir)
    expr_path = root / "data_pruned.csv"

    # Parse straight to float32 rather than casting afterwards: this avoids a
    # ~130 MB float64 intermediate, which matters when several simulation workers
    # load in parallel. The dtype map must name only the gene columns — a blanket
    # dtype is applied before index_col is honoured and would try to coerce the
    # 'sample_N' index labels to float.
    gene_cols = pd.read_csv(expr_path, index_col=0, nrows=0).columns
    X_raw = pd.read_csv(
        expr_path, index_col=0, dtype={c: np.float32 for c in gene_cols}
    ).values
    y_raw = pd.read_csv(root / "labels.csv", index_col=0)["Class"].values

    if X_raw.shape != (NUM_SAMPLES, NUM_FEATURES):
        raise ValueError(
            f"Expected data_pruned.csv to be {(NUM_SAMPLES, NUM_FEATURES)}, "
            f"got {X_raw.shape}. Regenerate it by running notebooks/01_eda.ipynb "
            "through the 'Prune constant genes' cell."
        )

    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)
    y = le.transform(y_raw).astype(np.int64)

    # Stratified so each class keeps its ~20 % share of the test set. An
    # unstratified permutation over-samples BRCA (68 vs 60) and under-samples
    # PRAD (23 vs 27), which destabilises macro-F1 — the headline metric here.
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=GLOBAL_TEST_FRACTION, random_state=SPLIT_SEED
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    # Scaler fit only on training samples to prevent test-set leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    X_test = scaler.transform(X_raw[test_idx]).astype(np.float32)

    _cache[key] = {
        "X_train": X_train,
        "y_train": y[train_idx],
        "X_test": X_test,
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "test_idx_sha1": hashlib.sha1(test_idx.tobytes()).hexdigest()[:12],
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
        # Needed so the L1 sparse signature can name real genes.
        "gene_names": gene_cols.to_numpy(),
    }
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_arrays(data_dir: str) -> dict:
    """Public read-only view of the preprocessed split.

    Used by the baseline scripts and notebooks so they need not reach into a
    private function. Keys: ``X_train``, ``y_train``, ``X_test``, ``y_test``,
    ``train_idx``, ``test_idx``, ``test_idx_sha1``.
    """
    return dict(_load_and_preprocess(data_dir))


def split_fingerprint(data_dir: str) -> dict:
    """Identity of the global split, embedded in every results file.

    ``make_report.py`` asserts this is identical across all experiments before
    building any comparison table.
    """
    cache = _load_and_preprocess(data_dir)
    y_test = cache["y_test"]
    return {
        "n_train": int(len(cache["y_train"])),
        "n_test": int(len(y_test)),
        "stratified": True,
        "split_seed": SPLIT_SEED,
        "test_idx_sha1": cache["test_idx_sha1"],
        "test_class_counts": {
            CLASSES[c]: int((y_test == c).sum()) for c in range(NUM_CLASSES)
        },
    }


def load_data(
    partition_id: int,
    num_partitions: int,
    data_dir: str,
    batch_size: int,
    partition_strategy: str = "iid",
    dirichlet_alpha: float = 0.5,
    val_fraction: float = 0.2,
    seed: int = 42,
    min_partition_size: int = 32,
    self_balancing: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation DataLoaders for the requested partition.

    Args:
        partition_id:        Index of this client's shard (0-based).
        num_partitions:      Total number of clients / shards.
        data_dir:            Path to the folder with data_pruned.csv and labels.csv.
        batch_size:          Mini-batch size for both loaders.
        partition_strategy:  ``"iid"``, ``"dirichlet"``, or ``"dirichlet-legacy"``.
        dirichlet_alpha:     Concentration parameter (Dirichlet strategies only).
        val_fraction:        Fraction of the partition held back for local validation.
        seed:                Partition draw and loader shuffling. Does *not* affect
                             the global train/test split, fixed at SPLIT_SEED.
        min_partition_size:  Minimum shard size (``"dirichlet"`` only).
        self_balancing:      Damp oversized shards (``"dirichlet"`` only).
    """
    cache = _load_and_preprocess(data_dir)
    X_train, y_train = cache["X_train"], cache["y_train"]

    shards = get_partitions(
        strategy=partition_strategy,
        y_train=y_train,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
        seed=seed,
        min_partition_size=min_partition_size,
        self_balancing=self_balancing,
    )
    idx = shards[partition_id]

    # Shuffle before slicing: the Dirichlet partitioners emit indices grouped by
    # class, so an unshuffled `idx[:n_val]` would make the local validation set
    # a near-pure single-class prefix.
    shard_rng = np.random.default_rng(seed + 1000 + partition_id)
    idx = shard_rng.permutation(idx)

    n_val = max(1, int(len(idx) * val_fraction))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    if len(idx_train) < 2:
        raise ValueError(
            f"Client {partition_id} has {len(idx_train)} training samples after the "
            f"validation split (shard size {len(idx)}). Raise dirichlet-alpha or use "
            "the 'dirichlet' strategy, which enforces min_partition_size."
        )

    def make_loader(i: np.ndarray, shuffle: bool, train: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(X_train[i]), torch.from_numpy(y_train[i]))
        # BatchNorm1d raises on a training batch of size 1. Drop such a tail batch,
        # but only when a full batch survives — dropping unconditionally would make
        # len(loader) == 0 on a small shard and zero-divide in task.train().
        drop_last = train and len(ds) > batch_size and len(ds) % batch_size == 1
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=torch.Generator().manual_seed(seed + partition_id),
        )

    # Eval loader keeps drop_last=False: test() runs under net.eval(), where
    # BatchNorm uses running statistics and a batch of 1 is safe.
    return (
        make_loader(idx_train, shuffle=True, train=True),
        make_loader(idx_val, shuffle=False, train=False),
    )


def load_test_data(data_dir: str, batch_size: int) -> DataLoader:
    """Return a DataLoader for the held-out centralised test set (server-side only)."""
    cache = _load_and_preprocess(data_dir)
    ds = TensorDataset(
        torch.from_numpy(cache["X_test"]),
        torch.from_numpy(cache["y_test"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_pooled_train_data(data_dir: str, batch_size: int, seed: int = 42) -> DataLoader:
    """Return a DataLoader over the *entire* training pool, no partitioning.

    This is what makes the centralized baseline centralized: all 640 training
    samples in one loader, as if every client had pooled its data.
    """
    cache = _load_and_preprocess(data_dir)
    ds = TensorDataset(
        torch.from_numpy(cache["X_train"]),
        torch.from_numpy(cache["y_train"]),
    )
    drop_last = len(ds) > batch_size and len(ds) % batch_size == 1
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
    )
