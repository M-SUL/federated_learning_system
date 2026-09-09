"""Client partitioning strategies for the federated simulation.

Three strategies are selectable via the ``partition-strategy`` run config:

  - ``"iid"``              : random uniform split (baseline)
  - ``"dirichlet"``        : non-IID split via ``flwr-datasets`` DirichletPartitioner
  - ``"dirichlet-legacy"`` : the original hand-rolled Dirichlet implementation

``"dirichlet"`` is preferred over ``"dirichlet-legacy"`` because it enforces a
``min_partition_size``: with 640 training samples across 5 clients the expected
shard is 128, but a single unlucky draw at alpha=0.5 can hand a client 3
samples, which crashes BatchNorm and contributes a garbage update weighted
3/640. The upstream partitioner resamples until the floor is met.

``"dirichlet-legacy"`` is retained so the local-only baseline can run before
`flwr-datasets` is installed, and as a cross-implementation check.

This module imports `datasets` / `flwr_datasets` lazily so that `dataset.py`
and every baseline script stay importable with only numpy/torch/sklearn.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

VALID_STRATEGIES = ("iid", "dirichlet", "dirichlet-legacy")

# Partitioning is deterministic in its inputs, but `load_data` is called twice
# per client per round (train + evaluate), so a 10-round 5-client run would
# otherwise redo the work ~100 times -- including the multi-second `datasets`
# import. Memoise on the full input key.
_partition_cache: dict[tuple, list[np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _iid(n: int, num_partitions: int, seed: int) -> list[np.ndarray]:
    """Shuffle all n training indices and split into equal shards."""
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(n), num_partitions))


def _dirichlet_legacy(
    y_train: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Original hand-rolled Dirichlet partition, preserved verbatim.

    Each class's samples are distributed across clients by drawing from
    Dirichlet(alpha, ..., alpha). Lower alpha = more skewed.

    Note this offers no minimum-partition-size guarantee -- that is precisely
    what ``_dirichlet_fds`` adds.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(y_train.max()) + 1
    partitions: list[list[int]] = [[] for _ in range(num_partitions)]

    for cls in range(num_classes):
        cls_idx = np.where(y_train == cls)[0].copy()
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.full(num_partitions, alpha))
        # Compute integer split points; last partition absorbs any rounding remainder.
        split_points = np.round(np.cumsum(proportions[:-1]) * len(cls_idx)).astype(int)
        for pid, chunk in enumerate(np.split(cls_idx, split_points)):
            partitions[pid].extend(chunk.tolist())

    return [np.array(p, dtype=int) for p in partitions]


def _dirichlet_fds(
    y_train: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_partition_size: int,
    self_balancing: bool,
) -> list[np.ndarray]:
    """Non-IID partition via the upstream ``flwr-datasets`` DirichletPartitioner.

    The partitioner only ever reads the column named by ``partition_by``, so we
    hand it a two-column table of (row index, label) rather than the full
    801x20264 expression matrix. The resulting assignment is identical, at a
    fraction of the memory -- the gene expression values never enter it, and
    nothing is downloaded.
    """
    # `Dataset.from_dict` never hits the network, but `datasets` / `huggingface_hub`
    # can attempt telemetry or cache setup on import. Force offline for determinism.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from datasets import Dataset
    from flwr_datasets.partitioner import DirichletPartitioner

    ds = Dataset.from_dict(
        {
            "idx": np.arange(len(y_train), dtype=np.int64),
            "label": y_train.astype(np.int64),
        }
    )

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=min_partition_size,
        self_balancing=self_balancing,
        shuffle=True,
        seed=seed,
    )
    partitioner.dataset = ds

    # Read indices back through the public `idx` column rather than the private
    # `_partition_id_to_indices` attribute, whose name has changed across releases.
    return [
        np.asarray(partitioner.load_partition(pid)["idx"], dtype=int)
        for pid in range(num_partitions)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_partitions(
    strategy: str,
    y_train: np.ndarray,
    num_partitions: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_partition_size: int = 32,
    self_balancing: bool = False,
) -> list[np.ndarray]:
    """Return one index array per client, indexing into the training pool.

    Args:
        strategy:            One of ``VALID_STRATEGIES``.
        y_train:             Integer labels for the training pool (test set excluded).
        num_partitions:      Number of clients.
        alpha:               Dirichlet concentration. Lower = more heterogeneous.
        seed:                Controls the partition draw. Determinism depends only
                             on (strategy, seed, alpha, num_partitions, labels) --
                             never on process-local state, since each simulation
                             worker builds its own partitioner.
        min_partition_size:  Minimum samples per client (``"dirichlet"`` only).
        self_balancing:      Damp the proportions of oversized partitions
                             (``"dirichlet"`` only). Reduces heterogeneity, so
                             report which setting was used.

    Raises:
        ValueError: on an unknown strategy, or if the result violates the
            disjoint / covering / non-degenerate post-conditions.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown partition strategy {strategy!r}. "
            f"Valid options: {', '.join(VALID_STRATEGIES)}"
        )

    labels_sha = hashlib.sha1(np.ascontiguousarray(y_train).tobytes()).hexdigest()[:12]
    key = (
        strategy, num_partitions, alpha, seed,
        min_partition_size, self_balancing, labels_sha,
    )
    if key in _partition_cache:
        return _partition_cache[key]

    if strategy == "iid":
        parts = _iid(len(y_train), num_partitions, seed)
    elif strategy == "dirichlet-legacy":
        parts = _dirichlet_legacy(y_train, num_partitions, alpha, seed)
    else:
        parts = _dirichlet_fds(
            y_train, num_partitions, alpha, seed, min_partition_size, self_balancing
        )

    _check_partitions(parts, len(y_train), strategy)
    _partition_cache[key] = parts
    return parts


def _check_partitions(parts: list[np.ndarray], n: int, strategy: str) -> None:
    """Assert the partition is a valid disjoint cover with usable shard sizes."""
    concat = np.concatenate(parts) if parts else np.array([], dtype=int)

    if len(concat) != len(np.unique(concat)):
        raise ValueError(f"{strategy}: partitions overlap -- a sample is on two clients.")
    if len(concat) != n or not np.array_equal(np.sort(concat), np.arange(n)):
        raise ValueError(
            f"{strategy}: partitions do not cover the training pool "
            f"({len(concat)} indices for {n} samples)."
        )

    smallest = min(len(p) for p in parts)
    if smallest < 2:
        raise ValueError(
            f"{strategy}: client shard of size {smallest} is unusable (need >= 2). "
            "Raise dirichlet-alpha, lower num-partitions, or use the 'dirichlet' "
            "strategy, which enforces min_partition_size."
        )


def label_distribution(
    parts: list[np.ndarray], y_train: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return the (num_clients, num_classes) count matrix for a partition."""
    return np.array(
        [np.bincount(y_train[p], minlength=num_classes) for p in parts], dtype=int
    )


def heterogeneity_index(counts: np.ndarray) -> float:
    """Mean Jensen-Shannon divergence between each client's label distribution
    and the global one.

    A single scalar summarising "how non-IID is this split", so accuracy can be
    plotted against heterogeneity as a continuous curve rather than three
    anecdotes. 0.0 = every client matches the global distribution.
    """
    eps = 1e-12
    global_p = counts.sum(axis=0) / max(counts.sum(), 1)
    divergences = []

    for row in counts:
        total = row.sum()
        if total == 0:
            continue
        p = row / total
        m = 0.5 * (p + global_p)
        # KL(p||m) and KL(global||m), both base 2 -> JS in [0, 1].
        kl_pm = np.sum(np.where(p > 0, p * np.log2((p + eps) / (m + eps)), 0.0))
        kl_gm = np.sum(
            np.where(global_p > 0, global_p * np.log2((global_p + eps) / (m + eps)), 0.0)
        )
        divergences.append(0.5 * kl_pm + 0.5 * kl_gm)

    return float(np.mean(divergences)) if divergences else 0.0
