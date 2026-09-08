"""Reproducibility, environment capture, and results-writing helpers.

Shared by the baseline scripts and the ServerApp so that every experiment emits
the same JSON envelope and uses the same directory naming.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch.

    Note that nothing in the original codebase seeded torch, so model
    initialisation and DataLoader shuffling were not reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    """Short HEAD hash, or None outside a repo / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def env_fingerprint() -> dict:
    """Versions and platform, recorded in every results file."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "scikit_learn": _pkg_version("scikit-learn"),
        "flwr": _pkg_version("flwr"),
        "flwr_datasets": _pkg_version("flwr-datasets"),
        "cuda_available": torch.cuda.is_available(),
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_bytes(model: torch.nn.Module) -> int:
    """Serialized fp32 size — one client's upload per round."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def communication_cost(model: torch.nn.Module, num_clients: int, num_rounds: int) -> dict:
    """Total simulated FedAvg traffic (upload + download, every client, every round)."""
    per_model = model_bytes(model)
    total = per_model * num_clients * num_rounds * 2
    return {
        "params": count_params(model),
        "bytes_per_model_fp32": per_model,
        "mb_per_model": round(per_model / 1e6, 2),
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
    }


def run_slug(
    strategy: str,
    alpha: float | None = None,
    seed: int = 42,
    norm: str = "batch",
) -> str:
    """Canonical directory name for one experiment configuration.

    Single source of truth so scripts and server_app cannot drift apart.

    Every knob that changes the result must appear here, or two different
    experiments write to the same directory and the second silently destroys the
    first. `norm` is suffixed only when it differs from the default, so existing
    result paths keep their names.
    """
    base = f"iid_seed{seed}" if (strategy == "iid" or alpha is None) \
        else f"{strategy}_a{alpha}_seed{seed}"
    return base if norm == "batch" else f"{base}_norm-{norm}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def result_envelope(
    experiment: str,
    slug: str,
    config: dict,
    split_fingerprint: dict,
    model_info: dict,
    metrics: dict,
    per_client: list | None = None,
    history: list | None = None,
) -> dict:
    """The one envelope shape every experiment writes.

    ``make_report.py`` relies on ``split_fingerprint.test_idx_sha1`` being present
    and identical across all files before it will build a comparison table.
    """
    payload = {
        "experiment": experiment,
        "slug": slug,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "env": env_fingerprint(),
        "config": config,
        "split_fingerprint": split_fingerprint,
        "model": model_info,
        "metrics": metrics,
    }
    if per_client is not None:
        payload["per_client"] = per_client
    if history is not None:
        payload["history"] = history
    return payload


def write_result(payload: dict, out_dir: str | Path, name: str = "metrics.json") -> Path:
    """Write one results JSON, creating parents as needed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)
    return path
