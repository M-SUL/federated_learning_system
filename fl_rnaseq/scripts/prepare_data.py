"""Fetch and prepare the TCGA PANCAN dataset without Jupyter.

The pruning step lived only in notebooks/01_eda.ipynb, which is fine on a laptop
and useless in a container. This reproduces it exactly -- drop the genes with
zero variance, leaving 801 x 20264 -- so a fresh machine can go from nothing to
a runnable dataset with one command.

    python -m scripts.prepare_data --data-dir <dir>

Idempotent: it downloads only what is missing and re-verifies what is present.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
INNER_DIR = "TCGA-PANCAN-HiSeq-801x20531"

EXPECTED_RAW = (801, 20531)
EXPECTED_PRUNED = (801, 20264)
CLASSES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def _download(dest: Path) -> Path:
    archive = dest / "uci_rnaseq.zip"
    if archive.exists():
        print(f"  archive already present: {archive}")
        return archive
    print(f"  downloading {UCI_URL}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(".partial")
    with urllib.request.urlopen(UCI_URL, timeout=300) as r, tmp.open("wb") as fh:
        shutil.copyfileobj(r, fh)
    # Rename only once complete, so an interrupted download is never mistaken
    # for a finished one on the next boot.
    tmp.rename(archive)
    print(f"  saved {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    return archive


def _extract(archive: Path, dest: Path) -> Path:
    target = dest / INNER_DIR
    if (target / "data.csv").exists():
        print(f"  already extracted: {target}")
        return target
    print("  extracting")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    # The zip contains a tarball, which contains the directory.
    for tgz in dest.glob("*.tar.gz"):
        with tarfile.open(tgz) as t:
            t.extractall(dest)
    if not (target / "data.csv").exists():
        raise SystemExit(f"expected {target / 'data.csv'} after extraction")
    return target


def _prune(data_dir: Path) -> None:
    """Drop zero-variance genes. Mirrors notebooks/01_eda.ipynb."""
    pruned = data_dir / "data_pruned.csv"
    if pruned.exists():
        cols = len(pd.read_csv(pruned, index_col=0, nrows=0).columns)
        if cols == EXPECTED_PRUNED[1]:
            print(f"  data_pruned.csv already correct ({cols} genes)")
            return
        print(f"  data_pruned.csv has {cols} genes, expected {EXPECTED_PRUNED[1]} — rebuilding")

    print("  loading raw matrix (this takes a minute)")
    X = pd.read_csv(data_dir / "data.csv", index_col=0)
    if X.shape != EXPECTED_RAW:
        raise SystemExit(f"raw matrix is {X.shape}, expected {EXPECTED_RAW}")

    constant = X.std(axis=0) == 0
    X_pruned = X.loc[:, X.columns[~constant]]
    print(f"  dropped {int(constant.sum())} constant genes -> {X_pruned.shape}")
    if X_pruned.shape != EXPECTED_PRUNED:
        raise SystemExit(f"pruned matrix is {X_pruned.shape}, expected {EXPECTED_PRUNED}")

    tmp = pruned.with_suffix(".partial")
    X_pruned.to_csv(tmp)
    tmp.rename(pruned)
    print(f"  wrote {pruned}")


def _verify(data_dir: Path) -> None:
    labels = pd.read_csv(data_dir / "labels.csv", index_col=0)["Class"]
    found = sorted(labels.unique())
    if found != CLASSES:
        raise SystemExit(f"labels are {found}, expected {CLASSES}")
    cols = len(pd.read_csv(data_dir / "data_pruned.csv", index_col=0, nrows=0).columns)
    if cols != EXPECTED_PRUNED[1]:
        raise SystemExit(f"data_pruned.csv has {cols} genes, expected {EXPECTED_PRUNED[1]}")
    print(f"  verified: {len(labels)} samples, {cols} genes, classes {found}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory that will contain TCGA-PANCAN-HiSeq-801x20531/")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the downloaded zip (default: delete to save ~70 MB)")
    args = ap.parse_args()

    root = args.data_dir.resolve()
    print(f"Preparing dataset in {root}")

    inner = root / INNER_DIR
    if not (inner / "data.csv").exists():
        archive = _download(root)
        inner = _extract(archive, root)
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            for leftover in root.glob("*.tar.gz"):
                leftover.unlink(missing_ok=True)
    else:
        print(f"  raw data already present: {inner}")

    _prune(inner)
    _verify(inner)
    print("Dataset ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
