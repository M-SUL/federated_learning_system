"""Turn the results files into one consistent shape.

`results/**/metrics.json` grew organically across three experiment families and
is genuinely inconsistent: `history` means three different things, `model` has
four shapes, `per_class` is missing from one file and lacks `precision` in eight
others, and `classes_predicted` is sometimes a strict subset of the classes
actually present. Absorbing all of that here means every view in the frontend
gets one shape with nulls, instead of each view rediscovering the same defects.

This module is PURE: dicts in, dicts out, no filesystem, no network. That is what
makes it testable against all 19 real files.

It deliberately does not import `fl_rnaseq` -- the dashboard must run on a plain
Python with no training dependencies. The few constants duplicated below name
their source of truth, and `tests/test_normalize.py` asserts they still match.
"""

from __future__ import annotations

from typing import Any

# Source of truth: fl_rnaseq/fl_rnaseq/dataset.py:29 (CLASSES)
CLASSES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

# Source of truth: fl_rnaseq/fl_rnaseq/metrics.py:39 (CLASS_COLORS)
# CVD-validated: worst adjacent pair dE 11.0 (deutan), 16.4 normal vision.
CLASS_COLORS = {
    "BRCA": "#0072B2",
    "COAD": "#E69F00",
    "KIRC": "#009E73",
    "LUAD": "#D55E00",
    "PRAD": "#CC79A7",
}
# Source of truth: fl_rnaseq/fl_rnaseq/metrics.py:47 (METHOD_COLORS)
METHOD_COLORS = {
    "centralized": "#0072B2",
    "federated": "#009E73",
    "local_only": "#D55E00",
}
INK, INK_MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

FAMILY_ORDER = ["centralized", "federated", "local_only"]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _label(family: str, slug: str, cfg: dict, experiment: str) -> str:
    """Human-readable run name for legends and tables."""
    if family == "centralized":
        return {
            "centralized_logreg": "Centralized logistic regression (L2)",
            "centralized_logreg_l1": "Centralized logistic regression (L1)",
        }.get(experiment) or f"Centralized MLP, {cfg.get('epochs') or cfg.get('num_rounds')} epochs"

    strategy, alpha = cfg.get("strategy"), cfg.get("alpha")
    part = "IID" if strategy == "iid" or alpha is None else f"Dirichlet a={alpha}"
    if strategy == "dirichlet-legacy":
        part += " (legacy partitioner)"
    prefix = "FedAvg" if family == "federated" else "Isolated"
    norm = cfg.get("norm")
    suffix = f", {norm}Norm" if norm and norm != "batch" else ""
    return f"{prefix}, {part}{suffix}"


# ---------------------------------------------------------------------------
# Curves — `history` means three different things
# ---------------------------------------------------------------------------

def _curves(history: list[dict] | None) -> dict:
    """Normalize `history` by probing its keys, never by filename or family.

    Three incompatible shapes share the file key `history`, and two of them even
    share the element key `round` while meaning different things (server rounds
    0-indexed from an untrained init, vs training epochs 1-indexed).
    """
    empty = {"kind": None, "x": None, "series": [], "points": []}
    if not history:
        return empty
    keys = set(history[0])

    if {"round", "accuracy", "macro_f1", "loss"} <= keys:
        return {
            "kind": "rounds",
            # round 0 is the untrained initial model, not a training step. Views
            # must mark it distinctly or the convergence story reads wrong.
            "x": {"key": "round", "label": "FedAvg round", "scale": "linear",
                  "zero_is_init": True},
            "series": [
                {"key": "macro_f1", "label": "Macro-F1", "axis": "metric", "band": None},
                {"key": "accuracy", "label": "Accuracy", "axis": "metric", "band": None},
                {"key": "loss", "label": "Test loss", "axis": "loss", "band": None},
            ],
            "points": [dict(h) for h in history],
        }

    if {"round", "train_loss", "val_loss", "val_macro_f1"} <= keys:
        return {
            "kind": "epochs",
            # Same "round" key as above, but these are 1-based epochs.
            "x": {"key": "round", "label": "Epoch", "scale": "linear",
                  "zero_is_init": False},
            "series": [
                {"key": "train_loss", "label": "Train loss", "axis": "loss", "band": None},
                {"key": "val_loss", "label": "Val loss", "axis": "loss", "band": None},
                {"key": "val_macro_f1", "label": "Val macro-F1", "axis": "metric", "band": None},
            ],
            "points": [dict(h) for h in history],
        }

    if {"C", "cv_macro_f1"} <= keys:
        return {
            "kind": "cv",
            # C spans 1e-4..1e4; a linear axis collapses it into a spike.
            "x": {"key": "C", "label": "C (inverse regularization)", "scale": "log",
                  "zero_is_init": False},
            "series": [
                {"key": "cv_macro_f1", "label": "CV macro-F1", "axis": "metric",
                 "band": "cv_std"},
                {"key": "train_macro_f1", "label": "Train macro-F1", "axis": "metric",
                 "band": None},
            ],
            "points": [dict(h) for h in history],
        }

    return empty


# ---------------------------------------------------------------------------
# Per-class and confusion
# ---------------------------------------------------------------------------

def _per_class(block: dict | None) -> tuple[list[dict] | None, bool]:
    """List form of `per_class`, dropping support-0 entries.

    A class the model never saw is reported as {precision:0, recall:0, f1:0,
    support:0}. Rendering that as a bar says "scored zero" when the truth is "no
    samples". Dropped here rather than in the views so no view can forget.
    `precision` is absent in every local_only file, hence the .get().
    """
    if not block:
        return None, False
    rows = []
    for cls in CLASSES:
        e = block.get(cls)
        if not e or int(e.get("support", 0)) <= 0:
            continue
        rows.append({
            "class": cls,
            "color": CLASS_COLORS[cls],
            "precision": e.get("precision"),
            "recall": e.get("recall"),
            "f1": e.get("f1"),
            "support": int(e.get("support", 0)),
        })
    return rows, True


def _missing_classes(metrics: dict) -> list[str]:
    """Classes present in the truth but never predicted.

    A set difference, never a positional zip: `classes_predicted` is a strict
    subset of `classes_present_in_truth`, so zipping silently mislabels.
    """
    truth = set(metrics.get("classes_present_in_truth") or [])
    pred = set(metrics.get("classes_predicted") or [])
    return [CLASSES[i] for i in sorted(truth - pred) if 0 <= i < len(CLASSES)]


def _headline(metrics: dict) -> dict:
    """Flat scores. All three families expose `accuracy` and `macro_f1`."""
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        # null when a client's val split held fewer than 2 classes -- a finding
        # about alpha=0.1, not missing data.
        "macro_auroc": metrics.get("macro_auroc"),
        "loss": metrics.get("loss"),
    }


# ---------------------------------------------------------------------------
# Sites (local_only only)
# ---------------------------------------------------------------------------

def _sites(per_client: list[dict] | None) -> list[dict] | None:
    if not per_client:
        return None
    out = []
    for c in per_client:
        # A degenerate client carries only {client_id, status, error} -- branch
        # on status before touching anything else.
        if c.get("status") != "ok":
            out.append({
                "site_id": c.get("client_id"), "ok": False,
                "status": c.get("status"), "error": c.get("error"),
                "n_train": None, "n_val": None, "n_shard": None,
                "n_classes_seen": 0, "label_counts": [], "train_loss": None,
                "seconds": None, "global": None, "local_val": None,
            })
            continue
        counts = c.get("label_counts") or {}
        g, lv = c.get("global") or {}, c.get("local_val") or {}
        out.append({
            "site_id": c.get("client_id"),
            "ok": True,
            "status": "ok",
            "error": None,
            "n_train": c.get("n_train"),
            "n_val": c.get("n_val"),
            # label_counts covers train+val, so it is the shard, not the train split.
            "n_shard": sum(int(v) for v in counts.values()) or None,
            "n_classes_seen": c.get("n_classes_seen"),
            "label_counts": [
                {"class": k, "count": int(counts.get(k, 0)), "color": CLASS_COLORS[k]}
                for k in CLASSES
            ],
            "train_loss": c.get("train_loss"),
            "seconds": c.get("seconds"),
            "global": {**_headline(g), "per_class": _per_class(g.get("per_class"))[0],
                       "confusion": g.get("confusion_matrix"),
                       "missing_classes": _missing_classes(g)} if g else None,
            "local_val": {**_headline(lv),
                          "missing_classes": _missing_classes(lv)} if lv else None,
        })
    return out


def _site_summary(metrics: dict) -> dict | None:
    keys = ("global_accuracy", "global_macro_f1", "local_val_accuracy")
    if not any(k in metrics for k in keys):
        return None
    return {
        **{k: metrics.get(k) for k in keys},
        "heterogeneity_js": metrics.get("heterogeneity_js"),
        "n_ok": metrics.get("n_clients_ok"),
        "n_degenerate": metrics.get("n_clients_degenerate"),
    }


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

# 640 training samples x 20264 genes x 4 bytes (float32) -- what pooling would move.
POOLED_TRAIN_BYTES = 640 * 20264 * 4


def _cost(family: str, model: dict, cfg: dict) -> dict:
    rounds = cfg.get("num_rounds") or 0
    clients = cfg.get("num_clients") or 0
    per = model.get("bytes_per_model_fp32")
    transfers = rounds * clients * 2 if family == "federated" else 0

    leaves = []
    if family == "federated":
        leaves = [
            {"item": "Model weights and biases", "kind": "parameters",
             "count": model.get("params"), "every_round": True, "protocol": True},
            {"item": "num-examples (shard size)", "kind": "scalar",
             "count": 1, "every_round": True, "protocol": True},
            {"item": "train_loss", "kind": "scalar",
             "count": 1, "every_round": True, "protocol": True},
        ]
        if cfg.get("norm", "batch") == "batch":
            leaves.append({
                "item": "BatchNorm running_mean / running_var", "kind": "distribution statistics",
                "count": 1280, "every_round": True, "protocol": True,
                "note": "These are per-feature distributional statistics of local "
                        "expression data, averaged across sites by FedAvg. The "
                        "norm='layer' variant removes them entirely -- and scored "
                        "higher at alpha=0.1, so the leak was not buying accuracy.",
            })
        leaves.append({
            "item": "Dashboard telemetry (label histogram, shard size)",
            "kind": "class distribution", "count": None, "every_round": True,
            "protocol": False,
            "note": "Simulation-only, for this dashboard. Not part of FedAvg: it "
                    "rides in a ConfigRecord that aggregation ignores. "
                    "emit-site-telemetry=false removes it.",
        })

    return {
        "weights_bytes_per_transfer": per,
        "transfers": transfers,
        "weights_bytes_total": model.get("total_bytes"),
        "total_gb": model.get("total_gb"),
        "raw_data_bytes_if_pooled": POOLED_TRAIN_BYTES,
        "raw_data_leaves_client": family == "centralized",
        "what_leaves_the_client": leaves,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize_run(raw: dict, family: str, artifacts: list[str] | None = None) -> dict:
    """One `metrics.json` -> the single shape every view consumes."""
    cfg = dict(raw.get("config") or {})
    metrics = dict(raw.get("metrics") or {})
    model = dict(raw.get("model") or {})
    slug = raw.get("slug", "")
    experiment = raw.get("experiment", "")

    per_class, present = _per_class(metrics.get("per_class"))
    health = model.get("health") or {}

    return {
        "id": f"{family}/{slug}",
        "family": family,
        "slug": slug,
        "experiment": experiment,
        "label": _label(family, slug, cfg, experiment),
        "timestamp": raw.get("timestamp"),
        "git_commit": raw.get("git_commit"),
        "env": raw.get("env") or {},

        "config": {
            "strategy": cfg.get("strategy"),
            "alpha": cfg.get("alpha"),
            "seed": cfg.get("seed"),
            "num_clients": cfg.get("num_clients"),
            "num_rounds": cfg.get("num_rounds"),
            "local_epochs": cfg.get("local_epochs"),
            "epochs": cfg.get("epochs"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "norm": cfg.get("norm"),
            "raw": cfg,
        },

        "headline": _headline(metrics),
        "per_class": per_class,
        "per_class_present": present,
        "classes_missing_from_predictions": _missing_classes(metrics),
        "confusion": {
            "present": bool(metrics.get("confusion_matrix")),
            "labels": list(CLASSES),
            "rows": metrics.get("confusion_matrix"),
        },
        "curves": _curves(raw.get("history")),
        "sites": _sites(raw.get("per_client")),
        "site_summary": _site_summary(metrics),

        "model": {
            # logreg reports num_params; MLP reports params; logreg_l1 neither.
            "kind": model.get("model") or cfg.get("model") or "MLP",
            "params": model.get("params") or model.get("num_params"),
            "bytes_per_model_fp32": model.get("bytes_per_model_fp32"),
            "mb_per_model": model.get("mb_per_model"),
            "total_gb": model.get("total_gb"),
            "raw": model,
        },
        # A run where every client failed in some round leaves the global model
        # untouched there; without this it reads as slow convergence.
        "health": {
            "degraded": bool(health.get("degraded")),
            "failed_rounds": health.get("failed_rounds") or [],
            "n_client_errors": health.get("n_client_errors") or 0,
        },
        "cost": _cost(family, model, cfg),
        "partition_slug": slug if family in ("federated", "local_only") else None,
        "artifacts": artifacts or [],
    }
