"""Assemble every stored result into one payload for the dashboard.

Deliberately does NOT read `results/summary/summary.csv`: that file is itself
derived by `make_report.py`, uses "-" sentinels, and leaves `heterogeneity_js`
blank on federated rows. Recomputing from the metrics.json files keeps one
derivation path, so the dashboard and the CSV cannot disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

from .normalize import (
    CLASS_COLORS, CLASSES, FAMILY_ORDER, GRID, INK, INK_MUTED, METHOD_COLORS,
    normalize_run,
)

_cache: dict = {}


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _partitions(results_dir: Path, runs: list[dict], warnings: list[str]) -> dict:
    """Shard composition per configuration.

    partitions.json covers only the 5 configurations `plot_partition_distribution`
    was run for, but 8 appear in the results. The missing 3 are recoverable: a
    local_only run records each client's label histogram and the run's JS
    divergence, which is exactly the same information. Reconstruct them and label
    the provenance rather than showing gaps.
    """
    out: dict[str, dict] = {}

    raw = _load(results_dir / "partitions" / "partitions.json") or {}
    for entry in raw.get("partitions", []):
        counts = entry.get("label_counts") or {}
        out[entry["slug"]] = {
            "slug": entry["slug"],
            "strategy": entry.get("strategy"),
            "alpha": entry.get("alpha"),
            "seed": entry.get("seed"),
            "heterogeneity_js": entry.get("heterogeneity_js"),
            "shard_sizes": entry.get("shard_sizes"),
            "classes_per_client": entry.get("classes_per_client"),
            "clients": [
                {
                    "site_id": i,
                    "label_counts": [
                        {"class": c, "count": int(counts.get(f"client_{i}", {}).get(c, 0)),
                         "color": CLASS_COLORS[c]}
                        for c in CLASSES
                    ],
                }
                for i in range(len(entry.get("shard_sizes") or []))
            ],
            "source": "partitions.json",
        }

    for run in runs:
        slug = run["slug"]
        if run["family"] != "local_only" or slug in out or not run["sites"]:
            continue
        sites = [s for s in run["sites"] if s["ok"]]
        out[slug] = {
            "slug": slug,
            "strategy": run["config"]["strategy"],
            "alpha": run["config"]["alpha"],
            "seed": run["config"]["seed"],
            "heterogeneity_js": (run["site_summary"] or {}).get("heterogeneity_js"),
            "shard_sizes": [s["n_shard"] for s in sites],
            "classes_per_client": [s["n_classes_seen"] for s in sites],
            "clients": [
                {"site_id": s["site_id"], "label_counts": s["label_counts"]} for s in sites
            ],
            "source": "derived-from-local-only",
        }
        warnings.append(
            f"partitions.json has no entry for {slug}; shard composition derived "
            "from the local-only run's per-client label counts."
        )

    return out


def _comparisons(runs: list[dict], partitions: dict) -> list[dict]:
    """Pair federated against isolated on identical shards.

    Joined on (strategy, alpha, seed, norm) rather than slug: local_only carries
    dirichlet-legacy_* rows that federated has no counterpart for, and joining by
    slug would orphan them silently. Legacy rows stay a separate series -- they
    are different shards, not a repeat of the same experiment.
    """
    ceiling = max(
        (r["headline"]["macro_f1"] or 0) for r in runs if r["family"] == "centralized"
    )
    keyed: dict[tuple, dict] = {}

    for r in runs:
        if r["family"] == "centralized":
            continue
        c = r["config"]
        key = (c["strategy"], c["alpha"], c["seed"], c["norm"] or "batch")
        slot = keyed.setdefault(key, {
            "key": "|".join(str(x) for x in key),
            "strategy": c["strategy"], "alpha": c["alpha"], "seed": c["seed"],
            "norm": c["norm"] or "batch",
            "is_legacy": c["strategy"] == "dirichlet-legacy",
            "centralized_ceiling": ceiling,
            "federated": None, "local_only": None, "gain": None,
            "heterogeneity_js": None, "partition_source": None,
        })
        slot["federated" if r["family"] == "federated" else "local_only"] = {
            "id": r["id"], "slug": r["slug"], "label": r["label"],
            "accuracy": r["headline"]["accuracy"],
            "macro_f1": r["headline"]["macro_f1"],
            "degraded": r["health"]["degraded"],
        }
        part = partitions.get(r["slug"])
        if part:
            slot["heterogeneity_js"] = part.get("heterogeneity_js")
            slot["partition_source"] = part.get("source")

    for slot in keyed.values():
        fed, loc = slot["federated"], slot["local_only"]
        if fed and loc and fed["macro_f1"] is not None and loc["macro_f1"] is not None:
            slot["gain"] = fed["macro_f1"] - loc["macro_f1"]

    return sorted(
        keyed.values(),
        key=lambda s: (s["is_legacy"], -(s["heterogeneity_js"] or 0)),
    )


def build(results_dir: Path) -> dict:
    """Everything the static views need, in one response."""
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("*/*/metrics.json"))
    key = tuple((str(p), p.stat().st_mtime_ns) for p in files)
    if _cache.get("key") == key:
        return _cache["payload"]

    warnings: list[str] = []
    runs: list[dict] = []
    for path in files:
        raw = _load(path)
        if not raw or "split_fingerprint" not in raw:
            continue
        family = path.parent.parent.name
        artifacts = sorted(p.name for p in path.parent.glob("*.png"))
        runs.append(normalize_run(raw, family, artifacts))

    runs.sort(key=lambda r: (FAMILY_ORDER.index(r["family"])
                             if r["family"] in FAMILY_ORDER else 9, r["slug"]))

    # Every experiment must be scored on the same test set, or none of the
    # comparisons mean anything. Surface it rather than assuming it.
    shas = {r["id"]: (json.loads((results_dir / r["family"] / r["slug"] / "metrics.json")
                                 .read_text(encoding="utf-8"))
                      .get("split_fingerprint", {}).get("test_idx_sha1"))
            for r in runs}
    distinct = {s for s in shas.values() if s}
    if len(distinct) > 1:
        warnings.append(
            "Runs were scored on DIFFERENT test splits "
            f"({', '.join(sorted(distinct))}) — comparisons are not valid."
        )

    split = {}
    if runs:
        first = _load(results_dir / runs[0]["family"] / runs[0]["slug"] / "metrics.json")
        split = (first or {}).get("split_fingerprint", {})
    split["consistent_across_runs"] = len(distinct) <= 1

    degraded = [r["id"] for r in runs if r["health"]["degraded"]]
    if degraded:
        warnings.append(
            "Some runs had rounds where every client failed, leaving the global "
            f"model unchanged: {', '.join(degraded)}."
        )

    partitions = _partitions(results_dir, runs, warnings)
    payload = {
        "schema_version": 1,
        "classes": list(CLASSES),
        "palette": {"classes": CLASS_COLORS, "methods": METHOD_COLORS,
                    "ink": INK, "ink_muted": INK_MUTED, "grid": GRID},
        "split": split,
        "runs": runs,
        "partitions": partitions,
        "comparisons": _comparisons(runs, partitions),
        "warnings": warnings,
    }
    _cache["key"], _cache["payload"] = key, payload
    return payload
