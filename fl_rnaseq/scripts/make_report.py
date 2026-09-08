"""Aggregate every results/**/metrics.json into the comparison table and figures.

Before building anything, this asserts that every experiment shares one global
train/test split, by comparing ``split_fingerprint.test_idx_sha1``. Comparing a
centralized number against a federated one computed on a different test set
would be the most embarrassing possible error in the write-up, and it is
otherwise invisible — so it is a hard failure here, not a warning.

    & "C:\\Program Files\\python311\\python.exe" -m scripts.make_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fl_rnaseq.metrics import GRID, INK, INK_MUTED

# Fixed display order and color per method family — assigned by identity, never
# by rank, so adding a run never repaints the others.
FAMILY_ORDER = ["centralized", "federated", "local_only"]
FAMILY_COLORS = {
    "centralized": "#0072B2",
    "federated": "#009E73",
    "local_only": "#D55E00",
}


def load_all(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  skipping unreadable {path}: {exc}")
            continue
        if "split_fingerprint" not in payload:
            continue  # not an experiment file
        payload["_path"] = path
        payload["_family"] = path.relative_to(results_dir).parts[0]
        rows.append(payload)
    return rows


def assert_one_split(rows: list[dict]) -> str:
    """Hard gate: every experiment must be scored on the same test set."""
    seen: dict[str, list[str]] = {}
    for r in rows:
        sha = r["split_fingerprint"].get("test_idx_sha1", "<missing>")
        seen.setdefault(sha, []).append(str(r["_path"]))

    if len(seen) > 1:
        print("\nFATAL: experiments were scored on different test splits.\n")
        for sha, paths in seen.items():
            print(f"  {sha}:")
            for p in paths:
                print(f"    {p}")
        raise SystemExit(
            "\nRe-run the affected experiments against one split before reporting. "
            "SPLIT_SEED in dataset.py must not have changed between runs."
        )
    return next(iter(seen))


def to_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        m, cfg = r["metrics"], r.get("config", {})
        per_class = m.get("per_class", {})
        records.append({
            "family": r["_family"],
            "experiment": r["experiment"],
            "slug": r["slug"],
            "strategy": cfg.get("strategy", "-"),
            "alpha": cfg.get("alpha"),
            "seed": cfg.get("seed"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "coad_recall": per_class.get("COAD", {}).get("recall"),
            "heterogeneity_js": m.get("heterogeneity_js"),
            "params": r.get("model", {}).get("params") or r.get("model", {}).get("num_params"),
            "mb_per_model": r.get("model", {}).get("mb_per_model"),
            "total_gb": r.get("model", {}).get("total_gb"),
        })
    df = pd.DataFrame(records)
    df["_order"] = df["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(9)
    return df.sort_values(["_order", "slug"]).drop(columns="_order").reset_index(drop=True)


def plot_headline(df: pd.DataFrame, path: Path) -> None:
    """Accuracy and macro-F1 per experiment, grouped by family.

    Horizontal bars: the experiment names are long, and rotating them or packing
    16 vertical groups makes the labels collide. The `dirichlet-legacy` runs are
    excluded — they are a cross-implementation check on a different partitioner,
    and interleaving them here obscures the comparison this figure exists for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["accuracy", "macro_f1"])
    d = d[d["strategy"] != "dirichlet-legacy"]
    if d.empty:
        return
    d = d.iloc[::-1]  # barh draws bottom-up; keep the table's reading order

    labels = [f"{r.family.replace('_', '-')} · {r.slug.replace('_seed42', '')}"
              for r in d.itertuples()]
    y = np.arange(len(d))
    height = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(d) + 1.6))
    # Two measures on one shared 0-1 scale — never a second axis.
    ax.barh(y + height / 2, d["accuracy"], height * 0.92, color="#0072B2", label="accuracy")
    ax.barh(y - height / 2, d["macro_f1"], height * 0.92, color="#E69F00", label="macro-F1")

    for yi, (a, f) in enumerate(zip(d["accuracy"], d["macro_f1"])):
        ax.text(a + 0.008, yi + height / 2, f"{a:.3f}", va="center", fontsize=7.5, color=INK)
        ax.text(f + 0.008, yi - height / 2, f"{f:.3f}", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("score on the global test set", color=INK)
    ax.set_title("Centralized vs federated vs isolated", color=INK, fontsize=11, pad=10)

    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, ncol=2,
              loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_sweep(df: pd.DataFrame, path: Path) -> None:
    """Macro-F1 against Dirichlet alpha, with the centralized ceiling as a reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    swept = df.dropna(subset=["alpha", "macro_f1"])
    # Only the flwr-datasets "dirichlet" partitioner belongs on the sweep: local-only
    # and federated share those exact shards, so the vertical gap between the two
    # lines is the effect of aggregation alone. The hand-rolled "dirichlet-legacy"
    # runs use different shards and would plot as a spurious second point per alpha.
    swept = swept[swept["strategy"] == "dirichlet"]
    if swept.empty:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ceiling = df[df["family"] == "centralized"]["macro_f1"].max()
    if pd.notna(ceiling):
        ax.axhline(ceiling, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"centralized ceiling {ceiling:.3f}", (ax.get_xlim()[0], ceiling),
                    textcoords="offset points", xytext=(4, 5), fontsize=8, color=INK)

    for family, grp in swept.groupby("family"):
        grp = grp.sort_values("alpha")
        ax.plot(grp["alpha"], grp["macro_f1"], linewidth=2, marker="o", markersize=6,
                color=FAMILY_COLORS.get(family, "#666666"),
                label=family.replace("_", "-"))
        for r in grp.itertuples():
            ax.annotate(f"{r.macro_f1:.3f}", (r.alpha, r.macro_f1),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet alpha  (lower = more heterogeneous)", color=INK)
    ax.set_ylabel("macro-F1 on the global test set", color=INK)
    ax.set_title("Degradation under label heterogeneity", color=INK, fontsize=11, pad=10)
    _style(ax)
    if swept["family"].nunique() >= 2:
        ax.legend(frameon=False, labelcolor=INK, fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = load_all(results_dir)
    if not rows:
        print(f"No experiment results under {results_dir}. Run the baselines first.")
        return 1

    sha = assert_one_split(rows)
    print(f"\n{len(rows)} experiments, all on split {sha}\n")

    df = to_frame(rows)
    out = results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    show = df[["family", "slug", "accuracy", "macro_f1", "coad_recall", "heterogeneity_js"]]
    print(show.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4f}"))

    plot_headline(df, out / "headline_comparison.png")
    plot_alpha_sweep(df, out / "alpha_sweep.png")

    env_lines = [f"split: {sha}", ""]
    if rows:
        for k, v in rows[0].get("env", {}).items():
            env_lines.append(f"{k}: {v}")
    (out / "environment.txt").write_text("\n".join(env_lines), encoding="utf-8")

    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
