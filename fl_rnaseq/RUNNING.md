# Running the system

Operator's manual: how to reproduce every number from a fresh clone. For what the
project *is*, see [README.md](README.md).

---

## 1. Prerequisites

Windows 11, Python 3.11. **Always name the interpreter explicitly:**

```powershell
$PY = "C:\Program Files\python311\python.exe"
```

> **The `python` trap.** Bare `python` on this machine resolves to an unrelated,
> empty virtualenv (`C:\Users\m_sul\.virtualenvs\code-WbJ7r2pu`) that has none of
> the project's packages. It fails with `ModuleNotFoundError: numpy`, which looks
> like a broken install rather than the wrong interpreter. Every command below
> uses `$PY` or an activated venv — never bare `python`.

`verify_env.py` prints `sys.executable` first and refuses to run from that venv.

---

## 2. Data setup

The expression matrices are ~400 MB and gitignored, so this section *is* the
reproduction path.

1. Download **TCGA-PANCAN-HiSeq-801x20531** from the UCI ML Repository:
   <https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq>
2. Extract so the layout is:
   ```
   federated_learning_system/
     gene+expression+cancer+rna+seq/
       TCGA-PANCAN-HiSeq-801x20531/
         data.csv      # 801 x 20531, raw
         labels.csv    # 801 class labels
     fl_rnaseq/        # this package
   ```
3. **Generate `data_pruned.csv`.** Run `notebooks/01_eda.ipynb` through the
   *Prune constant genes* cell. It drops the 267 genes with zero variance,
   leaving **801 x 20264**, and writes `data_pruned.csv` beside `data.csv`.

   > Ordering quirk: the notebook's load cell reads `data_pruned.csv`, which that
   > later cell creates. On a clean checkout, point the load cell at `data.csv`
   > for the first run, then switch it back.

4. Confirm:
   ```powershell
   cd fl_rnaseq
   & $PY -m scripts.verify_env
   ```
   This must pass before anything else. It checks the interpreter, package
   versions, the `(801, 20264)` shape, label encoding, split determinism and
   stratification, scaler leakage, a forward pass, and the partition invariants.
   `flwr` warnings are expected until step 3 below.

---

## 3. Environment

**Phases 1–2 below need no installation** — they run on the packages already
present (torch 2.0.1+cpu, scikit-learn, numpy, pandas, matplotlib). Only the
Flower simulation needs a venv.

> **Create the venv in the repo root, NOT inside `fl_rnaseq/`.** `flwr run .`
> packages the app directory into a bundle, and a `.venv` inside it fails with
> *"exceeds the maximum directory depth of 10"* — Ray ships example files nested
> far deeper than that. `[tool.hatch.build] exclude` does not prevent this.

```powershell
cd <repo root>                   # federated_learning_system/, the PARENT of fl_rnaseq/
& $PY -m venv .venv
.\.venv\Scripts\Activate.ps1     # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ./fl_rnaseq
```

Notes:
- `--system-site-packages` does **not** help. torch/sklearn live in the *user*
  site directory, and venvs disable user-site — the venv genuinely re-downloads
  them. Budget ~1.5 GB and 10–20 minutes.
- Install torch first from the CPU index so the resolver cannot pull a CUDA build.
- `numpy` is pinned `<2.0`: `datasets` → `pyarrow` may pull numpy 2.x, whose ABI
  break stops torch 2.0.1 loading with `_ARRAY_API not found`.
- Installing pulls **scikit-learn 1.9**, newer than the 1.4.2 in user site. The
  split fingerprint is identical across both (`cb67241e1200`), but for a clean
  write-up run every experiment from the venv rather than mixing interpreters.

### Federation config lives outside pyproject.toml (Flower ≥ 1.31)

Flower moved SuperLink connection settings out of `[tool.flwr.federations]` into
a global config file. The first `flwr run` migrates them automatically, comments
out the old block in `pyproject.toml`, and writes `~/.flwr/config.toml`:

```toml
[superlink]
default = "local-simulation"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 5
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
```

`options.num-supernodes` is what supplies `node_config["num-partitions"]` to each
client, so it must equal `num-partitions` in `[tool.flwr.app.config]` —
`server_app.py` raises if they disagree.

---

## 4. Which commands need the venv

| Phase | Command | venv? |
|---|---|---|
| 0 | `scripts.verify_env` | no |
| 1 | `scripts.run_centralized_baseline` | no |
| 2 | `scripts.run_local_only_baseline` (`iid`, `dirichlet-legacy`) | no |
| 3 | `scripts.plot_partition_distribution` | only for `dirichlet` rows |
| 4 | `flwr run .` | **yes** |
| 5 | `scripts.make_report` | no |

This is deliberate: a broken Flower or Ray install never blocks the baselines,
which are the two reference points the whole comparison rests on.

---

## 5. Running the experiments

Run everything from `fl_rnaseq/`. Add `-u` to see progress live rather than at exit.

### Phase 1 — Centralized baseline (the upper bound)

```powershell
& $PY -u -m scripts.run_centralized_baseline
```
~1 min for L2 logistic regression, plus several minutes for the L1/saga fit and
~15 min for the two MLP budgets. Skip the slow parts with `--skip-l1 --skip-mlp`.

Writes `results/centralized/{logreg,logreg_l1,mlp_30ep,mlp_6ep}/`.

### Phase 2 — Local-only baseline (the lower bound)

```powershell
& $PY -u -m scripts.run_local_only_baseline --strategy iid
& $PY -u -m scripts.run_local_only_baseline --strategy dirichlet-legacy --alpha 0.5
& $PY -u -m scripts.run_local_only_baseline --strategy dirichlet-legacy --alpha 0.1
```
~10 min each (5 clients x 30 epochs on CPU). Writes `results/local_only/<slug>/`.

Use `--strategy dirichlet` instead of `dirichlet-legacy` once the venv exists.

### Phase 3 — Partition characterisation

```powershell
& $PY -u -m scripts.plot_partition_distribution
```
Seconds. Writes the client x class heatmaps, per-config shard sizes, and the
Jensen-Shannon heterogeneity index to `results/partitions/`.

### Phase 4 — Federated runs (venv required)

```powershell
flwr run . local-simulation
flwr run . local-simulation --run-config "partition-strategy='dirichlet' dirichlet-alpha=1.0"
flwr run . local-simulation --run-config "partition-strategy='dirichlet' dirichlet-alpha=0.5"
flwr run . local-simulation --run-config "partition-strategy='dirichlet' dirichlet-alpha=0.1"
flwr run . local-simulation --run-config "partition-strategy='dirichlet-legacy' dirichlet-alpha=0.5"
flwr run . local-simulation --run-config "partition-strategy='dirichlet' dirichlet-alpha=0.5 norm='layer'"
```
Repeat key rows with `seed=43` and `seed=44` and report mean ± std. Vary only the
partition/init seed — the global train/test split is pinned to `SPLIT_SEED` in
`dataset.py` and must never move.

Writes `results/federated/<slug>/`.

### Phase 5 — Report

```powershell
& $PY -u -m scripts.make_report
```
Asserts every experiment shares one test split, then writes
`results/summary/{summary.csv, headline_comparison.png, alpha_sweep.png}`.

---

## 6. Expected results

| Experiment | Accuracy | macro-F1 | Note |
|---|---|---|---|
| Centralized logreg (L2) | **0.9938** | **0.9947** | best C = 0.1; exactly 1 error in 161 |
| Local-only, IID | 0.9839 ± 0.0050 | 0.9814 ± 0.0062 | isolation costs little when data is IID |
| Local-only, α=0.1 | *collapses* | *collapses far further* | clients see only 2–3 of 5 classes |
| Federated, IID | ≈ centralized | ≈ centralized | expected to be statistically indistinguishable |

**Read macro-F1, not accuracy.** The five tumour types are near-linearly
separable, so accuracy saturates near 1.0 and the centralized-vs-federated gap is
smaller than one test sample (0.6%). The findings that survive scrutiny are
macro-F1 degradation as α falls, the collapse of isolated clients, and the
~4.2 GB communication cost.

> **The confusion matrices are mostly zeros off the diagonal — this is correct,
> not a bug.** At 99.4% accuracy there is one misclassified sample in the whole
> test set. Cells are annotated with raw counts precisely so an empty cell reads
> as "0 samples" rather than an unformatted `0.00`.

**Tripwire (V6):** centralized logistic regression below **0.95** accuracy means
the label encoding or the scaling is broken. Stop and debug — do not run anything
else, because every other number is measured against this one.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: numpy` | Bare `python` hit the empty venv. Use `$PY`. |
| `could not convert string to float: 'sample_0'` | A blanket `dtype=` is being applied before `index_col`. `dataset.py` scopes the dtype map to gene columns only. |
| `mat1 and mat2 shapes cannot be multiplied (...20264 and 20531...)` | Stale `NUM_FEATURES`. It is 20264 (pruned), and `_load_and_preprocess` raises a clear error if the CSV disagrees. |
| `Expected data_pruned.csv to be (801, 20264)` | The pruned matrix is missing or stale. Re-run the notebook's prune cell. |
| `_ARRAY_API not found` | numpy 2.x against torch 2.0.1. Reinstall with the `numpy<2` pin. |
| `Expected more than 1 value per channel` | A BatchNorm training batch of size 1. `load_data` drops such a tail batch automatically; if it persists, raise `--batch-size` or `--min-partition-size`. |
| `Federation has N supernodes but num-partitions=M` | `options.num-supernodes` and `num-partitions` disagree. They must match, or the baselines and the federation would use different shards. |
| `DirichletPartitioner` raises at α=0.1 | It cannot satisfy `min_partition_size` on 640 samples. Lower `partition-min-size`, or record the infeasibility as a finding. (Observed to succeed at α=0.1 with a floor of 32, producing shards `[167, 267, 90, 44, 72]`.) |
| **`'charmap' codec can't encode character '\U0001f338'`** | The `flwr` CLI prints an emoji to a cp1252 console. Set `$env:PYTHONIOENCODING="utf-8"` and `$env:PYTHONUTF8="1"` before running. |
| **`exceeds the maximum directory depth of 10`** | A `.venv` inside `fl_rnaseq/` is being bundled into the FAB, and Ray's example files nest deeper than the limit. Put the venv in the repo root instead (§3). |
| **`Unable to launch flower-superlink: [WinError 2]`** | The venv's `Scripts/` is not on `PATH`. Activate the venv rather than calling `.venv-fl\Scripts\flwr.exe` directly. |
| `Device or resource busy` when deleting a venv | Orphaned `flower-superlink.exe` / `flower-superexec.exe` from a failed run still hold the files. `tasklist /FI "IMAGENAME eq flower-superlink.exe"`, then `taskkill /PID <pid> /F`. |
| Ray fails / `flwr run` hangs on Windows | Use the Ray-free deployment runtime (below). |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process RemoteSigned` |

### Ray-free fallback

Flower's simulation mode uses Ray, which is the least reliable dependency on
Windows. The `local-deployment` federation avoids it entirely — five real OS
processes over gRPC, which is arguably the more convincing demo anyway:

```powershell
# terminal 1
flower-superlink --insecure
# terminals 2-6, one per client (i = 0..4)
flower-supernode --insecure --superlink 127.0.0.1:9092 `
  --clientappio-api-address 127.0.0.1:906$i `
  --node-config "partition-id=$i num-partitions=5"
# terminal 7
flwr run . local-deployment
```

---

## 8. Reproducing a specific figure

| Figure | Command |
|---|---|
| `centralized/logreg/cv_curve.png` | `scripts.run_centralized_baseline` |
| `centralized/*/confusion_matrix.png` | `scripts.run_centralized_baseline` |
| `local_only/<slug>/per_client_performance.png` | `scripts.run_local_only_baseline --strategy ... --alpha ...` |
| `local_only/<slug>/confusion_client*.png` | same as above |
| `partitions/*.png`, `partitions/heterogeneity.png` | `scripts.plot_partition_distribution` |
| `federated/<slug>/rounds.png` | `flwr run . local-simulation --run-config "..."` |
| `summary/headline_comparison.png`, `summary/alpha_sweep.png` | `scripts.make_report` |
