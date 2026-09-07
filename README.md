# Federated Learning for RNA-seq Cancer Classification

Classifying five tumour types (BRCA, COAD, KIRC, LUAD, PRAD) from TCGA PANCAN
HiSeq gene-expression profiles, using [Flower](https://flower.ai) to train across
simulated clinical sites without pooling their data.

**To run anything, see [fl_rnaseq/RUNNING.md](fl_rnaseq/RUNNING.md).**

## The question

A federated accuracy figure means nothing on its own. This project measures it
against the two references that bracket it:

| | What it is | Role |
|---|---|---|
| **Centralized** | One model on the pooled training set — no privacy constraint | Upper bound |
| **Federated** | FedAvg across 5 clients; only model weights are shared | The system under test |
| **Local-only** | Each client trains alone, no communication | Lower bound federation must beat |

## Results so far

Global test set: 161 samples, stratified, identical across every experiment
(fingerprint `cb67241e1200`).

| Experiment | Accuracy | Macro-F1 | Local-val accuracy |
|---|---|---|---|
| Centralized logistic regression (L2, C=0.1) | **0.9938** | **0.9947** | — |
| Local-only, IID split | 0.9839 ± 0.0050 | 0.9814 ± 0.0062 | 0.9920 |
| Local-only, Dirichlet α=0.1 | 0.6025 ± 0.1442 | **0.4393** ± 0.1664 | 0.9956 |

**The headline is not accuracy.** The five tumour types are near-linearly
separable, so a well-regularized linear model makes one error in 161 and any
sensible method saturates near 1.0. What the numbers actually show:

1. **Under IID splits, federation buys almost nothing here** — isolated clients
   already reach 0.984. This is an honest finding, not a failure.
2. **Under realistic heterogeneity it buys a great deal.** At α=0.1 each client
   holds only 2–3 of the 5 tumour types, and an isolated model collapses to 0.44
   macro-F1 because it structurally cannot predict a class it never saw.
3. **Isolated sites cannot detect this themselves.** Every client scores ~99.6%
   on its *own* validation data while scoring 60% globally — an optimism gap of
   **+0.39**. That gap is the argument for federating.
4. **Federation is not free.** The MLP is 10.44 M parameters (41.8 MB), so 10
   rounds across 5 clients moves ~4.2 GB.

## Layout

```
fl_rnaseq/
  fl_rnaseq/
    dataset.py       # loading, stratified split, DataLoaders
    partitioning.py  # iid / dirichlet (flwr-datasets) / dirichlet-legacy
    task.py          # MLP, train, test
    metrics.py       # macro-F1, confusion matrices, figures
    utils.py         # seeding, results envelope, comms cost
    client_app.py    # Flower ClientApp
    server_app.py    # Flower ServerApp, FedAvg, centralised evaluation
  scripts/           # verify_env, baselines, partition plots, report
  notebooks/01_eda.ipynb
  results/           # metrics.json + figures, one directory per experiment
gene+expression+cancer+rna+seq/   # data (gitignored — see RUNNING.md)
```

## Method notes

- The global train/test split is stratified and pinned to `SPLIT_SEED = 42` in
  `dataset.py`. It is deliberately not parameterizable, and every results file
  carries a `test_idx_sha1` fingerprint; `make_report.py` refuses to build a
  comparison table unless all experiments share one split.
- The scaler is fit on training samples only.
- Constant genes (267 of 20531) are dropped in EDA, leaving 20264 features.
- Non-IID partitioning uses the standard `flwr-datasets` `DirichletPartitioner`,
  which enforces a minimum shard size; the original hand-rolled implementation
  is retained as `dirichlet-legacy` for cross-checking.
