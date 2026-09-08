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

## Results

Global test set: 161 samples, stratified, identical across all 16 experiments
(fingerprint `cb67241e1200`, asserted by `make_report.py`).

### Centralized reference

| Model | Accuracy | Macro-F1 | Note |
|---|---|---|---|
| MLP, 30 epochs | **1.0000** | **1.0000** | the ceiling |
| Logistic regression, L2 (C=0.1) | 0.9938 | 0.9947 | 1 error in 161 |
| Logistic regression, L1 | 0.9938 | 0.9947 | **only 64–191 of 20264 genes per class** |
| MLP, 6 epochs | 0.9938 | 0.9947 | gradient-step-matched to FL |

### Federated vs isolated, on identical shards

| Partition (JS divergence) | Local-only macro-F1 | Federated macro-F1 | Gain |
|---|---|---|---|
| IID (0.005) | 0.9814 | **1.0000** | +0.019 |
| Dirichlet α=1.0 (0.115) | 0.8449 | **1.0000** | +0.155 |
| Dirichlet α=0.5 (0.152) | 0.9115 | **0.9866** | +0.075 |
| Dirichlet α=0.1 (0.373) | **0.4137** | **0.9585** | **+0.545** |

### What the numbers say

**The headline is not accuracy.** The five tumour types are near-linearly
separable, so a well-regularized linear model makes one error in 161 and any
sensible method saturates. The findings that survive scrutiny:

1. **Under IID splits federation buys almost nothing here** — an isolated client
   already reaches 0.981 macro-F1. An honest result, not a failure.
2. **Under realistic heterogeneity it buys a great deal.** At α=0.1 each client
   holds only 2–3 of the 5 tumour types and an isolated model collapses to 0.414
   macro-F1: it structurally cannot predict a class it never saw, which shows as
   empty columns in its confusion matrix. FedAvg over the same shards reaches
   0.959 — within 0.04 of the centralized ceiling.
3. **Isolated sites cannot detect this themselves.** Every client scores ~99.6%
   on its *own* validation data while scoring 55% globally — an optimism gap of
   **+0.39**. A site evaluating itself locally would conclude its model is
   excellent. That gap is the argument for federating.
4. **Federation is not free.** The MLP is 10.44 M parameters (41.8 MB), so 10
   rounds across 5 clients moves **~4.2 GB**. That the L1 model matches full
   accuracy on ~1 % of the genes suggests most of that traffic is unnecessary.

### Caveats

- Single seed (42) per configuration. The α=1.0 local-only point (0.845) scoring
  below α=0.5 (0.912) is single-draw noise — their JS divergences are 0.115 and
  0.152, close enough to reorder. Multi-seed runs are the obvious next step.
- Local-only and federated share partitions at each α; the `dirichlet-legacy`
  rows in `summary.csv` use the hand-rolled partitioner and are kept only as a
  cross-implementation check.

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
