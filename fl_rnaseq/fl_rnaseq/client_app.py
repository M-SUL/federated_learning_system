"""Flower ClientApp for federated RNA-seq classification.

Note what does and does not leave the client: `train_fn` returns an ArrayRecord
of model parameters and a sample count. The expression matrix itself never
leaves — that property is the reason for federating rather than pooling.

Clients optionally attach a "site" ConfigRecord describing their shard (size,
label histogram). That is **dashboard telemetry, not part of FedAvg** — see
`_site_record` — and `emit-site-telemetry = false` removes it entirely.
"""

import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_rnaseq.dataset import CLASSES, NUM_CLASSES, load_data
from fl_rnaseq.task import MLP, test, train
from fl_rnaseq.utils import set_seed

app = ClientApp()


def _round_of(msg: Message) -> int:
    """Server round this message belongs to.

    FedAvg injects it into the config record (`fedavg.py:180` for train,
    `:294` for evaluate). It is deliberately NOT read from
    ``msg.metadata.group_id``: the message-based strategy never sets group_id
    (``FedAvg._construct_messages`` passes only content/type/dst_node_id), so it
    defaults to "" and every round would silently resolve to 0. The legacy
    ``flwr.server`` workflows do set it, which is where that assumption came from.
    """
    try:
        return int(msg.content["config"]["server-round"])
    except (KeyError, TypeError, ValueError):
        return 0


def _seed_client(msg: Message, context: Context) -> int:
    """Seed this client's RNG deterministically. Returns the server round.

    Each ClientApp invocation runs in its own simulation worker process, which
    starts with unseeded RNG state -- so without this, dropout masks and
    DataLoader shuffling differ between otherwise identical runs.

    The seed mixes in the round so a client does not replay the identical
    dropout pattern every round, while staying deterministic across runs.
    """
    rnd = _round_of(msg)
    base = int(context.run_config["seed"])
    pid = int(context.node_config["partition-id"])
    set_seed(base * 100_000 + pid * 1_000 + rnd)
    return rnd


def _loaders(context: Context):
    """Build this client's train/val loaders from the run and node config."""
    cfg = context.run_config
    return load_data(
        partition_id=context.node_config["partition-id"],
        num_partitions=context.node_config["num-partitions"],
        data_dir=cfg["data-dir"],
        batch_size=int(cfg["batch-size"]),
        partition_strategy=cfg["partition-strategy"],
        dirichlet_alpha=float(cfg["dirichlet-alpha"]),
        seed=int(cfg["seed"]),
        min_partition_size=int(cfg["partition-min-size"]),
        self_balancing=bool(cfg["dirichlet-self-balancing"]),
    )


def _site_record(
    context: Context, rnd: int, phase: str, trainloader, valloader, seconds: float
) -> ConfigRecord | None:
    """Describe this site for the live dashboard. Not used by aggregation.

    Returns None when ``emit-site-telemetry`` is false, making the reply
    byte-identical to a run without the dashboard.

    A ConfigRecord rather than extra MetricRecord keys, for three reasons:
      - MetricRecord rejects str and bool, so it could not carry class names.
      - Every extra MetricRecord key is weight-averaged into the saved results
        by `aggregate_metricrecords`; a weighted mean of `partition_id` is noise.
      - `validate_message_reply_consistency` counts only metric/array records and
        `aggregate_metricrecords` iterates only metric_records, so a ConfigRecord
        passes through untouched. The FedAvg math is provably unchanged.

    This telemetry is itself a disclosure -- a label histogram is exactly the
    kind of distributional summary federation is meant to avoid sharing. It is
    simulation-only and off by one config flag; the privacy ledger says so.
    """
    if not bool(context.run_config.get("emit-site-telemetry", True)):
        return None

    labels = trainloader.dataset.tensors[1]
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    return ConfigRecord({
        "partition_id": int(context.node_config["partition-id"]),
        "num_partitions": int(context.node_config["num-partitions"]),
        "round": rnd,
        "phase": phase,
        "n_train": len(trainloader.dataset),
        "n_val": len(valloader.dataset),
        "label_classes": list(CLASSES),
        "label_counts": counts.tolist(),
        "n_classes_seen": int((counts > 0).sum()),
        "seconds": round(seconds, 4),
    })


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    rnd = _seed_client(msg, context)
    model = MLP(norm=cfg["norm"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, valloader = _loaders(context)
    t0 = time.perf_counter()
    train_loss = train(
        model, trainloader, int(cfg["local-epochs"]), float(cfg["learning-rate"]), device
    )
    elapsed = time.perf_counter() - t0

    records = {
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": train_loss, "num-examples": len(trainloader.dataset)}),
    }
    site = _site_record(context, rnd, "train", trainloader, valloader, elapsed)
    if site is not None:
        records["site"] = site
    return Message(content=RecordDict(records), reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    rnd = _seed_client(msg, context)
    model = MLP(norm=cfg["norm"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, valloader = _loaders(context)
    t0 = time.perf_counter()
    eval_loss, eval_acc = test(model, valloader, device)
    elapsed = time.perf_counter() - t0

    records = {
        "metrics": MetricRecord({
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }),
    }
    site = _site_record(context, rnd, "evaluate", trainloader, valloader, elapsed)
    if site is not None:
        records["site"] = site
    return Message(content=RecordDict(records), reply_to=msg)
