"""Flower ClientApp for federated RNA-seq classification.

Note what does and does not leave the client: `train_fn` returns an ArrayRecord
of model parameters and a sample count. The expression matrix itself never
leaves — that property is the reason for federating rather than pooling.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_rnaseq.dataset import load_data
from fl_rnaseq.task import MLP, test, train
from fl_rnaseq.utils import set_seed

app = ClientApp()


def _seed_client(msg: Message, context: Context) -> None:
    """Make this client's local training reproducible.

    Each ClientApp invocation runs in its own simulation worker process, which
    starts with unseeded RNG state -- so without this, dropout masks and
    DataLoader shuffling differ between otherwise identical runs, and repeated
    runs of the same config disagree by ~0.01 macro-F1.

    The seed mixes in the round (Message.group_id) so a client does not replay
    the identical dropout pattern every round, while staying deterministic
    across runs.
    """
    try:
        rnd = int(msg.metadata.group_id)
    except (TypeError, ValueError):
        rnd = 0
    base = int(context.run_config["seed"])
    pid = int(context.node_config["partition-id"])
    set_seed(base * 100_000 + pid * 1_000 + rnd)


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


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    _seed_client(msg, context)
    model = MLP(norm=cfg["norm"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, _ = _loaders(context)
    train_loss = train(
        model, trainloader, int(cfg["local-epochs"]), float(cfg["learning-rate"]), device
    )

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": train_loss, "num-examples": len(trainloader.dataset)}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    _seed_client(msg, context)
    model = MLP(norm=cfg["norm"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, valloader = _loaders(context)
    eval_loss, eval_acc = test(model, valloader, device)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)
