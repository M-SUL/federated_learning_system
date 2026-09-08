"""Flower ServerApp: global model init, FedAvg strategy, centralised evaluation.

Server-side evaluation records macro-F1 and a confusion matrix every round, not
just accuracy: with an imbalance ratio of 3.85 and clients that may never see a
class under non-IID partitioning, accuracy hides the very failure mode this
project is measuring.
"""

from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_rnaseq.dataset import load_test_data, split_fingerprint
from fl_rnaseq.metrics import evaluate_model, plot_confusion_matrix, plot_metric_curves
from fl_rnaseq.task import MLP
from fl_rnaseq.utils import (
    communication_cost, result_envelope, run_slug, set_seed, write_result,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    num_rounds = int(cfg["num-server-rounds"])
    lr = float(cfg["learning-rate"])
    seed = int(cfg["seed"])
    strategy_name = cfg["partition-strategy"]
    alpha = None if strategy_name == "iid" else float(cfg["dirichlet-alpha"])

    # A silent mismatch here would mean the baselines and the federation use
    # different shards, invalidating every comparison — and it is otherwise
    # invisible in the output. Fail loudly instead.
    n_nodes = len(list(grid.get_node_ids()))
    if n_nodes != int(cfg["num-partitions"]):
        raise ValueError(
            f"Federation has {n_nodes} supernodes but num-partitions={cfg['num-partitions']}. "
            "Set options.num-supernodes in [tool.flwr.federations.local-simulation] "
            "to match, or the baselines and this run will partition the data differently."
        )

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global_model = MLP(norm=cfg["norm"])
    arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(fraction_evaluate=float(cfg["fraction-evaluate"]))

    comms = communication_cost(global_model, n_nodes, num_rounds)
    print(
        f"[setup] {comms['params']:,} params · {comms['mb_per_model']} MB/model · "
        f"{comms['total_gb']} GB total traffic over {num_rounds} rounds"
    )

    history: list[dict] = []
    last_report: dict = {}

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the global model on the held-out centralised test set."""
        nonlocal last_report
        model = MLP(norm=cfg["norm"])
        model.load_state_dict(arrays.to_torch_state_dict())

        loader = load_test_data(cfg["data-dir"], batch_size=int(cfg["batch-size"]))
        report = evaluate_model(model, loader, device)
        last_report = report

        history.append({
            "round": server_round,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "loss": report["loss"],
        })
        print(
            f"[round {server_round}] test-set eval — loss: {report['loss']:.4f}  "
            f"acc: {report['accuracy']:.4f}  macro-F1: {report['macro_f1']:.4f}"
        )
        # MetricRecord holds scalars only; the confusion matrix stays in `history`.
        return MetricRecord({
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "loss": report["loss"],
        })

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    slug = run_slug(strategy_name, alpha, seed, norm=cfg["norm"])
    out_dir = Path(cfg["results-dir"]) / "federated" / slug

    payload = result_envelope(
        experiment="federated_fedavg",
        slug=slug,
        config={
            "strategy": strategy_name, "alpha": alpha, "num_rounds": num_rounds,
            "num_clients": n_nodes, "local_epochs": int(cfg["local-epochs"]),
            "lr": lr, "batch_size": int(cfg["batch-size"]), "seed": seed,
            "norm": cfg["norm"], "min_partition_size": int(cfg["partition-min-size"]),
            "self_balancing": bool(cfg["dirichlet-self-balancing"]),
            "data_dir": cfg["data-dir"],
        },
        split_fingerprint=split_fingerprint(cfg["data-dir"]),
        model_info=comms,
        metrics=last_report,
        history=history,
    )
    write_result(payload, out_dir)

    if history:
        plot_metric_curves(
            history, ["accuracy", "macro_f1"], out_dir / "rounds.png",
            title=f"Global model over rounds — {slug}",
        )
    if last_report:
        plot_confusion_matrix(
            last_report["confusion_matrix"], out_dir / "confusion_matrix.png",
            title=f"FedAvg final model — {slug}",
        )
    print(f"[done] results written to {out_dir}")

    if cfg.get("save-model"):
        torch.save(result.arrays.to_torch_state_dict(), out_dir / "final_model.pt")
        print(f"Saved {out_dir / 'final_model.pt'}")
