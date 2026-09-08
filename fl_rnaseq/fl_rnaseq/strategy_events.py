"""FedAvg wrapped to emit per-client events, without changing what it computes.

Per-client data exists in exactly one place server-side: the `replies` passed to
`aggregate_train` / `aggregate_evaluate`. `train_metrics_aggr_fn` receives
`list[RecordDict]` with node identity stripped, and the `Result` returned by
`start()` holds only aggregated metrics -- neither can attribute anything to a
client. Hence this subclass.

Every override is thin: capture, delegate to super(), return super()'s value
unchanged. The aggregation math is untouched, and a test asserts the results are
bit-identical to stock FedAvg.
"""

from __future__ import annotations

import time
from logging import WARNING
from typing import Any, Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fl_rnaseq.events import EventEmitter


def _record_to_dict(record: Any) -> dict:
    """Copy a Metric/ConfigRecord into a plain dict. Tolerates absence."""
    if record is None:
        return {}
    try:
        return {k: record[k] for k in record}
    except BaseException:  # noqa: BLE001 - telemetry must never raise
        return {}


class FedAvgWithEvents(FedAvg):
    """FedAvg that reports what each client sent, to an EventEmitter."""

    def __init__(self, *args: Any, emitter: EventEmitter, site_key: str = "site", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._em = emitter
        self._site_key = site_key
        self._t_phase: dict[tuple[int, str], float] = {}
        # Rounds where every client reply failed. Stock FedAvg treats this as a
        # no-op -- it keeps the previous arrays and carries on -- so a run whose
        # clients all died produces a frozen but entirely plausible-looking
        # result file. Observed for real: five concurrent simulations exhausted
        # memory, Ray workers died importing sklearn, and the global model sat
        # unchanged for nine rounds at an accuracy that looked like slow
        # convergence. The server must be able to say that happened.
        self.failed_rounds: list[int] = []
        self.n_client_errors = 0

    # -- fan-out -----------------------------------------------------------

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        msgs = list(super().configure_train(server_round, arrays, config, grid))
        self._t_phase[(server_round, "train")] = time.perf_counter()
        self._em.emit(
            "round_start",
            round=server_round,
            phase="train",
            node_ids=[m.metadata.dst_node_id for m in msgs],
            config=_record_to_dict(config),
        )
        return msgs

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        msgs = list(super().configure_evaluate(server_round, arrays, config, grid))
        self._t_phase[(server_round, "evaluate")] = time.perf_counter()
        self._em.emit(
            "eval_fanout",
            round=server_round,
            phase="evaluate",
            node_ids=[m.metadata.dst_node_id for m in msgs],
        )
        return msgs

    # -- fan-in ------------------------------------------------------------

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # Materialize FIRST, once, and hand the SAME list to super(). `replies`
        # is annotated Iterable; consuming it here without this would leave
        # super() with an exhausted generator and silently aggregate nothing.
        replies = list(replies)
        self._emit_replies(server_round, replies, "train")
        arrays, metrics = super().aggregate_train(server_round, replies)
        fields = self._agg_fields(replies, metrics)
        if replies and fields["n_ok"] == 0:
            self.failed_rounds.append(server_round)
            log(
                WARNING,
                "Round %s: all %s client replies failed — the global model is "
                "UNCHANGED this round. First reason: %s",
                server_round, fields["n_error"], self._first_reason(replies),
            )
        self._em.emit("agg_train", round=server_round, **fields)
        return arrays, metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        replies = list(replies)
        self._emit_replies(server_round, replies, "evaluate")
        metrics = super().aggregate_evaluate(server_round, replies)
        self._em.emit("agg_eval", round=server_round, **self._agg_fields(replies, metrics))
        return metrics

    # -- helpers -----------------------------------------------------------

    def _agg_fields(self, replies: list[Message], metrics: MetricRecord | None) -> dict:
        n_error = sum(1 for m in replies if m.has_error())
        self.n_client_errors += n_error
        return {
            "n_replies": len(replies),
            "n_ok": len(replies) - n_error,
            "n_error": n_error,
            "aggregated_metrics": _record_to_dict(metrics),
        }

    @staticmethod
    def _first_reason(replies: list[Message]) -> str:
        """Shortened first error reason — Ray tracebacks run to thousands of chars."""
        for m in replies:
            if m.has_error():
                reason = (getattr(m.error, "reason", "") or "").strip().replace("\n", " ")
                return reason[:200] + ("…" if len(reason) > 200 else "")
        return "unknown"

    def _emit_replies(self, server_round: int, replies: list[Message], phase: str) -> None:
        """One event per client reply. Iterates the whole list unconditionally.

        No early return and no short-circuit: a branch that skipped ahead could
        leave super() with a partly-drained iterator on some future Grid whose
        replies are lazy.
        """
        t0 = self._t_phase.get((server_round, phase))
        for msg in replies:
            md = msg.metadata
            base = {
                "round": server_round,
                "phase": phase,
                "node_id": md.src_node_id,
                "server_elapsed_s": (time.perf_counter() - t0) if t0 else None,
            }

            # `Message.content` raises ValueError on an error-carrying message,
            # so has_error() must be checked before touching it.
            if msg.has_error():
                err = msg.error
                self._em.emit(
                    f"client_{phase}", ok=False, site_id=None, metrics={}, site={},
                    error={"code": getattr(err, "code", None),
                           "reason": getattr(err, "reason", None)},
                    **base,
                )
                continue

            content = msg.content
            metrics = next(iter(content.metric_records.values()), None)
            site = content.config_records.get(self._site_key)
            site_d = _record_to_dict(site)
            self._em.emit(
                f"client_{phase}", ok=True, error=None,
                site_id=site_d.get("partition_id"),
                metrics=_record_to_dict(metrics),
                site=site_d,
                **base,
            )
