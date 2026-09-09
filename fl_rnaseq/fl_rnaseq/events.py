"""Append-only event log for the live dashboard.

Flower exposes no event bus, callback list, or telemetry hook a dashboard could
subscribe to (`evaluate_fn` is the only per-round callback, and it carries no
per-client data), so the ServerApp emits its own. Events are JSON lines written
to ``<live-dir>/<run_id>.jsonl``; the web server tails that file. An append-only
file is the whole contract between producer and consumer -- no ports, no shared
memory, and a dashboard that starts late or restarts simply re-reads from the top.

**The emitter can never fail a training run.** Every public method swallows
BaseException and latches itself off after the first failure. This is what makes
"observe-only" a property of the code rather than an intention.

This module deliberately imports no flwr, so it is unit-testable with plain dicts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from logging import WARNING
from pathlib import Path
from typing import Any

from flwr.common.logger import log

from fl_rnaseq.utils import _jsonable

# Heartbeat cadence and the staleness threshold readers should use. A round takes
# 25-35 s here, so liveness can NOT be inferred from the JSONL's mtime -- any
# threshold tight enough to catch a crash would fire during a normal slow round.
# Hence a separate sentinel file touched on a fixed schedule.
HEARTBEAT_SECONDS = 5.0
STALE_AFTER_SECONDS = 20.0


def resolve_live_dir(run_config: dict, env: dict | None = None) -> Path:
    """Absolute directory for event logs. First hit wins.

    1. ``FL_RNASEQ_LIVE_DIR``
    2. ``live-dir`` run config, when non-empty
    3. ``<results-dir>/live``

    Rule 1 exists for ``local-deployment``: there the ServerApp is a superexec
    child whose CWD is the extracted FAB under ~/.flwr/apps/, so the relative
    ``results-dir`` would put events somewhere the dashboard never looks.
    """
    env = os.environ if env is None else env

    from_env = env.get("FL_RNASEQ_LIVE_DIR", "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()

    from_cfg = str(run_config.get("live-dir", "") or "").strip()
    if from_cfg:
        return Path(from_cfg).expanduser().resolve()

    return (Path(str(run_config.get("results-dir", "results"))).resolve() / "live")


class EventEmitter:
    """Writes one JSON object per line, plus a heartbeat sentinel.

    Args:
        run_id:   Unique per ``flwr run`` -- use ``context.run_id``, never
                  ``run_slug()``, which collides across repeats of a config by
                  design (one output directory per config).
        live_dir: Directory for ``<run_id>.jsonl`` and ``<run_id>.alive``.
        enabled:  False makes every method a no-op.
    """

    def __init__(self, run_id: int | str, live_dir: Path, enabled: bool = True):
        self._seq = 0
        self._lock = threading.Lock()
        self._disabled = not enabled
        self._fh = None
        self._hb: threading.Thread | None = None
        self._stop = threading.Event()

        self.path = Path(live_dir) / f"{run_id}.jsonl"
        self.alive_path = Path(live_dir) / f"{run_id}.alive"

        if self._disabled:
            return
        try:
            Path(live_dir).mkdir(parents=True, exist_ok=True)
            # newline="" so json lines are terminated by exactly one "\n" on
            # Windows too; a "\r\n" would still parse, but the tailing reader
            # commits its file position only on a "\n"-terminated readline and
            # keeping the bytes predictable makes that easier to reason about.
            self._fh = open(self.path, "a", encoding="utf-8", newline="")
            self.alive_path.touch()
            self._hb = threading.Thread(target=self._heartbeat, daemon=True)
            self._hb.start()
        except BaseException as exc:  # noqa: BLE001 - never fail the run
            self._latch(exc, "open")

    # -- internals ---------------------------------------------------------

    def _latch(self, exc: BaseException, where: str) -> None:
        """Disable permanently after the first failure, logging once."""
        if not self._disabled:
            self._disabled = True
            log(WARNING, "Live event log disabled (%s: %s: %s)", where, type(exc).__name__, exc)

    def _heartbeat(self) -> None:
        """Touch the sentinel so readers can distinguish 'slow' from 'dead'.

        The file stays zero bytes and is only ever `os.utime`d -- a rewritten
        file could be read empty mid-write, and a `stat` (unlike an `open`) holds
        no Windows file handle, so the producer's unlink() in close() can never
        hit a sharing violation caused by the dashboard.
        """
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                os.utime(self.alive_path, None)
            except BaseException:  # noqa: BLE001 - a dead heartbeat is not fatal
                return

    # -- public API --------------------------------------------------------

    def emit(self, event_type: str, **fields: Any) -> None:
        """Append one event. Never raises, never blocks on the reader."""
        if self._disabled or self._fh is None:
            return
        try:
            with self._lock:
                obj = {"seq": self._seq, "t": time.time(), "type": event_type, **fields}
                # default=_jsonable is required, not defensive: evaluate_model()
                # returns numpy scalars and a numpy-derived confusion matrix, so
                # without it the round-0 event raises TypeError and latches the
                # emitter off before the dashboard sees anything.
                line = json.dumps(obj, separators=(",", ":"), default=_jsonable)
                self._fh.write(line + "\n")
                self._fh.flush()
                self._seq += 1
        except BaseException as exc:  # noqa: BLE001 - never fail the run
            self._latch(exc, f"emit {event_type}")

    def close(self, status: str = "ok", **fields: Any) -> None:
        """Emit the terminal event, stop the heartbeat, remove the sentinel."""
        if not self._disabled:
            self.emit("run_end" if status == "ok" else "run_error", status=status, **fields)
        self._stop.set()
        try:
            if self._hb is not None:
                self._hb.join(timeout=HEARTBEAT_SECONDS + 1.0)
            if self._fh is not None:
                self._fh.close()
            self.alive_path.unlink(missing_ok=True)
        except BaseException as exc:  # noqa: BLE001
            self._latch(exc, "close")
        finally:
            self._fh = None
            self._disabled = True
