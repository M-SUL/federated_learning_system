"""Discover federation runs and tail their event logs.

The contract with the training side is one append-only JSONL file per run, plus a
zero-byte `<run_id>.alive` sentinel touched every few seconds while the run is
going. Liveness comes from the sentinel, never from the JSONL's mtime: a round
takes 25-35 s here, so any mtime threshold tight enough to notice a crash would
also fire during a normal slow round.

The sentinel is only ever `stat`ed, never opened, so the producer's `unlink()`
can never hit a Windows sharing violation caused by this reader.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

# Must exceed the producer's HEARTBEAT_SECONDS (5.0) with room for a stalled
# thread. Source of truth: fl_rnaseq/fl_rnaseq/events.py STALE_AFTER_SECONDS.
STALE_AFTER_SECONDS = 20.0

POLL_SECONDS = 0.25
KEEPALIVE_SECONDS = 15.0
IDLE_GRACE_SECONDS = 10.0

TERMINAL_TYPES = ("run_end", "run_error")


def _alive_fresh(alive_path: Path) -> bool:
    try:
        return (time.time() - os.path.getmtime(alive_path)) < STALE_AFTER_SECONDS
    except OSError:
        return False


def _read_first_and_last(path: Path) -> tuple[dict | None, dict | None]:
    """First line (run_start) and last parseable line, cheaply."""
    first = last = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if i == 0 or first is None:
                    first = obj
                last = obj
    except OSError:
        pass
    return first, last


def describe_run(jsonl: Path) -> dict:
    """Metadata + derived status for one run log."""
    first, last = _read_first_and_last(jsonl)
    alive = jsonl.with_suffix(".alive")
    fresh = _alive_fresh(alive)
    terminal = (last or {}).get("type") in TERMINAL_TYPES

    if fresh and not terminal:
        status = "running"
    elif terminal:
        status = "finished" if (last or {}).get("type") == "run_end" else "failed"
    else:
        # No sentinel and no terminal event: the process died without unwinding.
        status = "crashed"

    return {
        "run_id": jsonl.stem,
        "path": str(jsonl),
        "status": status,
        "is_live": status == "running",
        "started_at": (first or {}).get("t"),
        "slug": (first or {}).get("slug"),
        "config": (first or {}).get("config") or {},
        "num_rounds": (first or {}).get("num_rounds"),
        "num_clients": (first or {}).get("num_clients"),
        "last_seq": (last or {}).get("seq"),
        "last_type": (last or {}).get("type"),
    }


def list_runs(live_dir: Path) -> list[dict]:
    """Newest first."""
    if not live_dir.is_dir():
        return []
    runs = [describe_run(p) for p in live_dir.glob("*.jsonl")]
    runs.sort(key=lambda r: (r["started_at"] or 0), reverse=True)
    return runs


def current_run(live_dir: Path) -> tuple[dict | None, list[str]]:
    """The run to stream, plus the ids of any other simultaneously live runs.

    Never merges two runs into one stream -- concurrent runs are a real scenario
    (five were launched at once during development) and interleaving them would
    silently produce nonsense.
    """
    runs = list_runs(live_dir)
    live = [r for r in runs if r["is_live"]]
    if live:
        return live[0], [r["run_id"] for r in live[1:]]
    return (runs[0] if runs else None), []


def tail_events(jsonl: Path, since_seq: int = -1) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, payload)`` for each event, then terminate.

    ``kind`` is ``"event"`` for a log line (payload = the raw JSON text,
    re-emitted verbatim so curl and the browser see identical bytes),
    ``"keepalive"``, or ``"eof"``.

    Terminates on three independent conditions so a stuck stream cannot leak a
    thread: a terminal event in the log, a stale sentinel plus an idle period,
    or a write failure when the consumer has gone away.
    """
    alive = jsonl.with_suffix(".alive")
    pos = 0
    last_emit = last_data = time.monotonic()
    terminal_seen = False
    last_round = None

    try:
        fh = open(jsonl, "r", encoding="utf-8")
    except OSError:
        yield "eof", json.dumps({"reason": "unreadable"})
        return

    try:
        while True:
            fh.seek(pos)
            line = fh.readline()

            if line.endswith("\n"):
                # Commit the position ONLY for a complete line. A partial write
                # is re-read next tick instead of being emitted as broken JSON.
                pos = fh.tell()
                text = line.strip()
                if text:
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        # Skip it, but keep the advanced position: one bad line
                        # must not wedge the stream forever.
                        continue
                    seq = obj.get("seq", -1)
                    if seq > since_seq:
                        if obj.get("round") is not None:
                            last_round = obj["round"]
                        if obj.get("type") in TERMINAL_TYPES:
                            terminal_seen = True
                        yield "event", text
                        last_emit = last_data = time.monotonic()
                continue

            now = time.monotonic()
            if terminal_seen and now - last_data > 1.0:
                yield "eof", json.dumps({"reason": "run_finished"})
                return
            if not _alive_fresh(alive) and now - last_data > IDLE_GRACE_SECONDS:
                yield "eof", json.dumps({"reason": "stale", "last_round": last_round})
                return
            if now - last_emit > KEEPALIVE_SECONDS:
                # The only way to notice a closed browser tab: a read-only
                # server never learns the peer went away, but a write does.
                yield "keepalive", ""
                last_emit = now
            time.sleep(POLL_SECONDS)
    finally:
        fh.close()
