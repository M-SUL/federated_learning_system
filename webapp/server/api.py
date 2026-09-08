"""FastAPI app: static snapshot plus a live SSE stream of a running federation.

Observe-only by design — there is no route that starts, stops, or configures
training. The dashboard reads an append-only log the ServerApp writes; if this
process is not running, training is entirely unaffected.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import live as live_mod
from . import snapshot as snapshot_mod

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(results_dir: Path, live_dir: Path) -> FastAPI:
    app = FastAPI(title="FL RNA-seq Federation Console", docs_url=None, redoc_url=None)
    results_dir, live_dir = Path(results_dir), Path(live_dir)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "results_dir": str(results_dir),
            "live_dir": str(live_dir),
            "results_exists": results_dir.is_dir(),
            "live_exists": live_dir.is_dir(),
            # Exposed so the thread-leak check in the verification steps is a
            # one-line curl rather than a debugger session.
            "active_threads": threading.active_count(),
        }

    @app.get("/api/snapshot")
    def get_snapshot(request: Request) -> Response:
        payload = snapshot_mod.build(results_dir)
        etag = f'W/"{hash(json.dumps(payload, sort_keys=True, default=str)) & 0xFFFFFFFF:x}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304)
        return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": "no-cache"})

    @app.get("/api/runs")
    def get_runs() -> dict:
        runs = live_mod.list_runs(live_dir)
        current, others = live_mod.current_run(live_dir)
        return {
            "runs": runs,
            "current": current["run_id"] if current else None,
            "multiple_live": others,
        }

    @app.get("/api/events")
    def events(request: Request, run: str | None = None) -> Response:
        if run:
            target = live_dir / f"{run}.jsonl"
            info = live_mod.describe_run(target) if target.exists() else None
            others: list[str] = []
        else:
            info, others = live_mod.current_run(live_dir)

        if info is None:
            # 204 is the spec-defined signal that stops EventSource retrying.
            # Returning 200 with an empty body would hold a thread per idle tab.
            return Response(status_code=204)

        # A reconnecting EventSource replays its position automatically.
        try:
            since = int(request.headers.get("last-event-id", "-1"))
        except ValueError:
            since = -1

        def stream():
            hello = {"run": info, "multiple_live": others}
            yield f"event: hello\ndata: {json.dumps(hello)}\n\n"
            for kind, payload in live_mod.tail_events(Path(info["path"]), since_seq=since):
                if kind == "keepalive":
                    yield ": keepalive\n\n"
                elif kind == "eof":
                    yield f"event: eof\ndata: {payload}\n\n"
                else:
                    obj = json.loads(payload)
                    # data is the log line verbatim, so curl -N and the browser
                    # see byte-identical payloads.
                    yield f"id: {obj.get('seq','')}\nevent: {obj.get('type','event')}\ndata: {payload}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
