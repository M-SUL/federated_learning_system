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
from . import runner as runner_mod
from . import snapshot as snapshot_mod

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(
    results_dir: Path,
    live_dir: Path,
    app_dir: Path | None = None,
    allow_launch: bool = True,
) -> FastAPI:
    app = FastAPI(title="FL RNA-seq Federation Console", docs_url=None, redoc_url=None)
    results_dir, live_dir = Path(results_dir), Path(live_dir)
    app_dir = Path(app_dir) if app_dir else results_dir.parent
    launcher = runner_mod.Launcher(app_dir, live_dir)

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

    @app.get("/api/launch")
    def launch_info() -> dict:
        """What the launcher will accept, and whether it is usable right now."""
        ok, why = launcher.available()
        busy = launcher.busy()
        return {
            "enabled": allow_launch,
            # Disabled on a non-loopback bind unless explicitly overridden, so
            # binding 0.0.0.0 for a projector does not hand the network a way to
            # start jobs on this machine.
            "disabled_reason": None if allow_launch else
                "launching is disabled when not bound to localhost "
                "(re-run serve.py with --allow-remote-launch to override)",
            "available": ok,
            "unavailable_reason": why or None,
            "flwr_exe": str(launcher.flwr_exe) if launcher.flwr_exe else None,
            "app_dir": str(app_dir),
            "options": runner_mod.options(),
            "busy": busy,
            "last_error": launcher.last_error,
            # Changing this rewrites ~/.flwr/config.toml and restarts the
            # SuperLink, so the UI can warn that it is not a per-run knob.
            "current_clients": runner_mod.current_supernodes(),
            "flwr_config": str(runner_mod.FLWR_CONFIG),
        }

    @app.post("/api/run")
    async def start_run(request: Request) -> Response:
        if not allow_launch:
            return JSONResponse({"ok": False, "error": "launching is disabled"}, 403)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - malformed body
            return JSONResponse({"ok": False, "error": "invalid JSON body"}, 400)
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "error": "body must be an object"}, 400)
        try:
            result = launcher.start(payload)
        except runner_mod.ValidationError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, 400)
        return JSONResponse(result, 200 if result.get("ok") else 409)

    @app.post("/api/run/stop")
    def stop_run() -> Response:
        if not allow_launch:
            return JSONResponse({"ok": False, "error": "launching is disabled"}, 403)
        return JSONResponse(launcher.stop())

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html",
                            headers={"Cache-Control": "no-cache"})

    @app.middleware("http")
    async def revalidate_static(request: Request, call_next):
        """Force the browser to revalidate the app's own assets.

        Without this, a browser holds on to cached ES modules and keeps running
        an older build after the server is restarted -- which looks exactly like
        a broken feature: the API has the new routes, the served files are
        current, and the page still shows the previous UI. `no-cache` means
        "revalidate before use", not "do not store", so the ETag still saves the
        transfer when nothing changed.
        """
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
