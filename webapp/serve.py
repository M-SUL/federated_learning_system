"""Run the federation console.

    python webapp/serve.py                 # http://127.0.0.1:8000
    python webapp/serve.py --port 9000

Observe-only: it never starts training. Run `flwr run .` yourself in another
terminal and the live view picks it up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "fl_rnaseq" / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--live-dir", type=Path, default=None,
                    help="default: <results-dir>/live")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--app-dir", type=Path, default=REPO_ROOT / "fl_rnaseq",
                    help="the Flower app directory (holds pyproject.toml)")
    ap.add_argument("--no-launch", action="store_true",
                    help="observe only; disable the run launcher entirely")
    ap.add_argument("--allow-remote-launch", action="store_true",
                    help="permit launching when bound to a non-loopback address "
                         "(off by default: it lets the network start jobs here)")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    live_dir = (args.live_dir or (results_dir / "live")).resolve()

    if not results_dir.is_dir():
        print(f"WARNING: results dir not found: {results_dir}", file=sys.stderr)

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    allow_launch = not args.no_launch and (loopback or args.allow_remote_launch)

    import uvicorn
    from server.api import create_app

    print(f"  results : {results_dir}")
    print(f"  live    : {live_dir}")
    print(f"  app     : {args.app_dir}")
    print(f"  launch  : {'enabled' if allow_launch else 'disabled'}"
          + ("" if loopback or not allow_launch
             else "  (exposed on a non-loopback address!)"))
    if not loopback and not allow_launch and not args.no_launch:
        print("            bound to a non-loopback address, so launching is off; "
              "pass --allow-remote-launch to override")
    print(f"  console : http://{args.host}:{args.port}")

    uvicorn.run(
        create_app(results_dir, live_dir, app_dir=args.app_dir.resolve(),
                   allow_launch=allow_launch),
        host=args.host, port=args.port, log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
