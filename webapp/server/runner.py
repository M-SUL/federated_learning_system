"""Start federations from the dashboard, under strict constraints.

Three properties this module exists to guarantee:

1. **No free-text config reaches a subprocess.** Every field is validated against
   an explicit whitelist and the argv is built as a list with no shell. A text
   box wired to `--run-config` would be command-injection surface, and "it is
   only localhost" stops being true the moment someone binds 0.0.0.0 for a
   projector.

2. **Never two runs at once.** Five concurrent simulations exhausted memory
   during development: Ray workers died importing sklearn, and the affected run
   sat frozen for nine rounds reporting a plausible-looking score. A button makes
   that mistake one click away, so it is refused here.

3. **Launching is optional.** The dashboard remains fully usable, and runs
   started from a terminal behave exactly as before. Nothing here is on the path
   of an observed run.

Note `flwr run` is asynchronous: it submits the run to a local SuperLink and
exits within seconds while training continues in separate processes. So the
child's exit tells us whether the *submission* worked, and liveness continues to
come from the event log, not from the child.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import live as live_mod

# Whitelists. Values outside these never reach the subprocess.
STRATEGIES = ("iid", "dirichlet", "dirichlet-legacy")
ALPHAS = (0.1, 0.3, 0.5, 1.0)
NORMS = ("batch", "layer", "none")
ROUNDS = (2, 3, 5, 10, 15, 20)
SEEDS = (42, 43, 44)
# Capped at 10: every client is a Ray worker holding its own shard, and the 640
# training samples get thin fast (64 each at 10 clients, which the dirichlet
# strategy's 32-sample floor can struggle to satisfy at low alpha).
CLIENTS = (2, 3, 5, 8, 10)

# The client count lives in TWO files that must agree, or server_app raises:
#   ~/.flwr/config.toml   options.num-supernodes  -- how many clients Flower runs
#   pyproject.toml        num-partitions          -- how many shards the data splits into
# --run-config can set the second but not the first: num-supernodes is SuperLink
# connection config, not run config. So changing the client count means editing
# the Flower config file and restarting the SuperLink to pick it up.
FLWR_CONFIG = Path.home() / ".flwr" / "config.toml"
_SUPERNODES_RE = re.compile(r"^(\s*options\.num-supernodes\s*=\s*)(\d+)\s*$", re.MULTILINE)

# `flwr run` should return in a few seconds; anything longer means it is stuck
# (a port conflict, or a SuperLink that will not start) and we want the error.
SUBMIT_TIMEOUT_S = 180

# How long to wait after submission for the run's event log to appear, so the
# response carries a real run id and a confirmed start.
STARTUP_WAIT_S = 40
# How long a just-submitted run counts as busy before the .alive sentinel takes
# over. Must comfortably exceed the ServerApp's start-up time.
STARTUP_GRACE_S = 90

# A running simulation is a process tree: flower-superlink spawns
# flower-superexec, which spawns the ServerApp and Ray workers as plain
# `python.exe` children. There are no flwr-serverapp.exe processes to target, and
# killing by the python.exe image name would take down this dashboard and any
# other Python the user is running -- so the process TREE is the only safe handle.
#
# Both roots are torn down, in this order. Killing only the superexec was tried
# and does not work: the SuperLink does not reliably respawn one, so the run
# stops but every later launch is accepted and then silently never starts. A
# fresh SuperLink per launch is slower by a second or two and always works.
STOP_TREE_ROOTS = ("flower-superexec.exe", "flower-superlink.exe")

_RUN_ID_RE = re.compile(r"run\s+(\d{6,})")


def options() -> dict:
    """The exact choices the UI may offer. The UI never invents its own."""
    return {
        "strategy": list(STRATEGIES),
        "alpha": list(ALPHAS),
        "norm": list(NORMS),
        "rounds": list(ROUNDS),
        "seed": list(SEEDS),
        "clients": list(CLIENTS),
    }


def current_supernodes() -> int | None:
    """How many clients the Flower config is currently set to spawn."""
    try:
        match = _SUPERNODES_RE.search(FLWR_CONFIG.read_text(encoding="utf-8"))
        return int(match.group(2)) if match else None
    except OSError:
        return None


def set_supernodes(n: int) -> tuple[bool, str]:
    """Point the Flower config at `n` clients. Returns (changed, message).

    Touches exactly one line and only when the value differs, so an unrelated
    edit in the user's config is never clobbered. The SuperLink reads this at
    startup, so the caller must restart it for a change to take effect.
    """
    if n not in CLIENTS:
        return False, f"{n} is not an offered client count"
    try:
        text = FLWR_CONFIG.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"cannot read {FLWR_CONFIG}: {exc}"

    match = _SUPERNODES_RE.search(text)
    if not match:
        return False, f"no options.num-supernodes line in {FLWR_CONFIG}"
    if int(match.group(2)) == n:
        return False, "already set"

    try:
        FLWR_CONFIG.write_text(
            _SUPERNODES_RE.sub(lambda m: f"{m.group(1)}{n}", text, count=1),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"cannot write {FLWR_CONFIG}: {exc}"
    return True, f"num-supernodes {match.group(2)} -> {n}"


class ValidationError(ValueError):
    pass


def _validate(payload: dict) -> dict:
    """Coerce and whitelist. Raises ValidationError on anything unexpected."""
    def pick(name, allowed, cast):
        if name not in payload:
            raise ValidationError(f"missing field: {name}")
        try:
            value = cast(payload[name])
        except (TypeError, ValueError):
            raise ValidationError(f"{name}: not a valid value") from None
        if value not in allowed:
            raise ValidationError(f"{name}: {value!r} is not one of {list(allowed)}")
        return value

    strategy = pick("strategy", STRATEGIES, str)
    spec = {
        "strategy": strategy,
        "rounds": pick("rounds", ROUNDS, int),
        "norm": pick("norm", NORMS, str),
        "seed": pick("seed", SEEDS, int),
        "clients": pick("clients", CLIENTS, int),
        # alpha is meaningless for iid; accept it but do not pass it on.
        "alpha": None if strategy == "iid" else pick("alpha", ALPHAS, float),
    }
    unexpected = set(payload) - {"strategy", "rounds", "norm", "seed", "alpha", "clients"}
    if unexpected:
        raise ValidationError(f"unexpected field(s): {sorted(unexpected)}")
    return spec


def _run_config(spec: dict) -> str:
    """Build the --run-config string ourselves, only from validated values."""
    parts = [
        f"partition-strategy='{spec['strategy']}'",
        f"num-server-rounds={spec['rounds']}",
        f"norm='{spec['norm']}'",
        f"seed={spec['seed']}",
        # Must equal options.num-supernodes, which start() keeps in step.
        f"num-partitions={spec['clients']}",
    ]
    if spec["alpha"] is not None:
        parts.append(f"dirichlet-alpha={spec['alpha']}")
    return " ".join(parts)


class Launcher:
    """Submits runs and stops them. One at a time, by construction."""

    def __init__(self, app_dir: Path, live_dir: Path, flwr_exe: Path | None = None):
        self.app_dir = Path(app_dir)
        self.live_dir = Path(live_dir)
        self.flwr_exe = Path(flwr_exe) if flwr_exe else self._find_flwr()
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.last_spec: dict | None = None
        # `flwr run` returns once the run is SUBMITTED; the ServerApp that writes
        # the .alive sentinel starts moments later in another process. Without a
        # timestamp guard there is a window where a just-launched run is
        # invisible to busy(), and a second click starts a concurrent run --
        # which is the exact failure this class exists to prevent.
        self._launched_at: float = 0.0
        self._pending_id: str | None = None

    def _find_flwr(self) -> Path | None:
        for candidate in (
            self.app_dir.parent / ".venv-fl" / "Scripts" / "flwr.exe",
            self.app_dir.parent / ".venv-fl" / "bin" / "flwr",
        ):
            if candidate.exists():
                return candidate
        # In a container everything shares one environment, so flwr is simply on
        # PATH rather than in a sibling venv.
        found = shutil.which("flwr")
        return Path(found) if found else None

    def available(self) -> tuple[bool, str]:
        if self.flwr_exe is None or not Path(self.flwr_exe).exists():
            return False, "flwr executable not found (expected in .venv-fl)"
        if not (self.app_dir / "pyproject.toml").exists():
            return False, f"no Flower app at {self.app_dir}"
        return True, ""

    def busy(self) -> dict | None:
        """The currently live run, if any.

        Covers the startup window as well as the steady state: a run submitted
        seconds ago has no .alive sentinel yet, but must still count as busy.
        """
        current, _ = live_mod.current_run(self.live_dir)
        if current and current["is_live"]:
            return current

        if self._launched_at and (time.time() - self._launched_at) < STARTUP_GRACE_S:
            # If the pending run has already written a terminal event it is
            # genuinely over, however recently it was launched.
            if current and current["run_id"] == self._pending_id \
                    and current["last_type"] in ("run_end", "run_error"):
                return None
            return current if current and current["run_id"] == self._pending_id else {
                "run_id": self._pending_id or "(starting)",
                "status": "starting", "is_live": True, "slug": None,
                "started_at": self._launched_at, "last_type": None, "last_seq": None,
                "config": {}, "num_rounds": None, "num_clients": None, "path": None,
            }
        return None

    def _await_start(self, known: set[str], deadline: float) -> str | None:
        """Wait for the submitted run's log to appear, so we report a real id."""
        while time.time() < deadline:
            for run in live_mod.list_runs(self.live_dir):
                if run["run_id"] not in known:
                    return run["run_id"]
            time.sleep(0.4)
        return None

    def start(self, payload: dict) -> dict:
        ok, why = self.available()
        if not ok:
            return {"ok": False, "error": why}

        spec = _validate(payload)  # raises before anything is spawned

        # Serialized so two clicks cannot race past the busy check.
        with self._lock:
            busy = self.busy()
            if busy:
                return {"ok": False, "error":
                        f"run {busy['run_id']} is already in progress. Concurrent "
                        "simulations exhaust memory and corrupt each other's results.",
                        "busy_run": busy}

            # Keep the two client-count settings in step. The SuperLink reads
            # num-supernodes once at startup, so a change only takes effect after
            # it restarts -- and the next launch spawns a fresh one.
            changed, detail = set_supernodes(spec["clients"])
            if changed:
                self._kill_trees()
                time.sleep(2.0)

            known = {r["run_id"] for r in live_mod.list_runs(self.live_dir)}
            argv = [
                str(self.flwr_exe), "run", ".", "local-simulation",
                "--run-config", _run_config(spec),
            ]
            env = {
                **os.environ,
                # The flwr CLI prints an emoji and dies on a cp1252 console.
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                # `flwr run` spawns flower-superlink itself and resolves it via
                # PATH, so calling flwr.exe by absolute path is not enough: without
                # its directory on PATH the run is accepted and then never starts,
                # failing with "[WinError 2] The system cannot find the file
                # specified".
                "PATH": str(Path(self.flwr_exe).parent) + os.pathsep + os.environ.get("PATH", ""),
            }
            try:
                proc = subprocess.run(
                    argv, cwd=str(self.app_dir), env=env,
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=SUBMIT_TIMEOUT_S,
                    # No shell: argv stays a list so nothing is re-parsed.
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                self.last_error = "flwr run did not return within 3 minutes"
                return {"ok": False, "error": self.last_error}
            except OSError as exc:
                self.last_error = f"could not launch flwr: {exc}"
                return {"ok": False, "error": self.last_error}

            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                self.last_error = out.strip()[-600:] or f"exit code {proc.returncode}"
                return {"ok": False, "error": self.last_error, "argv": argv[1:]}

            # Mark busy immediately: the run exists from this moment even though
            # its event log has not appeared yet.
            self._launched_at = time.time()

            match = _RUN_ID_RE.search(out)
            run_id = match.group(1) if match else None
            # Confirm the ServerApp actually came up, and learn the id it used.
            appeared = self._await_start(known, self._launched_at + STARTUP_WAIT_S)
            # The live run is authoritative: stdout parsing and a stale log left
            # by an earlier aborted run can both mislead.
            current, _ = live_mod.current_run(self.live_dir)
            if current and current["is_live"]:
                run_id = current["run_id"]
            else:
                run_id = appeared or run_id
            self._pending_id = run_id

            self.last_spec, self.last_error = spec, None
            return {"ok": True, "run_id": run_id, "spec": spec,
                    "confirmed": appeared is not None,
                    "run_config": _run_config(spec), "output": out.strip()[-600:]}

    def _kill_trees(self) -> list[str]:
        """Tear down the SuperExec and SuperLink process trees.

        Two implementations because the process model differs. On Windows
        taskkill /T walks the tree for us. On Linux the same names exist as
        console scripts, so pkill by name plus its children is the equivalent;
        `pkill -f` is used because the executable is a python shim whose argv[0]
        carries the real name.
        """
        killed = []
        for name in STOP_TREE_ROOTS:
            base = name[:-4] if name.endswith(".exe") else name
            if sys.platform == "win32":
                cmd = ["taskkill", "/F", "/T", "/IM", name]
            else:
                # -f matches the full command line, which is what a console
                # script looks like once python has exec'd it.
                cmd = ["pkill", "-9", "-f", base]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=30, shell=False)
                if r.returncode == 0:
                    killed.append(base)
            except (OSError, subprocess.SubprocessError):
                continue

        if sys.platform != "win32":
            # pkill does not reap the Ray workers, which are children rather than
            # name matches. They exit once their parent is gone, but a stray
            # `ray` process would keep memory pinned on a small VM.
            try:
                subprocess.run(["pkill", "-9", "-f", "ray::"], capture_output=True,
                               timeout=15, shell=False)
            except (OSError, subprocess.SubprocessError):
                pass
        return killed

    def stop(self) -> dict:
        """Terminate the processes a simulation runs in.

        `flwr run` has already exited by now, so there is no child to signal --
        training lives in the SuperExec-spawned processes. Only one run is ever
        permitted, so stopping by image name cannot catch someone else's work.
        """
        killed = self._kill_trees()

        # Clear the busy grace so the UI does not keep claiming a run is starting.
        self._launched_at, self._pending_id = 0.0, None

        return {
            "ok": True,
            "stopped": bool(killed),
            "terminated": killed,
            "note": "Completed rounds are kept in the event log; the run shows as "
                    "ended unexpectedly. The next launch starts a fresh SuperLink."
                    if killed else "Nothing was running.",
        }
