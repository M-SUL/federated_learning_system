"""HTTP basic auth for public deployments.

Off entirely when no password is configured, so nothing changes for the normal
localhost workflow. The important rule is enforced in serve.py rather than here:
a publicly-bound instance that can LAUNCH runs refuses to start without a
password, because otherwise the only thing between the internet and a machine
that spawns CPU-heavy jobs is nobody knowing the URL.
"""

from __future__ import annotations

import base64
import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse, Response

REALM = "FL RNA-seq Federation Console"
# Left open so a container orchestrator can probe the app without credentials.
PUBLIC_PATHS = ("/api/health",)


def credentials_from_env(env: dict | None = None) -> tuple[str, str] | None:
    env = os.environ if env is None else env
    password = env.get("DASHBOARD_PASSWORD", "")
    if not password:
        return None
    return env.get("DASHBOARD_USER", "admin"), password


def _unauthorized() -> Response:
    return JSONResponse(
        {"detail": "authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": f'Basic realm="{REALM}", charset="UTF-8"'},
    )


def check(header: str | None, user: str, password: str) -> bool:
    """Constant-time credential comparison against an Authorization header."""
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        got_user, _, got_pass = decoded.partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    # compare_digest on both halves: a plain == would leak the password length
    # and prefix through timing.
    return (hmac.compare_digest(got_user, user)
            and hmac.compare_digest(got_pass, password))


def install(app, user: str, password: str) -> None:
    """Require basic auth on every route except the health probe."""

    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        if not check(request.headers.get("authorization"), user, password):
            return _unauthorized()
        return await call_next(request)
