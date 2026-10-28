"""Serves the built web UI from the same process as the API (the single-container deployment).

Nothing happens unless a built UI is present: the folder named by ``UI_DIR`` (default ``/app/ui``) must hold ``index.html``. Without it
the server behaves exactly as before, so local development and the API-only image are unchanged.

What is served (all of it public: the page has to load before it can send a token):
  * ``/``                  the UI when the caller asks for HTML (a browser); the JSON liveness answer otherwise
  * ``/assets/...``        the build's fingerprinted files, cached for a year
  * ``/<file>``            each file at the top level of the folder (favicon, logo, ...), found once at start-up
  * ``/runtime-config.js`` settings for the page, read from the environment at request time, so they can differ per environment
                           without rebuilding the image: every ``VITE_*`` variable of the running service, e.g.
                           ``window.__APP_CONFIG__ = {"VITE_API_TOKEN": "...", "VITE_DEFAULT_COUNTRY": "IN"}``

Only ``VITE_*`` variables are ever sent to the page (the same convention Vite uses for "public to the browser"), so MONGODB_URI,
GEMINI_API_KEY and the like can never leak through here. ``UI_API_TOKEN`` is accepted as another name for ``VITE_API_TOKEN``.
Whatever is sent is visible to every visitor who can load the page: the token is a convenience for a UI that sits behind the
platform's own access control (IAP, a load balancer, a VPN), not a secret from the people who open the page.

The folder is ``UI_DIR``, or ``STATIC_DIR`` if that is what the image sets.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

UI_DIR = Path(os.getenv("UI_DIR") or os.getenv("STATIC_DIR") or "/app/ui")
CONFIG_PATH = "/runtime-config.js"
ASSETS_PREFIX = "/assets"
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
}

_top_level: set[str] = set()      # "/favicon.svg" and so on, filled by mount_ui


def available() -> bool:
    return (UI_DIR / "index.html").is_file()


def is_ui_path(path: str) -> bool:
    """True for the addresses that serve the page itself, which need no token. Always False when no UI is installed."""
    if not _top_level and not available():
        return False
    return path == CONFIG_PATH or path in _top_level or path.startswith(ASSETS_PREFIX + "/")


def wants_html(request: Request) -> bool:
    """A browser navigation: it lists text/html. API clients send application/json or */* and keep getting JSON."""
    return "text/html" in (request.headers.get("accept") or "").lower()


def index_response() -> Response:
    return FileResponse(UI_DIR / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-cache", **_SECURITY_HEADERS})


def serve_index_if_browser(request: Request) -> Response | None:
    """For ``GET /``: the UI for a browser, None when the caller should get the usual JSON."""
    if available() and wants_html(request):
        return index_response()
    return None


def runtime_settings(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The VITE_* variables the page may read. Empty values are left out; UI_API_TOKEN fills VITE_API_TOKEN when that is unset."""
    source = os.environ if env is None else env
    settings = {k: v.strip() for k, v in source.items() if k.startswith("VITE_") and v.strip()}
    legacy = (source.get("UI_API_TOKEN") or "").strip()
    if legacy and "VITE_API_TOKEN" not in settings:
        settings["VITE_API_TOKEN"] = legacy
    return dict(sorted(settings.items()))


def runtime_config_body(env: Mapping[str, str] | None = None) -> str:
    """JavaScript that gives the page its settings. json.dumps quotes the values; '<' is escaped so a value can never close a script tag."""
    payload = json.dumps(runtime_settings(env)).replace("<", "\\u003c")
    return f"window.__APP_CONFIG__ = {payload};\n"


def mount_ui(app: FastAPI) -> bool:
    """Add the UI routes to ``app``. Returns False (and adds nothing) when there is no built UI. Call after the API routes."""
    if not available():
        logger.info("No web UI at %s: serving the API only", UI_DIR)
        return False

    @app.get(CONFIG_PATH, include_in_schema=False)
    def runtime_config() -> Response:
        return Response(runtime_config_body(), media_type="text/javascript",
                        headers={"Cache-Control": "no-store", **_SECURITY_HEADERS})

    assets = UI_DIR / "assets"
    if assets.is_dir():
        app.mount(ASSETS_PREFIX, StaticFiles(directory=str(assets)), name="ui-assets")

    for entry in sorted(UI_DIR.iterdir()):
        name = entry.name
        if not entry.is_file() or name in {"index.html", "runtime-config.js"} or name.startswith("."):
            continue
        _top_level.add("/" + name)
        app.add_api_route("/" + name, _file_endpoint(entry), methods=["GET"], include_in_schema=False)

    if "VITE_API_TOKEN" not in runtime_settings():
        logger.warning("No VITE_API_TOKEN (or UI_API_TOKEN) is set: the page will not send a token, so it only works while API authentication is off")
    logger.info("Serving the web UI from %s (%d top-level file(s))", UI_DIR, len(_top_level))
    return True


def _file_endpoint(path: Path):
    def endpoint() -> Response:
        return FileResponse(path, headers={"Cache-Control": "public, max-age=3600", **_SECURITY_HEADERS})
    return endpoint
