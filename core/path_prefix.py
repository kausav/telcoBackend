"""Lets the app live under a path prefix such as ``/source`` (a shared load balancer that gives each app one path).

Set ``URL_PREFIX=/source``. A request for ``/source`` or ``/source/anything`` is handled as ``/`` or ``/anything``, so the web UI's
files, ``/runtime-config.js`` and every API route work under the prefix without being rewritten. Requests that do not start with
the prefix are untouched (health probes from Cloud Run itself call ``/health``). With no ``URL_PREFIX`` this does nothing.

It is a plain ASGI wrapper so it runs before everything else, including the token check, which therefore sees the stripped path.
"""
from __future__ import annotations

import os


def normalise(prefix: str | None) -> str:
    """'' for none; otherwise '/name' with no trailing slash ('source', '/source/' and ' /source ' all give '/source')."""
    cleaned = (prefix or "").strip().strip("/")
    return "/" + cleaned if cleaned else ""


URL_PREFIX = normalise(os.getenv("URL_PREFIX"))


class StripPrefixMiddleware:
    def __init__(self, app, prefix: str | None = None) -> None:
        self.app = app
        self.prefix = URL_PREFIX if prefix is None else normalise(prefix)

    async def __call__(self, scope, receive, send):
        prefix = self.prefix
        if prefix and scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == prefix or path.startswith(prefix + "/"):
                scope = dict(scope)
                scope["path"] = path[len(prefix):] or "/"
                raw = scope.get("raw_path")
                if raw:
                    raw_prefix = prefix.encode()
                    if raw == raw_prefix or raw.startswith(raw_prefix + b"/") or raw.startswith(raw_prefix + b"?"):
                        scope["raw_path"] = raw[len(raw_prefix):] or b"/"
        await self.app(scope, receive, send)
