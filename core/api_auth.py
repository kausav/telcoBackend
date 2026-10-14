"""Static API-token authentication (a stop-gap until a real login flow exists).

Callers send the token as ``Authorization: Bearer <token>`` or ``X-API-Key: <token>``. Tokens come from the environment
(``API_STATIC_TOKENS``, comma-separated for rotation); with none configured every protected request is refused, and ``off`` switches authentication off. Probes (``/health``, ``/ready``)
and CORS preflight requests never need a token. A token is never logged and is compared in constant time.
"""
from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from config.runtime import API_AUTH_DISABLED, API_AUTH_PROTECT_DOCS, API_AUTH_PUBLIC_PATHS, API_STATIC_TOKENS
from core.error_handlers import _json_response
from core.errors import ErrorCode

logger = logging.getLogger(__name__)

DOC_PATHS = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


_VALID = [_digest(t) for t in API_STATIC_TOKENS]


def enabled() -> bool:
    return not API_AUTH_DISABLED


def presented_token(request: Request) -> str:
    """The token the caller sent, from the Authorization header (Bearer) or X-API-Key; "" when there is none."""
    header = str(request.headers.get("Authorization") or "").strip()
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    return str(request.headers.get("X-API-Key") or "").strip()


def token_ok(token: str) -> bool:
    if not token:
        return False
    candidate = _digest(token)
    return any(hmac.compare_digest(candidate, valid) for valid in _VALID)       # no early exit on the first match: same work for every token


def needs_token(request: Request) -> bool:
    if not enabled() or request.method == "OPTIONS":
        return False
    path = request.url.path.rstrip("/") or "/"
    if path in API_AUTH_PUBLIC_PATHS:
        return False
    if path in DOC_PATHS and not API_AUTH_PROTECT_DOCS:
        return False
    return True


async def api_token_middleware(request: Request, call_next):
    if needs_token(request) and not token_ok(presented_token(request)):
        response: JSONResponse = _json_response(request, 401, ErrorCode.UNAUTHORIZED,
                                                "A valid API token is required. Send it as 'Authorization: Bearer <token>' or 'X-API-Key: <token>'.")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    return await call_next(request)


def log_startup() -> None:
    if API_AUTH_DISABLED:
        logger.warning("API authentication is OFF (API_STATIC_TOKENS=off): do not run like this outside local development")
    elif _VALID:
        logger.info("API authentication: static token required (%d token(s) configured)", len(_VALID))
    else:
        logger.error("API_STATIC_TOKENS is not set: authentication is ON and no token is accepted, so every protected request is refused")


def openapi_security(schema: dict) -> dict:
    """Add the two accepted token headers to the OpenAPI document so the docs page offers an Authorize button."""
    if not enabled():
        return schema
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {}).update({
        "BearerToken": {"type": "http", "scheme": "bearer", "description": "Static API token"},
        "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key", "description": "Static API token"},
    })
    schema["security"] = [{"BearerToken": []}, {"ApiKey": []}]
    return schema
