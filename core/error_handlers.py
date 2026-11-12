"""FastAPI exception handlers for the canonical API error contract."""
from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import ErrorCode, LLMUpstreamError, build_error_response, error_code_for_status
from core.network_diagnostics import get_public_egress_ip
from core.observability import bind, mark_error_logged

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id(request: Request) -> str:
    """Return the request id assigned by middleware, with a safe fallback."""
    return getattr(request.state, "request_id", None) or request.headers.get(REQUEST_ID_HEADER) or "unknown"


def _normalize_http_detail(detail: Any) -> tuple[str, Any | None]:
    """Convert legacy HTTPException detail values into message/details."""
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("error") or "Request failed"
        details = detail.get("details")
        if details is None:
            details = {k: v for k, v in detail.items() if k not in {"error", "message", "details"}}
            if not details:
                details = None
        return str(message), details
    if isinstance(detail, list):
        return "Request validation failed", detail
    if detail is None:
        return "Request failed", None
    return str(detail), None


def _safe_upstream_diagnostics(detail: Any) -> dict[str, Any]:
    """Return safe diagnostics for an upstream failure without exposing secrets."""
    message, details = _normalize_http_detail(detail)
    # Remove common credential/token patterns before returning provider errors.
    safe_message = re.sub(r"(?i)(api[_-]?key|token|authorization|bearer|password|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", message)
    # Avoid returning excessively large provider exception payloads.
    safe_message = safe_message[:2000]
    diagnostics: dict[str, Any] = {"reason": safe_message}
    if isinstance(details, dict):
        safe_details = {}
        for key, value in details.items():
            if str(key).lower() in {"api_key", "apikey", "token", "authorization", "password", "secret"}:
                safe_details[key] = "[REDACTED]"
            else:
                safe_details[key] = value
        diagnostics["upstream_details"] = safe_details


    return diagnostics


def _json_response(request: Request, status_code: int, code: ErrorCode, message: str, details: Any | None = None) -> JSONResponse:
    """Serialize the canonical error model and preserve the request id in a header."""
    request_id = get_request_id(request)
    bind(error_code=getattr(code, "value", str(code)), error_status=status_code)
    payload = build_error_response(
        code=code,
        message=message,
        details=details,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def llm_upstream_exception_handler(request: Request, exc: LLMUpstreamError) -> JSONResponse:
    """Return actionable LLM diagnostics, including the provider-visible public egress IP."""
    public_ip = exc.public_egress_ip or get_public_egress_ip()
    details: dict[str, Any] = {
        "provider": exc.provider,
        "reason": str(exc)[:2000],
        "public_egress_ip": public_ip or "unknown",
        "whitelist_hint": (
            "If the provider or Google Cloud control requires source-IP allowlisting, "
            "whitelist this public egress IP."
            if public_ip
            else "Public egress IP could not be detected. Set PUBLIC_EGRESS_IP in deployment configuration."
        ),
    }
    if exc.model:
        details["model"] = exc.model
    if exc.status_code:
        details["upstream_status_code"] = exc.status_code

    logger.error(
        "LLM upstream failure: %s %s provider=%s model=%s upstream_status=%s",
        request.method,
        request.url.path,
        exc.provider,
        exc.model or "unknown",
        exc.status_code or "-",
        exc_info=exc,
        extra={"event": "llm_upstream_failure", "provider": exc.provider, "model": exc.model or "unknown",
               "upstream_status": exc.status_code, "public_egress_ip": public_ip or "unknown"},
    )
    return _json_response(
        request,
        502,
        ErrorCode.UPSTREAM_ERROR,
        "LLM upstream request failed",
        details,
    )


def _is_retryable(exc: HTTPException) -> bool:
    """A 503 that tells the client when to ask again (the scenario's generation design is still running) is not a server fault."""
    return exc.status_code == 503 and isinstance(exc.detail, dict) and "retryAfterSeconds" in exc.detail


def _log_http_exception(request: Request, exc: HTTPException, message: str) -> None:
    """Write the reason a request was refused. The client may be told less (5xx detail is hidden); the log keeps the original text.

    5xx is ERROR with the traceback (one ERROR per real fault, which is what alerts count); 4xx is WARNING, an unknown path INFO.
    """
    status = exc.status_code
    fields = {"event": "http_error", "error_status": status}
    text = message[:500]
    if status >= 500 and not _is_retryable(exc):
        logger.error("%s %s -> %d %s", request.method, request.url.path, status, text, exc_info=exc, extra=fields)
    elif status >= 500:
        logger.info("%s %s -> %d %s (retryable)", request.method, request.url.path, status, text, extra=fields)
    elif status == 404:
        logger.info("%s %s -> 404 %s", request.method, request.url.path, text, extra=fields)
    elif status >= 400:
        logger.warning("%s %s -> %d %s", request.method, request.url.path, status, text, extra=fields)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize application and FastAPI HTTP exceptions."""
    code = error_code_for_status(exc.status_code)
    message, details = _normalize_http_detail(exc.detail)
    _log_http_exception(request, exc, message)

    # Preserve redacted provider diagnostics for upstream failures.
    if exc.status_code == 502:
        message = "An upstream service failed to complete the request"
        details = _safe_upstream_diagnostics(exc.detail)
    elif exc.status_code >= 500 and not _is_retryable(exc):
        message = "An unexpected server error occurred"
        details = None

    return _json_response(request, exc.status_code, code, message, details)


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Normalize request-body, query, path, header, and form validation errors."""
    details: list[dict[str, Any]] = []
    for error in exc.errors():
        location = list(error.get("loc", []))
        if location and location[0] in {"body", "query", "path", "header", "form"}:
            location = location[1:]
        details.append(
            {
                "field": ".".join(str(part) for part in location) or None,
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type"),
            }
        )

    summary = "; ".join(f"{d['field'] or 'body'}: {d['message']}" for d in details[:10])
    logger.warning("%s %s -> 422 validation failed: %s", request.method, request.url.path, summary[:500],
                   extra={"event": "validation_error", "error_status": 422, "invalid_fields": [d["field"] for d in details[:20]]})
    return _json_response(
        request,
        422,
        ErrorCode.VALIDATION_ERROR,
        "Request validation failed",
        details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected exceptions without exposing internal implementation details."""
    logger.error(
        "Unhandled %s on %s %s: %s",
        type(exc).__name__,
        request.method,
        request.url.path,
        str(exc)[:500],
        exc_info=exc,
        extra={"event": "unhandled_exception", "error_status": 500},
    )
    mark_error_logged()
    return _json_response(
        request,
        500,
        ErrorCode.INTERNAL_SERVER_ERROR,
        "An unexpected server error occurred",
    )
