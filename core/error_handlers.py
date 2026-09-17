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
        "LLM upstream failure [request_id=%s provider=%s model=%s public_egress_ip=%s]",
        get_request_id(request),
        exc.provider,
        exc.model or "unknown",
        public_ip or "unknown",
    )
    return _json_response(
        request,
        502,
        ErrorCode.UPSTREAM_ERROR,
        "LLM upstream request failed",
        details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize application and FastAPI HTTP exceptions."""
    code = error_code_for_status(exc.status_code)
    message, details = _normalize_http_detail(exc.detail)

    # Preserve redacted provider diagnostics for upstream failures.
    if exc.status_code == 502:
        message = "An upstream service failed to complete the request"
        details = _safe_upstream_diagnostics(exc.detail)
    elif exc.status_code >= 500:
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

    return _json_response(
        request,
        422,
        ErrorCode.VALIDATION_ERROR,
        "Request validation failed",
        details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected exceptions without exposing internal implementation details."""
    request_id = get_request_id(request)
    logger.exception(
        "Unhandled API error [request_id=%s] %s %s",
        request_id,
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return _json_response(
        request,
        500,
        ErrorCode.INTERNAL_SERVER_ERROR,
        "An unexpected server error occurred",
    )
