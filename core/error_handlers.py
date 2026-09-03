"""FastAPI exception handlers for the canonical API error contract."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import ErrorCode, build_error_response, error_code_for_status

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


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize application and FastAPI HTTP exceptions."""
    code = error_code_for_status(exc.status_code)
    message, details = _normalize_http_detail(exc.detail)

    # Never expose implementation details from an unexpected 5xx HTTPException.
    if exc.status_code >= 500:
        if exc.status_code == 502:
            message = "An upstream service failed to complete the request"
        else:
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
