"""Application error contracts used by the HTTP layer.

Keep error codes stable: clients should branch on ``code`` rather than on the
human-readable ``message``.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    HTTP_ERROR = "HTTP_ERROR"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    details: Any | None = None
    request_id: str


class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for an API error")
    error: ErrorDetail


STATUS_TO_ERROR_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.TOO_MANY_REQUESTS,
    502: ErrorCode.UPSTREAM_ERROR,
}


def error_code_for_status(status_code: int) -> ErrorCode:
    """Return the stable public error code for an HTTP status."""
    return STATUS_TO_ERROR_CODE.get(status_code, ErrorCode.HTTP_ERROR)


def build_error_response(
    *,
    code: ErrorCode,
    message: str,
    request_id: str,
    details: Any | None = None,
) -> ErrorResponse:
    """Build the canonical error response object."""
    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
            request_id=request_id,
        )
    )
