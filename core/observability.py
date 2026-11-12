"""Request tracing and structured logging.

What this gives an operator
---------------------------
* Every request gets one **trace id**. On Cloud Run it is the id Google already put in ``X-Cloud-Trace-Context`` (so the id in
  the application logs is the same one Cloud Run's own request log and Cloud Trace use); anywhere else it is generated. Unless the
  caller sent its own ``X-Request-ID``, the request id is the trace id, so the id a user reads out of an error body is the id to
  search for in the logs.
* Every log line written while that request is being served carries the ids (``trace_id``, ``request_id``) without any call site
  passing them: they travel in ``contextvars``, which asyncio tasks, ``run_in_threadpool`` and (via ``propagate_context``)
  plain thread pools all inherit.
* On Cloud Run the logs are one JSON object per line on stdout. Cloud Logging turns the special keys into first-class fields:
  ``severity``, ``logging.googleapis.com/trace`` (groups every line of one request, "show entries for this trace"),
  ``logging.googleapis.com/spanId``, ``httpRequest`` and ``logging.googleapis.com/sourceLocation``. ``serviceContext`` plus a stack
  trace in ``message`` is what Error Reporting groups on. Locally the same lines are plain text.
* One access-log line per request (method, path, status, latency, size, caller IP) and nothing from the request body is logged.
* Secrets are masked in everything written (bearer tokens, API keys, ``password=``/``token=`` pairs, MongoDB URI passwords).

Nothing here needs a dependency or a new environment variable: the project id comes from the Cloud Run metadata server (or
``GOOGLE_CLOUD_PROJECT`` when set), and ``LOG_LEVEL`` / ``LOG_FORMAT`` are optional overrides of ``config.constants``.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from urllib.request import ProxyHandler, Request as _UrlRequest, build_opener

from starlette.datastructures import Headers, MutableHeaders

from config import constants as _c

REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"

_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "WARN": logging.WARNING,
           "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
_STATIC_SUFFIXES = (".js", ".css", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".txt")


# ---- Request context -------------------------------------------------------------------------------------------------
# The value is one mutable dict that every task/thread of the request shares by reference, so a field bound deep inside the
# request (the scenario id, an error code) is visible to the access-log line the outermost middleware writes at the end.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("modelos_log_context", default=None)


def get_context() -> dict[str, Any]:
    return _context.get() or {}


def bind(**fields: Any) -> None:
    """Attach fields (``scenario_id=...``) to every later log line of the current request. ``None`` values are ignored."""
    current = _context.get()
    if current is None:
        current = {}
        _context.set(current)
    for key, value in fields.items():
        if value is not None:
            current[key] = value


def clear_context() -> None:
    _context.set(None)


def current_trace_id() -> str | None:
    return get_context().get("trace_id")


def current_request_id() -> str | None:
    return get_context().get("request_id")


def mark_error_logged() -> None:
    """Remember that the traceback of this request's failure is already in the log (see ``_DropDuplicateAsgiError``)."""
    bind(_error_logged=True)


def propagate_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` for ``ThreadPoolExecutor.submit/map`` so its log lines keep the request's ids (pools do not copy contextvars)."""
    base = contextvars.copy_context()

    def runner(*args: Any, **kwargs: Any) -> Any:
        return base.copy().run(fn, *args, **kwargs)

    return runner


# ---- Trace headers ---------------------------------------------------------------------------------------------------
_CLOUD_TRACE = re.compile(r"^([0-9a-fA-F]{32})(?:/(\d{1,20}))?(?:;o=(\d))?$")
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_ZERO32 = "0" * 32


def parse_trace_headers(headers: Any) -> tuple[str | None, str | None, bool | None]:
    """(trace id, span id as 16 hex chars, sampled) from ``X-Cloud-Trace-Context`` or W3C ``traceparent``; Nones when absent/invalid."""
    raw = str(headers.get("x-cloud-trace-context") or "").strip()
    match = _CLOUD_TRACE.match(raw)
    if match and match.group(1) != _ZERO32:
        span = None
        if match.group(2):
            try:
                span = format(int(match.group(2)), "016x") if int(match.group(2)) < 2 ** 64 else None
            except ValueError:
                span = None
        sampled = None if match.group(3) is None else match.group(3) == "1"
        return match.group(1).lower(), span, sampled
    parent = _TRACEPARENT.match(str(headers.get("traceparent") or "").strip().lower())
    if parent and parent.group(1) != _ZERO32:
        return parent.group(1), parent.group(2), bool(int(parent.group(3), 16) & 1)
    return None, None, None


# ---- Secret masking --------------------------------------------------------------------------------------------------
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|x-api-key|x-goog-api-key|access[_-]?token|auth[_-]?token|token|authorization|password|passwd|secret)"
                r"([\"']?\s*[:=]\s*[\"']?)[^\s,;\"'&}]+"), r"\1\2[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "[REDACTED]"),
    (re.compile(r"(?i)(mongodb(?:\+srv)?://[^/\s:@]+:)[^@\s/]+@"), r"\1[REDACTED]@"),
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…[truncated {len(text) - limit} chars]"


# ---- Formatters ------------------------------------------------------------------------------------------------------
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}
_OWN_KEYS = frozenset({"severity", "message", "timestamp", "logger", "trace_id", "request_id", "serviceContext",
                       "logging.googleapis.com/trace", "logging.googleapis.com/spanId", "logging.googleapis.com/sourceLocation",
                       "logging.googleapis.com/trace_sampled", "exception"})


def _exception_text(formatter: logging.Formatter, record: logging.LogRecord) -> str:
    if record.exc_info and record.exc_info[0] is not None:
        return formatter.formatException(record.exc_info)
    return record.exc_text or ""


class JsonFormatter(logging.Formatter):
    """One JSON object per line in the shape Cloud Logging and Error Reporting understand."""

    def __init__(self, project_id: str = "", service: str = "", revision: str = "", max_chars: int = 20_000) -> None:
        super().__init__()
        self.project_id = project_id
        self.service = service or "modelos-source"
        self.revision = revision
        self.max_chars = max_chars

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        trace_back = _exception_text(self, record)
        if trace_back:
            message = f"{message}\n{trace_back}"          # Error Reporting reads the stack trace from the message text
        payload: dict[str, Any] = {
            "severity": record.levelname if record.levelname in _LEVELS else "DEFAULT",
            "message": _clip(redact(message), self.max_chars),
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "logger": record.name,
            "serviceContext": {"service": self.service, **({"version": self.revision} if self.revision else {})},
        }
        context = dict(get_context())
        trace_id = getattr(record, "trace_id", None) or context.pop("trace_id", None)
        context.pop("trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id
            if self.project_id:
                payload["logging.googleapis.com/trace"] = f"projects/{self.project_id}/traces/{trace_id}"
        span_id = context.pop("span_id", None)
        if span_id:
            payload["logging.googleapis.com/spanId"] = span_id
        sampled = context.pop("trace_sampled", None)
        if sampled is not None:
            payload["logging.googleapis.com/trace_sampled"] = bool(sampled)
        request_id = context.pop("request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = {"type": record.exc_info[0].__name__, "message": _clip(redact(str(record.exc_info[1])), 1000)}
        if record.levelno >= logging.WARNING:
            payload["logging.googleapis.com/sourceLocation"] = {"file": record.pathname, "line": str(record.lineno), "function": record.funcName}

        for key, value in context.items():                 # fields bound for this request (scenario_id, error_code, ...)
            if not key.startswith("_") and key not in _OWN_KEYS:
                payload[key] = value
        for key, value in record.__dict__.items():          # fields passed as logger.xxx(..., extra={...})
            if key in _STANDARD_ATTRS or key in _OWN_KEYS or key.startswith("_"):
                continue
            payload[key] = value
        for key, value in list(payload.items()):
            if isinstance(value, str) and key not in {"message", "timestamp", "logger"}:
                payload[key] = _clip(redact(value), 2000)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            return json.dumps({"severity": payload["severity"], "message": payload["message"], "request_id": request_id}, default=str)


class TextFormatter(logging.Formatter):
    """Readable single-line format for a developer's terminal; same ids, same masking."""

    def __init__(self, max_chars: int = 20_000) -> None:
        super().__init__()
        self.max_chars = max_chars

    def format(self, record: logging.LogRecord) -> str:
        context = get_context()
        rid = getattr(record, "request_id", None) or context.get("request_id") or "-"
        stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        line = f"{stamp} {record.levelname:<8} [{rid}] {record.name}: {record.getMessage()}"
        trace_back = _exception_text(self, record)
        if trace_back:
            line = f"{line}\n{trace_back}"
        return _clip(redact(line), self.max_chars)


class ContextFilter(logging.Filter):
    """Copy the request's ids onto the record so every handler/formatter sees them, including records from other libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = get_context()
        for key in ("trace_id", "request_id"):
            if not getattr(record, key, None) and context.get(key):
                setattr(record, key, context[key])
        return True


class _DropDuplicateAsgiError(logging.Filter):
    """Uvicorn re-logs "Exception in ASGI application" for a 500 our handler already logged with its traceback; keep one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if get_context().get("_error_logged") and str(record.msg).startswith("Exception in ASGI application"):
            return False
        return True


# ---- Configuration ---------------------------------------------------------------------------------------------------
_state: dict[str, Any] = {"configured": False, "project_id": "", "service": "", "revision": "", "format": "text"}


def _level() -> int:
    return _LEVELS.get(str(os.getenv("LOG_LEVEL") or _c.LOG_LEVEL).strip().upper(), logging.INFO)


def _format_name() -> str:
    choice = str(os.getenv("LOG_FORMAT") or _c.LOG_FORMAT).strip().lower()
    if choice in {"json", "text"}:
        return choice
    return "json" if os.getenv("K_SERVICE") else "text"


def resolve_project_id() -> str:
    """The GCP project id, for the ``logging.googleapis.com/trace`` field. Empty when it cannot be found (the trace id is still logged)."""
    for name in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    if os.getenv("K_SERVICE"):                                # on Cloud Run: ask the metadata server (one short call, at start-up)
        try:
            request = _UrlRequest("http://metadata.google.internal/computeMetadata/v1/project/project-id", headers={"Metadata-Flavor": "Google"})
            with build_opener(ProxyHandler({})).open(request, timeout=1.5) as response:
                return response.read().decode("utf-8").strip()
        except Exception:
            return ""
    return ""


def _install_excepthooks() -> None:
    crash = logging.getLogger("crash")

    def on_main(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
        else:
            crash.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def on_thread(args):
        if args.exc_type is not SystemExit:
            crash.error("Uncaught exception in thread %s", getattr(args.thread, "name", "?"), exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = on_main
    threading.excepthook = on_thread


def configure_logging(force: bool = False) -> None:
    """Install the log handler once per process. Safe to call again (a no-op unless ``force``)."""
    if _state["configured"] and not force:
        return
    level = _level()
    fmt = _format_name()
    max_chars = int(_c.LOG_MAX_MESSAGE_CHARS)
    project_id = resolve_project_id() if fmt == "json" else ""
    service = os.getenv("K_SERVICE") or "modelos-source"
    revision = os.getenv("K_REVISION") or ""

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name("modelos-log")
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter(project_id, service, revision, max_chars) if fmt == "json" else TextFormatter(max_chars))

    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == "modelos-log":
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn installs its own handlers; route its error/server lines through ours and replace its access line with ours.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.error").addFilter(_DropDuplicateAsgiError())
    access = logging.getLogger("uvicorn.access")
    access.handlers[:] = [logging.NullHandler()]
    access.propagate = False

    for name in _c.LOG_QUIET_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    logging.captureWarnings(True)
    _install_excepthooks()

    _state.update(configured=True, project_id=project_id, service=service, revision=revision, format=fmt)


def log_startup_banner(logger: logging.Logger) -> None:
    logger.info(
        "Logging ready: format=%s level=%s service=%s revision=%s project=%s",
        _state["format"], logging.getLevelName(logging.getLogger().level), _state["service"], _state["revision"] or "-",
        _state["project_id"] or ("unknown" if _state["format"] == "json" else "-"),
        extra={"event": "startup"},
    )
    if _state["format"] == "json" and not _state["project_id"]:
        logger.warning("GCP project id not found: logs carry trace_id but not the logging.googleapis.com/trace link. "
                       "Set GOOGLE_CLOUD_PROJECT to the project id to enable it.")


# ---- Middleware ------------------------------------------------------------------------------------------------------
class TraceContextMiddleware:
    """Outermost ASGI wrapper: assigns the ids, echoes them in the response headers and writes the access-log line.

    Plain ASGI (not ``BaseHTTPMiddleware``) so it adds no task hop, never buffers a streaming body and sees the final status even
    when an inner layer raises. It also sets ``request.state.request_id`` / ``trace_id`` for the existing error handlers.
    """

    def __init__(self, app: Any, access_log: bool = True) -> None:
        self.app = app
        self.access_log = access_log
        self.logger = logging.getLogger("access")

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        trace_id, span_id, sampled = parse_trace_headers(headers)
        incoming = str(headers.get("x-request-id") or "").strip()
        trace_id = trace_id or uuid.uuid4().hex
        request_id = incoming if _REQUEST_ID.match(incoming) else trace_id
        method = scope.get("method", "")
        path = scope.get("path", "")
        forwarded = str(headers.get("x-forwarded-for") or "").split(",")[0].strip()
        client = forwarded or ((scope.get("client") or ("", 0))[0] or "")

        _context.set({"trace_id": trace_id, "request_id": request_id, "span_id": span_id, "trace_sampled": sampled,
                      "http_method": method, "path": path})
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["trace_id"] = trace_id

        status: int | None = None
        sent = 0
        failed = False
        started = perf_counter()

        async def send_with_ids(message: dict) -> None:
            nonlocal status, sent
            if message["type"] == "http.response.start":
                status = int(message["status"])
                out = MutableHeaders(scope=message)
                out[REQUEST_ID_HEADER] = request_id
                out[TRACE_ID_HEADER] = trace_id
            elif message["type"] == "http.response.body":
                sent += len(message.get("body") or b"")
            await send(message)

        try:
            await self.app(scope, receive, send_with_ids)
        except Exception:
            failed = True
            raise
        finally:
            if self.access_log:
                self._log(method, path, status, failed, perf_counter() - started, sent, client, headers, scope)

    def _log(self, method: str, path: str, status: int | None, failed: bool, seconds: float, sent: int, client: str,
             headers: Headers, scope: dict) -> None:
        aborted = status is None
        shown = status if status is not None else (500 if failed else 499)       # 499: the caller went away before a response
        quiet = path.endswith(tuple(_c.LOG_QUIET_PATHS)) or (method == "GET" and path.lower().endswith(_STATIC_SUFFIXES))
        if shown >= 500:
            level = logging.WARNING        # the ERROR line, with the traceback, is written by the error handler: one ERROR per fault
        elif quiet and shown < 400:
            level = logging.DEBUG
        else:
            level = logging.INFO
        if not self.logger.isEnabledFor(level):
            return
        elapsed_ms = round(seconds * 1000.0, 1)
        self.logger.log(
            level,
            "%s %s -> %d in %.0fms",
            method, path, shown, elapsed_ms,
            extra={
                "event": "request",
                "status": shown,
                "duration_ms": elapsed_ms,
                "httpRequest": {
                    "requestMethod": method,
                    "requestUrl": path,
                    "status": shown,
                    "responseSize": str(sent),
                    "userAgent": str(headers.get("user-agent") or "")[:300],
                    "remoteIp": client,
                    "latency": f"{seconds:.3f}s",
                    "protocol": f"HTTP/{scope.get('http_version', '1.1')}",
                },
                **({"client_disconnected": True} if aborted and not failed else {}),
            },
        )
