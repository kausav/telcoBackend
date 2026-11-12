"""Checks for request tracing and structured logging (core/observability.py and its use in core/error_handlers.py).

Run from the backend folder:  python test_observability.py        (needs starlette and pydantic; no database, no network, no Gemini)
When FastAPI is not installed a tiny stand-in for the four names the error handlers import is used, so the real handlers still run.
Drives a small Starlette app through the real TraceContextMiddleware and reads back the JSON lines it writes.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BACKEND_DIR", HERE))
sys.path.insert(0, str(ROOT))

os.environ.update(LOG_FORMAT="json", LOG_LEVEL="DEBUG", GOOGLE_CLOUD_PROJECT="test-proj", K_SERVICE="ingenii-source-ui", K_REVISION="rev-007",
                  MONGODB_URI="mongodb://private-marker", GEMINI_API_KEY="private-marker", API_STATIC_TOKENS="off")

try:                                                         # a stand-in only when FastAPI is missing
    import fastapi  # noqa: F401
except ImportError:
    from starlette.exceptions import HTTPException as _HTTPException
    from starlette.requests import Request as _Request
    from starlette.responses import JSONResponse as _JSONResponse

    class _RequestValidationError(Exception):
        def __init__(self, errors):
            self._errors = errors

        def errors(self):
            return self._errors

    shim = types.ModuleType("fastapi")
    shim.HTTPException, shim.Request = _HTTPException, _Request
    shim_exc = types.ModuleType("fastapi.exceptions")
    shim_exc.RequestValidationError = _RequestValidationError
    shim_resp = types.ModuleType("fastapi.responses")
    shim_resp.JSONResponse = _JSONResponse
    sys.modules.update({"fastapi": shim, "fastapi.exceptions": shim_exc, "fastapi.responses": shim_resp})

from fastapi import HTTPException                                       # noqa: E402
from fastapi.exceptions import RequestValidationError                   # noqa: E402
from starlette.applications import Starlette                            # noqa: E402
from starlette.concurrency import run_in_threadpool                     # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse         # noqa: E402
from starlette.routing import Route                                     # noqa: E402

import logging                                                          # noqa: E402

from core import observability as obs                                   # noqa: E402
from core.error_handlers import (                                       # noqa: E402
    http_exception_handler, llm_upstream_exception_handler, request_validation_exception_handler, unhandled_exception_handler,
)
from core.errors import LLMUpstreamError                                # noqa: E402

TRACE = "105445aa7843bc8bf206b12000100000"
failures: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


child = logging.getLogger("app.demo")


async def ok(request):
    obs.bind(scenario_id="LB-01")
    child.info("doing the work")
    return JSONResponse({"request_id": request.state.request_id})


def threaded_work():
    child.info("inside a worker thread")
    return "done"


async def threaded(request):
    await run_in_threadpool(threaded_work)
    return PlainTextResponse("ok")


async def pool(request):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(obs.propagate_context(lambda n: child.info("pool item %d", n)), [1, 2]))
    return PlainTextResponse("ok")


async def boom(request):
    raise RuntimeError("database exploded password=hunter2 key=AIzaSyA1234567890abcdefghijklmnopqrstuv")


async def http500(request):
    try:
        raise ValueError("root cause here")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail={"error": "internal detail the client must not see"}) from exc


async def http400(request):
    raise HTTPException(status_code=400, detail={"error": "bad input"})


async def http404(request):
    raise HTTPException(status_code=404, detail="nothing here")


async def retry503(request):
    raise HTTPException(status_code=503, detail={"error": "still designing", "retryAfterSeconds": 30})


async def invalid(request):
    raise RequestValidationError([{"loc": ["body", "count"], "msg": "Input should be a valid integer", "type": "int_parsing"}])


async def llm(request):
    try:
        raise TimeoutError("deadline exceeded")
    except TimeoutError as exc:
        raise LLMUpstreamError("Gemini JSON request failed: TimeoutError", provider="Google Gemini", model="gemini-x", status_code=504) from exc


async def health(request):
    return JSONResponse({"status": "ok"})


async def asset(request):
    return PlainTextResponse("a{}")


app = Starlette(
    routes=[Route("/ok", ok), Route("/threaded", threaded), Route("/pool", pool), Route("/boom", boom), Route("/http500", http500),
            Route("/http400", http400), Route("/http404", http404), Route("/retry503", retry503), Route("/invalid", invalid),
            Route("/llm", llm), Route("/health", health), Route("/assets/app.js", asset)],
    exception_handlers={HTTPException: http_exception_handler, RequestValidationError: request_validation_exception_handler,
                        LLMUpstreamError: llm_upstream_exception_handler, Exception: unhandled_exception_handler},
)
app.add_middleware(obs.TraceContextMiddleware)


class Result:
    def __init__(self, status, headers, body, lines):
        self.status, self.headers, self.body, self.lines = status, headers, body, lines

    def json(self):
        return json.loads(self.body)

    def access(self):
        return [r for r in self.lines if r.get("logger") == "access"]

    def by_logger(self, name):
        return [r for r in self.lines if r.get("logger") == name]

    def errors(self):
        return [r for r in self.lines if r["severity"] in ("ERROR", "CRITICAL")]


def call(path: str, headers: dict[str, str] | None = None, method: str = "GET") -> Result:
    """Run one request through the ASGI app, returning the response and the log lines written while it ran."""
    stream = io.StringIO()
    real, sys.stdout = sys.stdout, stream
    try:
        obs.configure_logging(force=True)
        obs.clear_context()
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path, "raw_path": path.encode(),
                 "query_string": b"", "root_path": "", "scheme": "http", "server": ("testserver", 80), "client": ("10.1.2.3", 5555),
                 "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()] + [(b"user-agent", b"selftest")]}
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        async def run():
            try:
                await app(scope, receive, send)
            except Exception:
                pass                                  # the server re-raises after answering 500; a real server logs it, we only need the lines

        asyncio.run(run())
        logging.getLogger().handlers[0].flush()
    finally:
        sys.stdout = real
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    out_headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    return Result(start["status"], out_headers, body, lines)


# ---- header parsing ------------------------------------------------------------------------------------------------------
from starlette.datastructures import Headers   # noqa: E402

t, s, o = obs.parse_trace_headers(Headers({"x-cloud-trace-context": f"{TRACE.upper()}/1234567890123;o=1"}))
check("Cloud Trace header parsed: id lower-cased, span as 16 hex, sampled", (t, s, o) == (TRACE, format(1234567890123, "016x"), True), (t, s, o))
check("Cloud Trace header without span/option still gives the id", obs.parse_trace_headers(Headers({"x-cloud-trace-context": TRACE}))[0] == TRACE)
check("W3C traceparent parsed", obs.parse_trace_headers(Headers({"traceparent": f"00-{TRACE}-00f067aa0ba902b7-01"})) == (TRACE, "00f067aa0ba902b7", True))
check("all-zero or malformed trace ids are rejected",
      obs.parse_trace_headers(Headers({"x-cloud-trace-context": "0" * 32 + "/1"}))[0] is None
      and obs.parse_trace_headers(Headers({"x-cloud-trace-context": "not-a-trace"}))[0] is None
      and obs.parse_trace_headers(Headers({}))[0] is None)

# ---- masking -----------------------------------------------------------------------------------------------------------
masked = obs.redact("Authorization: Bearer abcdefghijkl1234 api_key=SECRET1 token: SECRET2 mongodb+srv://user:p%40ss@cluster0.x.mongodb.net/db AIzaSyA1234567890abcdefghijklmnopqrstuv")
check("secrets are masked", not any(x in masked for x in ("abcdefghijkl1234", "SECRET1", "SECRET2", "p%40ss", "AIzaSyA")) and "user:" in masked, masked)
check("ordinary text is left alone", obs.redact("API authentication: static token required (1 token(s) configured)") == "API authentication: static token required (1 token(s) configured)")

# ---- a normal request ----------------------------------------------------------------------------------------------------
r = call("/ok", {"X-Cloud-Trace-Context": f"{TRACE}/77;o=1"})
check("200 OK", r.status == 200, r.status)
check("response carries X-Request-ID and X-Trace-ID equal to the Cloud trace id", r.headers.get("x-request-id") == TRACE and r.headers.get("x-trace-id") == TRACE, r.headers)
check("handler saw the same id in request.state", r.json()["request_id"] == TRACE)
work = r.by_logger("app.demo")[0]
check("a log line from application code carries the ids without passing them", work.get("request_id") == TRACE and work.get("trace_id") == TRACE, work)
check("... and the Cloud Logging trace link, span and sampled flag",
      work.get("logging.googleapis.com/trace") == f"projects/test-proj/traces/{TRACE}" and work.get("logging.googleapis.com/spanId") == format(77, "016x")
      and work.get("logging.googleapis.com/trace_sampled") is True, work)
check("... and the service context Error Reporting needs", work.get("serviceContext") == {"service": "ingenii-source-ui", "version": "rev-007"}, work.get("serviceContext"))
acc = r.access()[0]
check("one access line with method, status, latency and size", acc["status"] == 200 and acc["httpRequest"]["requestMethod"] == "GET" and acc["httpRequest"]["latency"].endswith("s")
      and acc["httpRequest"]["remoteIp"] == "10.1.2.3" and acc["severity"] == "INFO" and "duration_ms" in acc, acc)
check("a field bound inside the request (scenario_id) reaches the access line", acc.get("scenario_id") == "LB-01", acc)
check("exactly one access line per request", len(r.access()) == 1)

# ---- ids from the caller / generated ---------------------------------------------------------------------------------------
r = call("/ok", {"X-Cloud-Trace-Context": f"{TRACE}/1", "X-Request-ID": "client-abc.123"})
check("a valid caller X-Request-ID is kept, the trace id stays the Cloud one", r.headers["x-request-id"] == "client-abc.123" and r.headers["x-trace-id"] == TRACE, r.headers)
r = call("/ok", {"X-Cloud-Trace-Context": f"{TRACE}/1", "X-Request-ID": "bad id with spaces <script>"})
check("an unsafe X-Request-ID is replaced", r.headers["x-request-id"] == TRACE, r.headers)
r = call("/ok")
generated = r.headers["x-trace-id"]
check("no trace header (local run): a 32-hex id is generated and used for both", len(generated) == 32 and r.headers["x-request-id"] == generated, r.headers)
r2 = call("/ok")
check("every request gets a different generated id", r2.headers["x-trace-id"] != generated)
r = call("/ok", {"traceparent": f"00-{TRACE}-00f067aa0ba902b7-01"})
check("W3C traceparent is honoured when no Cloud header is present", r.headers["x-trace-id"] == TRACE)

# ---- threads ---------------------------------------------------------------------------------------------------------------
r = call("/threaded", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
check("run_in_threadpool keeps the ids", r.by_logger("app.demo")[0].get("request_id") == TRACE, r.by_logger("app.demo"))
r = call("/pool", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
items = r.by_logger("app.demo")
check("ThreadPoolExecutor + propagate_context keeps the ids (both workers)", len(items) == 2 and all(i.get("request_id") == TRACE for i in items), items)

# ---- failures --------------------------------------------------------------------------------------------------------------
r = call("/boom", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
check("unhandled exception answers 500 with the canonical body and the request id", r.status == 500 and r.json()["error"]["request_id"] == TRACE and "hunter2" not in r.body.decode(), (r.status, r.body))
errs = r.errors()
check("exactly one ERROR for the fault, with the traceback and the ids", len(errs) == 1 and "Traceback" in errs[0]["message"] and "RuntimeError" in errs[0]["message"]
      and errs[0]["request_id"] == TRACE and errs[0]["logging.googleapis.com/trace"].endswith(TRACE), errs)
check("the ERROR names the exception type and has a source location", errs[0]["exception"]["type"] == "RuntimeError" and errs[0]["logging.googleapis.com/sourceLocation"]["function"], errs[0])
check("secrets in the exception text are masked in the log", "hunter2" not in errs[0]["message"] and "AIzaSy" not in errs[0]["message"], errs[0]["message"][:300])
acc = r.access()[0]
check("the access line for a 500 is WARNING status 500 (not a second ERROR)", acc["status"] == 500 and acc["severity"] == "WARNING", acc)

r = call("/http500", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
errs = r.errors()
check("HTTPException 500: client sees a generic message", r.status == 500 and "internal detail" not in r.body.decode(), r.body)
check("... the log keeps the real detail and the root cause chain", len(errs) == 1 and "internal detail the client must not see" in errs[0]["message"] and "root cause here" in errs[0]["message"], errs)

r = call("/http400", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
warn = [x for x in r.lines if x["severity"] == "WARNING" and x["logger"] != "access"]
check("4xx is a WARNING with the reason, never an ERROR", r.status == 400 and len(warn) == 1 and "bad input" in warn[0]["message"] and not r.errors(), r.lines)
r = call("/http404", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
check("404 is INFO only", r.status == 404 and not r.errors() and not [x for x in r.lines if x["severity"] == "WARNING"], r.lines)
r = call("/retry503", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
check("a retryable 503 (design still running) is not an error", r.status == 503 and not r.errors(), r.lines)
r = call("/invalid", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
warn = [x for x in r.lines if x["severity"] == "WARNING" and x["logger"] != "access"]
check("validation errors: 422, one WARNING naming the fields", r.status == 422 and len(warn) == 1 and warn[0].get("invalid_fields") == ["count"] and not r.errors(), r.lines)
r = call("/llm", {"X-Cloud-Trace-Context": f"{TRACE}/1"})
errs = r.errors()
check("LLM failure: 502, one ERROR with provider/model/status and the cause", r.status == 502 and len(errs) == 1 and errs[0].get("provider") == "Google Gemini"
      and errs[0].get("upstream_status") == 504 and "TimeoutError" in errs[0]["message"], (r.status, errs))
check("the error body's request_id is the trace id to search for", r.json()["error"]["request_id"] == TRACE)

# ---- noise control -----------------------------------------------------------------------------------------------------------
logging.getLogger().setLevel(logging.INFO)
os.environ["LOG_LEVEL"] = "INFO"
r = call("/health")
check("probe calls are not logged at INFO", not r.access(), r.lines)
r = call("/assets/app.js")
check("static asset hits are not logged at INFO", not r.access(), r.lines)
os.environ["LOG_LEVEL"] = "DEBUG"

# ---- text format for local development --------------------------------------------------------------------------------------
os.environ["LOG_FORMAT"] = "text"
stream = io.StringIO()
real, sys.stdout = sys.stdout, stream
try:
    obs.configure_logging(force=True)
    obs.clear_context()
    obs.bind(request_id="abc123")
    logging.getLogger("demo").warning("hello token=SECRET9")
finally:
    sys.stdout = real
text = stream.getvalue()
check("text format shows the request id, is one line and masks secrets", "[abc123]" in text and "SECRET9" not in text and text.count("\n") == 1, text)
os.environ["LOG_FORMAT"] = "json"
obs.configure_logging(force=True)
check("configure_logging(force=True) never stacks a second handler", sum(1 for h in logging.getLogger().handlers if h.get_name() == "modelos-log") == 1)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: " + "; ".join(failures))
    sys.exit(1)
print("All checks passed.")
