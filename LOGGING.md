# Logging and tracing

Every request gets one **trace id**. It is returned in the `X-Request-ID` and `X-Trace-ID` response headers, it is the `request_id` inside every error body, and it is written on **every** log line produced while that request was being served (including lines from worker threads and the background design job). On Cloud Run it is the id Google already assigns (`X-Cloud-Trace-Context`), so the application's lines sit in the same trace as Cloud Run's own request log. If the caller sends its own `X-Request-ID` that value is kept as `request_id` and the trace id stays Google's.

Code: `core/observability.py` (middleware, formatter, masking), `core/error_handlers.py` (what is logged for each kind of failure). Tests: `python test_observability.py`.

## What is written

| Severity | When | Notes |
|---|---|---|
| ERROR | unhandled exception; HTTP 5xx; LLM/Gemini request failed for good | with the full stack trace; **one ERROR per real fault**, so counting ERRORs counts faults |
| WARNING | HTTP 4xx (not 404); validation failures; a single Gemini call that failed; the access line of a 5xx | validation lines name the invalid fields, never the values |
| INFO | access line per request (method, path, status, ms, size, caller IP), scenario confirmed, Gemini call timing, `[Latency]` lines | probe and static-file hits are DEBUG |

Not logged: request or response bodies, prompts, Gemini responses, tokens. Bearer tokens, `api_key=`/`token=`/`password=` pairs, Google API keys and MongoDB URI passwords are masked in everything written.

Useful fields on JSON lines: `trace_id`, `request_id`, `scenario_id`, `draft_id`, `event` (`request`, `http_error`, `validation_error`, `unhandled_exception`, `llm_call`, `llm_call_failed`, `llm_upstream_failure`, `propose_done`, `generate_done`, `scenario_confirmed`), `status`, `duration_ms`, `error_code`, `provider`, `model`, `upstream_status`.

## Settings

Nothing is required. Defaults: JSON lines on Cloud Run (`K_SERVICE` is set there), readable text on a laptop, level INFO. Optional overrides, no rebuild needed: `LOG_LEVEL=DEBUG` and `LOG_FORMAT=json|text`. The project id for the trace link is read from the Cloud Run metadata server; set `GOOGLE_CLOUD_PROJECT` only if the startup line says "GCP project id not found".

## Finding a failure (Cloud Console → Logging → Logs Explorer)

A user reports an error and reads out the request id from the message. Paste (replace `<id>`):

    trace="projects/prod-y-in2300452/traces/<id>"

or, if the id came from a caller-supplied header: `jsonPayload.request_id="<id>"`. "Show entries for this trace" on any line gives the same view.

Other filters (add `resource.type="cloud_run_revision" resource.labels.service_name="ingenii-source-ui"`):

- all errors: `severity>=ERROR`
- Gemini trouble: `jsonPayload.event=("llm_call_failed" OR "llm_upstream_failure")`
- one scenario: `jsonPayload.scenario_id="LB-01"`
- slow requests: `jsonPayload.event="request" jsonPayload.duration_ms>20000`
- server errors by the access line: `httpRequest.status>=500`

The same from a terminal: `gcloud logging read '<filter>' --project=prod-y-in2300452 --limit=50 --freshness=1d`.

**Error Reporting** (Console → Error Reporting) needs no setup: ERROR lines carry the stack trace and `serviceContext`, so identical faults are grouped with counts, first/last seen and the trace of the latest occurrence.

## Alerting (optional)

Log-based alert on any ERROR in the service: Console → Logging → Logs-based alerts → Create, filter `resource.type="cloud_run_revision" resource.labels.service_name="ingenii-source-ui" severity>=ERROR`, rate limit 5 min, pick a notification channel. A gcloud policy file with the same filter is in the delivery notes.
