"""Where verified generation specs are kept: MongoDB (``generation_specs``) with a process-local cache.

A spec is looked up by its key, which hashes everything it was compiled from, so an entry never changes meaning.
A spec can later be replaced by a better revision of itself (a refinement); a process re-reads what it cached after
``CACHE_SECONDS`` so that several workers converge on the newest revision. A database that cannot be reached degrades to
the process cache (the spec is then recompiled after a restart); it never fails a request.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from synth.spec import GenerationSpec

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SPECS: dict[str, tuple[GenerationSpec, float]] = {}
_FAILED: dict[str, tuple[float, str, float]] = {}      # key -> (when, why, how long a retry is held back)
FAILURE_TTL_SECONDS = 300.0
TRANSIENT_FAILURE_TTL_SECONDS = 30.0
CACHE_SECONDS = 60.0


def _model():
    from models.generation_spec import GenerationSpecModel

    return GenerationSpecModel


def load(key: str | None) -> GenerationSpec | None:
    if not key:
        return None
    with _LOCK:
        cached = _SPECS.get(key)
    if cached is not None and time.monotonic() - cached[1] < CACHE_SECONDS:
        return cached[0]
    try:
        document = _model().get(key)
    except Exception as exc:
        logger.warning("generation spec %s could not be read from MongoDB: %s: %s", key, type(exc).__name__, exc)
        return cached[0] if cached is not None else None
    if not document:
        return cached[0] if cached is not None else None
    try:
        spec = GenerationSpec.model_validate(document["spec"])
    except Exception as exc:        # a stored spec this version cannot read is the same as no spec: it is recompiled
        logger.warning("stored generation spec %s is not valid for this version: %s", key, exc)
        return cached[0] if cached is not None else None
    with _LOCK:
        _SPECS[key] = (spec, time.monotonic())
    return spec


def save(key: str, spec: GenerationSpec, meta: dict[str, Any]) -> bool:
    """Persist ``spec`` under ``key`` (a revision above 0 replaces an earlier revision). Returns False when only the process cache holds it."""
    with _LOCK:
        current = _SPECS.get(key)
        if current is not None and current[0].revision > spec.revision:
            return True                                     # a newer revision is already here
        _SPECS[key] = (spec, time.monotonic())
        _FAILED.pop(key, None)
    try:
        if spec.revision:
            _model().put_revision(key, spec.model_dump(mode="json"), meta, spec.revision)
        else:
            _model().put_if_absent(key, spec.model_dump(mode="json"), meta)
        return True
    except Exception as exc:
        logger.warning("generation spec %s could not be stored in MongoDB: %s: %s", key, type(exc).__name__, exc)
        return False


def annotate(key: str, spec: GenerationSpec, warnings: list[str], reviewed: bool = True) -> GenerationSpec:
    """The same spec with the outcome of a review recorded on it (behaviour and revision unchanged)."""
    updated = spec.model_copy(update={"warnings": warnings, "reviewed": reviewed})
    with _LOCK:
        current = _SPECS.get(key)
        if current is None or current[0].revision <= updated.revision:
            _SPECS[key] = (updated, time.monotonic())
    try:
        _model().set_review(key, warnings, reviewed)
    except Exception as exc:
        logger.warning("review of generation spec %s could not be stored in MongoDB: %s: %s", key, type(exc).__name__, exc)
    return updated


def remember_failure(key: str, reason: str = "", *, transient: bool = False) -> None:
    """Hold back a retry of ``key`` for a while; a failure that may pass by itself (the model was unreachable) only briefly."""
    with _LOCK:
        _FAILED[key] = (time.monotonic(), reason, TRANSIENT_FAILURE_TTL_SECONDS if transient else FAILURE_TTL_SECONDS)


def recently_failed(key: str) -> bool:
    """True while a compile of ``key`` failed so recently that retrying would only repeat the wait."""
    with _LOCK:
        entry = _FAILED.get(key)
    return entry is not None and time.monotonic() - entry[0] < entry[2]


def failure_reason(key: str | None) -> str:
    """Why the last design of ``key`` failed ('' when it did not, or is not remembered)."""
    with _LOCK:
        entry = _FAILED.get(key) if key else None
    return entry[1] if entry else ""


def clear_cache() -> None:
    with _LOCK:
        _SPECS.clear()
        _FAILED.clear()
