"""Where verified generation specs are kept: MongoDB (``generation_specs``) with a process-local cache.

A spec is looked up by its key, which hashes everything it was compiled from, so an entry never changes meaning.
A database that cannot be reached degrades to the process cache (the spec is then recompiled after a restart); it never
fails a request.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from synth.spec import GenerationSpec

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SPECS: dict[str, GenerationSpec] = {}
_FAILED: dict[str, float] = {}
FAILURE_TTL_SECONDS = 300.0


def _model():
    from models.generation_spec import GenerationSpecModel

    return GenerationSpecModel


def load(key: str | None) -> GenerationSpec | None:
    if not key:
        return None
    with _LOCK:
        cached = _SPECS.get(key)
    if cached is not None:
        return cached
    try:
        document = _model().get(key)
    except Exception as exc:
        logger.warning("generation spec %s could not be read from MongoDB: %s: %s", key, type(exc).__name__, exc)
        return None
    if not document:
        return None
    try:
        spec = GenerationSpec.model_validate(document["spec"])
    except Exception as exc:        # a stored spec this version cannot read is the same as no spec: it is recompiled
        logger.warning("stored generation spec %s is not valid for this version: %s", key, exc)
        return None
    with _LOCK:
        _SPECS[key] = spec
    return spec


def save(key: str, spec: GenerationSpec, meta: dict[str, Any]) -> bool:
    """Persist ``spec`` under ``key``. Returns False when only the process cache holds it."""
    with _LOCK:
        _SPECS[key] = spec
        _FAILED.pop(key, None)
    try:
        _model().put_if_absent(key, spec.model_dump(mode="json"), meta)
        return True
    except Exception as exc:
        logger.warning("generation spec %s could not be stored in MongoDB: %s: %s", key, type(exc).__name__, exc)
        return False


def remember_failure(key: str) -> None:
    with _LOCK:
        _FAILED[key] = time.monotonic()


def recently_failed(key: str) -> bool:
    """True while a compile of ``key`` failed so recently that retrying would only repeat the wait."""
    with _LOCK:
        at = _FAILED.get(key)
    return at is not None and time.monotonic() - at < FAILURE_TTL_SECONDS


def clear_cache() -> None:
    with _LOCK:
        _SPECS.clear()
        _FAILED.clear()
