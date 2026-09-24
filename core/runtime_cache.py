"""Small process-local caches for immutable runtime artifacts."""
from __future__ import annotations
from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Any
import os

_LOCK = RLock()
_MAX_ITEMS = max(16, int(os.getenv("RUNTIME_CACHE_MAX_ITEMS", "256")))
_PROPOSAL_TTL = max(30, int(os.getenv("PROPOSAL_CACHE_TTL_SECONDS", "1800")))
_SCHEMA: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_ORCHESTRATOR: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_PROPOSAL: OrderedDict[tuple, tuple[float, dict[str, Any]]] = OrderedDict()


def _bounded_set(cache: OrderedDict, key: tuple, value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_ITEMS:
        cache.popitem(last=False)


def get_schema(key: tuple) -> dict[str, Any] | None:
    with _LOCK:
        value = _SCHEMA.get(key)
        if value is not None:
            _SCHEMA.move_to_end(key)
        return value


def set_schema(key: tuple, value: dict[str, Any]) -> None:
    with _LOCK:
        _bounded_set(_SCHEMA, key, value)


def get_orchestrator(key: tuple) -> dict[str, Any] | None:
    with _LOCK:
        value = _ORCHESTRATOR.get(key)
        if value is not None:
            _ORCHESTRATOR.move_to_end(key)
        return value


def set_orchestrator(key: tuple, value: dict[str, Any]) -> None:
    with _LOCK:
        _bounded_set(_ORCHESTRATOR, key, value)


def get_proposal(key: tuple) -> dict[str, Any] | None:
    with _LOCK:
        item = _PROPOSAL.get(key)
        if item is None:
            return None
        expires_at, value = item
        if monotonic() >= expires_at:
            _PROPOSAL.pop(key, None)
            return None
        _PROPOSAL.move_to_end(key)
        return value


def set_proposal(key: tuple, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
    ttl = max(30, int(ttl_seconds or _PROPOSAL_TTL))
    with _LOCK:
        _bounded_set(_PROPOSAL, key, (monotonic() + ttl, value))


def clear_scenario(scenario_id: str) -> None:
    """Clear generation caches associated with a persisted scenario.

    Proposal caches are intentionally not cleared by raw scenario ID. They are keyed by
    semantic request inputs plus a fingerprint of the persisted recommendation names, so
    recommendation-set changes naturally select a different cache entry.
    """
    with _LOCK:
        for cache in (_SCHEMA, _ORCHESTRATOR):
            for key in list(cache):
                if key and key[0] == scenario_id:
                    cache.pop(key, None)
