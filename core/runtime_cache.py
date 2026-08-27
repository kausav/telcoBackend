"""Small process-local caches for immutable confirmed-scenario runtime artifacts.

The confirmed scenario itself remains persisted in SQLite. These caches only avoid
rebuilding the same schema/rules/validation plan on every /scenario/generate call.
They are safe to miss: a cache miss simply recomputes the artifact.
"""
from __future__ import annotations
from threading import RLock
from typing import Any

_LOCK = RLock()
_RULES: dict[tuple, dict[str, Any]] = {}
_ORCHESTRATOR: dict[tuple, dict[str, Any]] = {}


def get_rules(key: tuple):
    with _LOCK:
        return _RULES.get(key)


def set_rules(key: tuple, value: dict[str, Any]) -> None:
    with _LOCK:
        _RULES[key] = value


def get_orchestrator(key: tuple):
    with _LOCK:
        return _ORCHESTRATOR.get(key)


def set_orchestrator(key: tuple, value: dict[str, Any]) -> None:
    with _LOCK:
        _ORCHESTRATOR[key] = value


def clear_scenario(scenario_id: str) -> None:
    with _LOCK:
        for cache in (_RULES, _ORCHESTRATOR):
            for key in list(cache):
                if key and key[0] == scenario_id:
                    cache.pop(key, None)
