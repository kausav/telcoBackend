"""Compiled scenario schema/cache used by the fast generation path.

The public API schema is unchanged.  This module turns a confirmed scenario into
small lookup tables once and reuses them for subsequent /scenario/generate calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from core.dynamic_scenarios import (
    resolve_entity_key,
    resolve_events,
    resolve_variables,
)


@dataclass(frozen=True)
class CompiledEvent:
    event_type: str
    sequence: int
    fields: tuple[str, ...]
    min_occurrences: int
    max_occurrences: int


@dataclass(frozen=True)
class CompiledScenario:
    scenario_id: str
    entity_key: str | None
    variables: tuple[dict[str, Any], ...]
    field_order: tuple[str, ...]
    variable_by_name: dict[str, dict[str, Any]]
    entity_fields: tuple[str, ...]
    events: tuple[CompiledEvent, ...]
    event_by_type: dict[str, CompiledEvent]


_CACHE: dict[str, CompiledScenario] = {}
_LOCK = RLock()


def _dependency_closure(variable_by_name: dict[str, dict[str, Any]], names: set[str]) -> set[str]:
    """Include declared dependencies so formula/time generators have their inputs."""
    required = set(names)
    changed = True
    while changed:
        changed = False
        for name in tuple(required):
            var = variable_by_name.get(name)
            if not var:
                continue
            for dep in var.get("depends_on", []) or []:
                if dep in variable_by_name and dep not in required:
                    required.add(dep)
                    changed = True
    return required


def compile_scenario(scenario_id: str, force: bool = False) -> CompiledScenario:
    with _LOCK:
        if not force and scenario_id in _CACHE:
            return _CACHE[scenario_id]

        resolved = resolve_variables(scenario_id)
        if resolved is None:
            from config.variables import VARIABLES, FIELD_ORDER
            variables = VARIABLES
            field_order = FIELD_ORDER
        else:
            variables, field_order = resolved

        entity_key = resolve_entity_key(scenario_id)
        events_raw = resolve_events(scenario_id)

        variable_by_name = {str(v["name"]): v for v in variables}
        event_field_names: set[str] = set()
        compiled_events: list[CompiledEvent] = []

        for position, event in enumerate(events_raw or [{"event_type": "BUSINESS_EVENT", "sequence": 1, "fields": []}], start=1):
            event_type = str(event.get("event_type", "BUSINESS_EVENT")).strip() or "BUSINESS_EVENT"
            sequence = int(event.get("sequence", position))
            fields = tuple(
                str(name) for name in (event.get("fields", []) or [])
                if str(name) in variable_by_name
            )
            event_field_names.update(fields)
            min_occ = max(1, int(event.get("min_occurrences", 1)))
            max_occ = max(min_occ, int(event.get("max_occurrences", 20)))
            compiled_events.append(CompiledEvent(event_type, sequence, fields, min_occ, max_occ))

        # Everything not explicitly owned by an event is stable entity/context data.
        # Always include entity_key even if the designer omitted it from event fields.
        entity_names = set(variable_by_name) - event_field_names
        if entity_key and entity_key in variable_by_name:
            entity_names.add(entity_key)

        # Entity fields also need their dependencies, but dependencies used only by
        # an event remain event-local and are generated only when that event is emitted.
        entity_names = _dependency_closure(variable_by_name, entity_names)
        entity_fields = tuple(v["name"] for v in variables if v["name"] in entity_names)

        compiled = CompiledScenario(
            scenario_id=scenario_id,
            entity_key=entity_key,
            variables=tuple(variables),
            field_order=tuple(field_order),
            variable_by_name=variable_by_name,
            entity_fields=entity_fields,
            events=tuple(compiled_events),
            event_by_type={e.event_type: e for e in compiled_events},
        )
        _CACHE[scenario_id] = compiled
        return compiled


def invalidate_scenario(scenario_id: str) -> None:
    with _LOCK:
        _CACHE.pop(scenario_id, None)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
