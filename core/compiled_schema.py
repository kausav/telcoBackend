"""Compiled user-history/flat-record schema used by the fast generation path."""
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from typing import Any
from core.dynamic_scenarios import resolve_entity_key, resolve_variables

@dataclass(frozen=True)
class CompiledScenario:
    scenario_id: str
    entity_key: str | None
    variables: tuple[dict[str, Any], ...]
    field_order: tuple[str, ...]
    variable_by_name: dict[str, dict[str, Any]]
    user_fields: tuple[str, ...]
    record_fields: tuple[str, ...]

_CACHE={}; _LOCK=RLock()

def infer_history_field_sets(variables: list[dict[str, Any]], entity_key: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer stable user fields without requiring a scope column in the CSV.

    Transactional definitions are interpreted as: fields before the first
    datetime/date history anchor belong to the user context; fields from that
    anchor onward belong to repeated history records. The entity key and its
    dependency closure are always stable user fields. This keeps the CSV simple
    while preserving deterministic per-user history generation.
    """
    by_name={str(v.get('name')):v for v in variables if v.get('name')}
    names=[str(v['name']) for v in variables if v.get('name')]

    # Agentic proposals carry an explicit scope/grain assigned by the compiler.
    # When present, trust it over positional heuristics so transactional status,
    # amounts, decisions, timestamps, etc. are regenerated per transaction rather
    # than being frozen at the user/entity level. CSV imports without scope retain
    # the backwards-compatible timestamp-anchor heuristic below.
    scoped = {
        name: str(var.get("scope") or "").strip().lower()
        for name, var in by_name.items()
        if str(var.get("scope") or "").strip()
    }
    if scoped:
        stable = {name for name, scope in scoped.items() if scope == "entity"}
        if entity_key and entity_key in by_name:
            stable.add(entity_key)
        changed=True
        while changed:
            changed=False
            for name in tuple(stable):
                var=by_name.get(name)
                for dep in (var.get("depends_on", []) if isinstance(var, dict) else []) or []:
                    if dep in by_name and dep not in stable and scoped.get(dep) == "entity":
                        stable.add(dep); changed=True
        user_fields=tuple(n for n in names if n in stable)
        record_fields=tuple(n for n in names if n not in stable)
        return user_fields, record_fields

    timestamp_index=None
    preferred=("record_timestamp","transaction_timestamp","timestamp","created_at","updated_at")
    for preferred_name in preferred:
        for i,v in enumerate(variables):
            if str(v.get('name',''))==preferred_name and str(v.get('dtype','')).lower() in {'datetime','date'}:
                timestamp_index=i; break
        if timestamp_index is not None: break
    if timestamp_index is None:
        for i,v in enumerate(variables):
            if str(v.get('dtype','')).lower() in {'datetime','date'}:
                timestamp_index=i; break

    user_names=set()
    if entity_key and entity_key in by_name:
        user_names.add(entity_key)
    # Fields before the first history timestamp are stable user context.
    if timestamp_index is not None:
        user_names.update(names[:timestamp_index])

    # Keep all dependencies of stable user fields stable as well.
    changed=True
    while changed:
        changed=False
        for name in tuple(user_names):
            var=by_name.get(name)
            if not var: continue
            for dep in var.get('depends_on',[]) or []:
                if dep in by_name and dep not in user_names:
                    user_names.add(dep); changed=True

    user_fields=tuple(n for n in names if n in user_names)
    record_fields=tuple(n for n in names if n not in user_names)
    return user_fields, record_fields

def compile_scenario(scenario_id: str, force: bool=False)->CompiledScenario:
    with _LOCK:
        if not force and scenario_id in _CACHE: return _CACHE[scenario_id]
        resolved=resolve_variables(scenario_id)
        if resolved is None: raise ValueError(f"Unknown scenario '{scenario_id}'")
        variables,field_order=resolved
        entity_key=resolve_entity_key(scenario_id)
        by_name={str(v['name']):v for v in variables}
        user_fields,record_fields=infer_history_field_sets(variables, entity_key)
        compiled=CompiledScenario(scenario_id,entity_key,tuple(variables),tuple(field_order),by_name,user_fields,record_fields)
        _CACHE[scenario_id]=compiled
        return compiled

def invalidate_scenario(scenario_id:str)->None:
    with _LOCK: _CACHE.pop(scenario_id,None)
