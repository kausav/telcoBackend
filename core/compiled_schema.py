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

def compile_scenario(scenario_id: str, force: bool=False)->CompiledScenario:
    with _LOCK:
        if not force and scenario_id in _CACHE: return _CACHE[scenario_id]
        resolved=resolve_variables(scenario_id)
        if resolved is None: raise ValueError(f"Unknown scenario '{scenario_id}'")
        variables,field_order=resolved
        entity_key=resolve_entity_key(scenario_id)
        by_name={str(v['name']):v for v in variables}
        user_names={str(v['name']) for v in variables if str(v.get('scope','record')).lower()=='user'}
        if entity_key and entity_key in by_name: user_names.add(entity_key)
        # Dependency closure keeps user context self-contained.
        changed=True
        while changed:
            changed=False
            for name in tuple(user_names):
                var=by_name.get(name)
                if not var: continue
                for dep in var.get('depends_on',[]) or []:
                    if dep in by_name and dep not in user_names:
                        user_names.add(dep); changed=True
        user_fields=tuple(v['name'] for v in variables if v['name'] in user_names)
        record_fields=tuple(v['name'] for v in variables if v['name'] not in user_names)
        compiled=CompiledScenario(scenario_id,entity_key,tuple(variables),tuple(field_order),by_name,user_fields,record_fields)
        _CACHE[scenario_id]=compiled
        return compiled

def invalidate_scenario(scenario_id:str)->None:
    with _LOCK: _CACHE.pop(scenario_id,None)
def clear_cache()->None:
    with _LOCK: _CACHE.clear()
