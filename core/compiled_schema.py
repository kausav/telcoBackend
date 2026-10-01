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
        # A behaviour pack states for every column whether it belongs to the entity or to each event; that
        # declaration is final, so the naming heuristics below only apply to columns no pack governs.
        governed = {name for name, var in by_name.items() if var.get("concept")}
        # Identity fields can arrive from persisted DB definitions with a legacy transaction scope.
        # Their business identity is still entity-stable, so keep the value across a user's history.
        identity_tokens = {"customer", "account", "user", "member", "patient", "policyholder"}
        for name, var in by_name.items():
            if name in governed:
                continue
            low = str(name).casefold()
            desc = str(var.get("description") or "").casefold()
            looks_like_identity = (
                any(low.endswith(f"_{token}_id") or low == f"{token}_id" for token in identity_tokens)
                or (low.endswith("_id") and any(token in low for token in identity_tokens) and "payment" not in low and "transaction" not in low and "order" not in low)
                or ("unique identifier" in desc and any(token in low for token in identity_tokens))
            )
            if looks_like_identity:
                stable.add(name)

        # Stateful resource identifiers (for example a balance bucket, entitlement, inventory
        # balance, or subscription resource) are often declared transaction-scoped in source
        # schemas even though the same resource persists across an entity's history. Promote an ID
        # to stable context only when its owner also exposes lifecycle/state-like fields such as
        # remaining quantity, validity, balance, status, or state. Transaction IDs expose request/
        # amount/status but do not satisfy this resource-state fingerprint.
        for name, var in by_name.items():
            low = str(name).casefold()
            if not low.endswith("_id") or name in stable or name in governed:
                continue
            owner = low[:-3].rstrip("_")
            if not owner:
                continue
            stateful_sibling = False
            for sibling in names:
                if sibling == name:
                    continue
                sibling_low = str(sibling).casefold()
                if not sibling_low.startswith(owner + "_"):
                    continue
                suffix = sibling_low[len(owner) + 1:]
                if any(token in suffix for token in ("remaining", "reserved", "valid_for", "validity", "entitlement", "capacity")):
                    stateful_sibling = True
            if stateful_sibling:
                stable.add(name)
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

def compile_scenario(requested_scenario_id: str, force: bool=False)->CompiledScenario:
    with _LOCK:
        if not force and requested_scenario_id in _CACHE: return _CACHE[requested_scenario_id]
        resolved=resolve_variables(requested_scenario_id)
        if resolved is None: raise ValueError(f"Unknown requested_scenario_id '{requested_scenario_id}'")
        variables,field_order=resolved
        entity_key=resolve_entity_key(requested_scenario_id)
        by_name={str(v['name']):v for v in variables}
        user_fields,record_fields=infer_history_field_sets(variables, entity_key)
        compiled=CompiledScenario(requested_scenario_id,entity_key,tuple(variables),tuple(field_order),by_name,user_fields,record_fields)
        _CACHE[requested_scenario_id]=compiled
        return compiled

def invalidate_scenario(requested_scenario_id:str)->None:
    with _LOCK: _CACHE.pop(requested_scenario_id,None)
