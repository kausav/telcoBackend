"""Deterministic generation rules for confirmed agentic scenarios.

This module contains no LLM or scenario-ID-specific logic. It turns the confirmed
variable contract plus request context into machine-readable generation guardrails.
"""
from __future__ import annotations

from typing import Any

from core.scenario_semantics import derive_scenario_semantics, temporal_role
from core.temporal_contract import (
    is_supported_temporal_rule,
    normalize_temporal_family,
    source_declared_max_delay_seconds,
)


_RULE_PARAM_KEYS = (
    "choices", "values", "min", "max", "buckets", "weights", "precision",
    "currency", "target", "mapping", "country", "timezone", "prefix", "digits",
)


def _validated_behavioral_rules(state: Any, variables: list[dict]) -> list[dict[str, Any]]:
    """Return only proposal rules that reference the confirmed executable schema."""
    raw = (getattr(state, "scenario_context", {}) or {}).get("behavioral_rules") or []
    if not isinstance(raw, list):
        return []
    by_name = {str(v.get("name") or "").strip(): v for v in variables if isinstance(v, dict) and v.get("name")}
    allowed = set(by_name)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in raw[:30]:
        if not isinstance(rule, dict):
            continue
        when = rule.get("when") if isinstance(rule.get("when"), dict) else {}
        then = rule.get("then") if isinstance(rule.get("then"), dict) else {}
        if not when or not then or not set(map(str, when)).issubset(allowed) or not set(map(str, then)).issubset(allowed):
            continue
        # Conditions may use a small comparison language; assignments remain scalar or a list
        # of declared values. This lets proposal-time rules express thresholds such as
        # days_to_depletion <= 2 without embedding industry-specific logic in the generator.
        safe_when = {}
        safe_condition = True
        for field, value in when.items():
            if isinstance(value, dict):
                op = str(value.get("op") or "").strip().lower()
                if op not in {"=", "==", "eq", "!=", "ne", "<", "lt", "<=", "lte", ">", "gt", ">=", "gte", "in", "not_in", "notin"}:
                    safe_condition = False
                    break
                operand = value.get("value")
                if isinstance(operand, (dict, tuple, set)) or operand is None:
                    safe_condition = False
                    break
                safe_when[str(field)] = {"op": op, "value": operand}
            elif isinstance(value, (str, int, float, bool)) or isinstance(value, list):
                if isinstance(value, list) and any(isinstance(item, (dict, tuple, set)) for item in value):
                    safe_condition = False
                    break
                safe_when[str(field)] = value
            else:
                safe_condition = False
                break
        if not safe_condition:
            continue
        safe_then = {}
        if any(not (isinstance(v, (str, int, float, bool)) or isinstance(v, list)) for v in then.values()):
            continue
        if any(isinstance(v, list) and any(isinstance(item, (dict, tuple, set)) for item in v) for v in then.values()):
            continue
        assignments_match_contract = True
        for field, value in then.items():
            variable = by_name[str(field)]
            params = variable.get("params") if isinstance(variable.get("params"), dict) else {}
            declared_values = params.get("choices")
            if isinstance(declared_values, list) and declared_values:
                assigned_values = value if isinstance(value, list) else [value]
                declared_norm = {
                    str(item).strip().lower().replace("-", "_").replace(" ", "_")
                    for item in declared_values
                }
                if any(
                    str(item).strip().lower().replace("-", "_").replace(" ", "_") not in declared_norm
                    for item in assigned_values
                ):
                    assignments_match_contract = False
                    break
            safe_then[str(field)] = value
        if not assignments_match_contract:
            continue
        signature = repr((sorted((str(k), repr(v)) for k, v in safe_when.items()), sorted((str(k), repr(v)) for k, v in safe_then.items())))
        if signature in seen:
            continue
        seen.add(signature)
        result.append({"when": safe_when, "then": safe_then})
    return result


def _deterministic_offer_rules(variables: list[dict]) -> list[dict[str, Any]]:
    """Add universal offer lifecycle relationships when the confirmed schema exposes them."""
    names = [str(v.get("name") or "") for v in variables if v.get("name")]

    def find_flag(*tokens: str) -> str | None:
        for name in names:
            low = name.casefold()
            if "flag" not in low or "offer" not in low:
                continue
            if any(token in low for token in tokens):
                return name
        return None

    def find(*needles: str) -> str | None:
        for name in names:
            low = name.casefold()
            if all(token in low for token in needles):
                return name
        return None

    presented = find_flag("presented", "presentation")
    accepted = find_flag("accepted", "acceptance")
    converted = find_flag("converted", "conversion")
    response = next((
        name for name in names
        if "offer" in name.casefold() and "response" in name.casefold()
    ), None)

    rules: list[dict[str, Any]] = []
    if presented and accepted:
        rules.append({"when": {presented: False}, "then": {accepted: False}})
    if accepted and converted:
        rules.append({"when": {accepted: False}, "then": {converted: False}})
    if presented and converted:
        rules.append({"when": {presented: False}, "then": {converted: False}})

    if response and accepted:
        response_var = next((v for v in variables if str(v.get("name") or "") == response), {})
        choices = list((response_var.get("params") or {}).get("choices") or [])
        positive = [
            choice for choice in choices
            if str(choice).casefold().replace("-", "_").replace(" ", "_") in {
                "completed", "complete", "success", "successful", "accepted", "approved",
                "converted", "fulfilled", "settled", "authorized", "done",
            }
        ]
        negative = [
            choice for choice in choices
            if any(token in str(choice).casefold() for token in ("fail", "reject", "declin", "den", "error", "cancel"))
        ]
        if negative:
            rules.append({"when": {response: negative}, "then": {accepted: False}})
        if positive:
            rules.append({"when": {accepted: True}, "then": {response: positive}})

    # De-duplicate without making any assumption about fields that are not present.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in rules:
        sig = repr(rule)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(rule)
    return deduped


def build_deterministic_rules(state: Any, variables: list[dict]) -> dict[str, Any]:
    """Build deterministic rules from the confirmed schema and full scenario context."""
    names = {str(v.get("name")) for v in variables if v.get("name")}
    by_name = {str(v.get("name")): v for v in variables if v.get("name")}
    constraints: dict[str, dict[str, Any]] = {}
    formulas: list[dict[str, str]] = []

    for var in variables:
        name = str(var.get("name") or "")
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        generation_constraint: dict[str, Any] = {}
        for key in _RULE_PARAM_KEYS:
            if key in params:
                generation_constraint["valid_values" if key == "choices" else key] = params[key]
        constraints[name] = generation_constraint
        if var.get("formula"):
            formulas.append({"field": name, "expression": str(var["formula"])})

    temporal: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(before: str, after: str, max_delay: int | None, reason: str) -> None:
        if before not in names or after not in names or before == after:
            return
        before_var = by_name.get(before)
        after_var = by_name.get(after)
        if before_var and after_var and not is_supported_temporal_rule(before_var, after_var):
            return
        key = (before, after)
        if key in seen_edges:
            return
        seen_edges.add(key)
        item: dict[str, Any] = {
            "before": before,
            "after": after,
            "min_delay_seconds": 0,
            "reason": reason,
        }
        if max_delay is not None:
            item["max_delay_seconds"] = int(max_delay)
        temporal.append(item)

    for child_name, child in by_name.items():
        if str(child.get("dtype", "")).lower() != "datetime":
            continue
        for dep in child.get("depends_on", []) or []:
            parent_name = str(dep)
            parent = by_name.get(parent_name)
            if parent and is_supported_temporal_rule(parent, child):
                add_edge(
                    parent_name,
                    child_name,
                    source_declared_max_delay_seconds(parent, child),
                    "Confirmed source-backed datetime dependency implies chronological causality.",
                )

    datetime_vars = [v for v in variables if str(v.get("dtype", "")).lower() == "datetime"]
    # Do not infer chronology merely from generic role words (start/end/request/etc.).
    # That creates false edges across unrelated resources, e.g.
    # ``bucket_valid_for_start`` -> ``topup_valid_for_end``. Only add lifecycle edges when
    # the fields belong to the same semantic family/resource. Explicit depends_on edges and
    # the suffix-based pairs below remain authoritative.
    lifecycle_pairs = {
        ("start", "presentation"),
        ("start", "response"),
        ("start", "completion"),
        ("start", "end"),
        ("dispatch", "presentation"),
        ("dispatch", "response"),
        ("presentation", "response"),
        ("presentation", "completion"),
        ("response", "completion"),
    }
    for parent in datetime_vars:
        for child in datetime_vars:
            if parent is child:
                continue
            parent_role = temporal_role(parent)
            child_role = temporal_role(child)
            same_family = normalize_temporal_family(parent.get("name")) == normalize_temporal_family(child.get("name"))
            if same_family and (parent_role, child_role) in lifecycle_pairs and str(parent.get("scope", "transaction")) == str(child.get("scope", "transaction")):
                add_edge(
                    str(parent.get("name")),
                    str(child.get("name")),
                    source_declared_max_delay_seconds(parent, child),
                    "Same-resource lifecycle semantics imply chronological order.",
                )

    semantics = derive_scenario_semantics(state, variables)
    for field, preferred in semantics.get("preferred_values", {}).items():
        if field in constraints and preferred:
            constraints[field]["preferred_values"] = preferred

    scenario_mode = str(semantics.get("outcome_mode") or "mixed")
    domain = str(getattr(state, "domain", "") or "").strip()
    return {
        "scenario_summary": "Confirmed scenario schema with deterministic scenario-context semantics; no scenario-ID lookup.",
        "domain": domain,
        "industry_type": str(getattr(state, "industry", "") or "").strip(),
        "scenario_type": str(getattr(state, "scenario_type", "") or "").strip(),
        "use_case": str(getattr(state, "use_case", "") or "").strip(),
        "type_of_data": str(getattr(state, "type_of_data", "") or "").strip().lower(),
        "entity_key": str(getattr(state, "entity_key", "") or "").strip() or None,
        "country": str(getattr(state, "country", "") or "").strip().upper() or None,
        "variable_sources": dict(getattr(state, "scenario_context", {}).get("variable_sources") or {}),
        "db_variable_names": sorted(set(getattr(state, "scenario_context", {}).get("db_variable_names") or [])),
        "source_policy": str(getattr(state, "scenario_context", {}).get("source_policy") or "").strip(),
        "agentic": bool(getattr(state, "scenario_context", {}).get("agentic", False)),
        "scenario_mode": scenario_mode,
        "business_rules": [
            "Entity, field and relationship vocabulary is frozen to the approved compiled schema.",
            "Unknown fields, values and relationships are non-executable.",
            "Generate coherent business events first; do not independently randomize causally related fields.",
            "Every source-backed reference field must preserve the semantic type of the referenced resource.",
            "Every paired start/end period must satisfy start <= end.",
            "Every request/confirmation lifecycle must satisfy request <= confirmation when confirmation exists.",
            "Values describing the same transaction, entity, or balance snapshot must be mutually consistent.",
        ],
        "generation_policy": [
            "Generate root/independent fields first and all dependent fields from their actual dependencies.",
            "Never independently sample two fields when the confirmed schema or universal semantics imply a relationship.",
            "Generate temporal pairs and event lifecycles causally; never generate parent and child timestamps independently when a relationship is confirmed.",
            "Treat event occurrence and event ordering separately: if scenario semantics indicate that an event did not occur, its nullable event timestamp must be null and must not be fabricated merely to satisfy a temporal rule.",
            "Only create temporal relationships supported by an explicit datetime dependency or an unambiguous same-resource lifecycle pair; never relate sibling resources merely because they share request/confirmation/start/end terminology.",
            "When a temporal child has multiple causal parents, generate it from the intersection of all applicable lower/upper bounds rather than satisfying only the first parent.",
            "If temporal constraints are mutually incompatible, preserve hard causal ordering and treat optional maximum-delay guidance as soft rather than manufacturing a contradiction.",
            "Apply scenarioType, businessScenario, expectedOutcome, businessResponse, useCase, domain, industry, country, and typeOfData before a record reaches validation.",
            "Use exact confirmed enum/value vocabularies and numeric constraints; never invent source values.",
            "Keep entity-level attributes stable across that entity's transactions and keep transaction/event attributes at transaction grain.",
            "Preserve reference/identifier consistency across related source-backed fields.",
            "A correctly generated record should normally require zero validation repairs; repeated repair/retry is treated as a generator defect.",
        ],
        "domain_invariants": [
            "Stable entity identities must remain unchanged across that entity's generated history.",
            "Fields belonging to one resource lifecycle must describe the same state/transition rather than independent random events.",
            "Validity windows must contain their related event/transaction timeline when the source contract exposes the relationship.",
            "Source-backed fields and persisted business extensions must remain consistent with their selected lifecycle state and declared relationships.",
        ],
        "field_constraints": {
            name: {
                "description": str(var.get("description", "")),
                "nullable": bool(var.get("nullable", False)),
            }
            for name, var in by_name.items()
        },
        "cross_field_rules": [
            "Every declared field dependency must resolve to a field in the confirmed schema.",
            "Causal timestamps must respect declared temporal order and bounded delays.",
            "Scenario outcome/state fields must follow deterministic scenario semantic guardrails.",
        ],
        "conditional_rules": _validated_behavioral_rules(state, variables) + _deterministic_offer_rules(variables),
        "generation_constraints": constraints,
        "formula_rules": formulas,
        "temporal_rules": temporal,
        "scenario_semantics": semantics,
    }
