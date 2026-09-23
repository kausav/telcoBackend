"""Deterministic generation rules for confirmed agentic scenarios.

This module contains no LLM or scenario-ID-specific logic. It turns the confirmed
variable contract plus request context into machine-readable generation guardrails.
"""
from __future__ import annotations

from typing import Any
import re

from core.scenario_semantics import derive_scenario_semantics, temporal_delay_limit_seconds, temporal_role


_RULE_PARAM_KEYS = (
    "choices", "values", "min", "max", "buckets", "weights", "precision",
    "currency", "target", "mapping", "country", "timezone", "prefix", "digits",
)


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
            if parent and str(parent.get("dtype", "")).lower() == "datetime":
                add_edge(
                    parent_name,
                    child_name,
                    temporal_delay_limit_seconds(child, parent),
                    "Confirmed datetime dependency implies chronological causality.",
                )

    datetime_vars = [v for v in variables if str(v.get("dtype", "")).lower() == "datetime"]
    lifecycle_pairs = {
        ("presentation", "response"),
        ("dispatch", "response"),
        ("start", "completion"),
        ("start", "end"),
        ("presentation", "completion"),
    }
    for parent in datetime_vars:
        for child in datetime_vars:
            if parent is child:
                continue
            parent_role = temporal_role(parent)
            child_role = temporal_role(child)
            if (parent_role, child_role) in lifecycle_pairs and str(parent.get("scope", "transaction")) == str(child.get("scope", "transaction")):
                add_edge(
                    str(parent.get("name")),
                    str(child.get("name")),
                    temporal_delay_limit_seconds(child, parent),
                    "Scenario lifecycle semantics imply chronological order.",
                )

    # Low Balance & Top-up uses short operational lifecycles. The supplied TMF654
    # Swagger defines request/confirmation timestamps but does not define a synthetic
    # latency window, so keep this domain's generated event gaps bounded to one day.
    domain_text = str(getattr(state, "domain", "") or "").strip().lower()
    if "low balance" in domain_text and ("top up" in domain_text or "top-up" in domain_text or "recharge" in domain_text):
        compact = lambda value: re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
        datetime_names = [str(v.get("name")) for v in variables if str(v.get("dtype", "")).lower() == "datetime" and v.get("name")]
        requested_fields = [n for n in datetime_names if "request" in compact(n)]
        confirmation_fields = [n for n in datetime_names if "confirm" in compact(n) or "completion" in compact(n)]
        for parent in requested_fields:
            for child in confirmation_fields:
                add_edge(parent, child, 24 * 3600, "Low Balance top-up lifecycle is bounded to a realistic operational window.")

        # Override broader generic lifecycle ceilings for intervention response timing.
        # This specifically prevents presentation -> decision examples from drifting into
        # multi-week/month gaps while leaving unrelated timestamps unconstrained.
        for item in temporal:
            parent = compact(item.get("before"))
            child = compact(item.get("after"))
            if (
                ("present" in parent or "offer" in parent or "dispatch" in parent)
                and ("decision" in child or "declin" in child or "response" in child or "accept" in child)
            ):
                item["max_delay_seconds"] = min(int(item.get("max_delay_seconds", 24 * 3600) or 24 * 3600), 24 * 3600)

    semantics = derive_scenario_semantics(state, variables)
    for field, preferred in semantics.get("preferred_values", {}).items():
        if field in constraints and preferred:
            constraints[field]["preferred_values"] = preferred

    scenario_mode = str(semantics.get("outcome_mode") or "mixed")
    domain = str(getattr(state, "domain", "") or "").strip()
    return {
        "scenario_summary": "Confirmed scenario schema with deterministic scenario-context semantics; no scenario-ID lookup.",
        "domain": domain,
        "scenario_type": str(getattr(state, "scenario_type", "") or "").strip(),
        "use_case": str(getattr(state, "use_case", "") or "").strip(),
        "type_of_data": str(getattr(state, "type_of_data", "") or "").strip().lower(),
        "entity_key": str(getattr(state, "entity_key", "") or "").strip() or None,
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
        "domain_invariants": [
            "Low Balance & Top-up records must keep subscriber/account/msisdn stable across a subscriber history.",
            "Low Balance & Top-up balances, usage types, units, top-up amounts, statuses, and timestamps must describe the same recharge lifecycle.",
            "Low Balance & Top-up validity windows must be derived from the recharge/plan timeline rather than independently sampled.",
            "Scenario-specific outcome fields may extend the official source model, but they must remain consistent with official status and transaction state fields.",
        ] if "low balance" in domain.lower() else [],
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
        "conditional_rules": [],
        "generation_constraints": constraints,
        "formula_rules": formulas,
        "temporal_rules": temporal,
        "scenario_semantics": semantics,
    }
