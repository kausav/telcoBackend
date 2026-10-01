"""The output contract implied by the variable definitions a person curated in the DB.

A curated variable definition says more than its name: ``choices`` are the only values allowed,
``min``/``max`` bound a number, ``nullable: false`` forbids gaps. The behaviour pack decides *how*
values are produced (distributions, causality); this module checks that what it produced still stays
inside what the definitions allow, and reports every breach. Nothing is silently "repaired".

Deliberately not contractual: ``weights`` and ``days_back``. They are sampling hints of the generic
generators (equal weights and a fixed ``days_back`` are what an unconfigured variable carries), so the
pack's own distributions and history window replace them.
"""
from __future__ import annotations

from typing import Any

from synth.concepts import CURATED

MAX_EXAMPLES = 3
_NUMERIC = {"integer", "int", "float", "decimal", "number"}


def _definition_params(variable: dict[str, Any]) -> dict[str, Any]:
    params = variable.get("params") if isinstance(variable.get("params"), dict) else {}
    if str(variable.get("gen") or "") == "behavior_pack":
        prov = variable.get("provenance") if isinstance(variable.get("provenance"), dict) else {}
        legacy = prov.get("legacy_params")
        params = legacy if isinstance(legacy, dict) else {}
    return params


def build_contract(variables: list[dict[str, Any]], concept_columns: dict[str, str]) -> list[dict[str, Any]]:
    """One entry per curated column that constrains its values."""
    by_column = {str(v.get("name")): v for v in variables}
    contract: list[dict[str, Any]] = []
    for concept, column in concept_columns.items():
        v = by_column.get(column)
        if not v or str(v.get("source") or "").upper() not in CURATED:
            continue
        params = _definition_params(v)
        entry: dict[str, Any] = {"concept": concept, "column": column}
        if isinstance(params.get("choices"), list) and params["choices"]:
            entry["choices"] = list(params["choices"])
        if str(v.get("dtype") or "").lower() in _NUMERIC:
            for key in ("min", "max"):
                if isinstance(params.get(key), (int, float)) and not isinstance(params.get(key), bool):
                    entry[key] = params[key]
        if v.get("nullable") is False:
            entry["not_null"] = True
        if len(entry) > 2:
            contract.append(entry)
    return contract


def _allowed(value: Any, choices: list[Any]) -> bool:
    if isinstance(value, bool):
        return value in [c for c in choices if isinstance(c, bool)]
    return str(value).casefold() in {str(c).casefold() for c in choices if not isinstance(c, bool)}


def check_contract(rows: list[dict[str, Any]], contract: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Violations of ``contract`` over ``rows`` (``concept -> value`` dicts)."""
    violations: list[dict[str, Any]] = []
    for entry in contract:
        concept = entry["concept"]
        bad: dict[str, list[Any]] = {"not_null": [], "choices": [], "min": [], "max": []}
        counts = dict.fromkeys(bad, 0)
        for i, row in enumerate(rows):
            value = row.get(concept)
            problems: list[str] = []
            if value is None:
                if entry.get("not_null"):
                    problems.append("not_null")
            else:
                if "choices" in entry and not _allowed(value, entry["choices"]):
                    problems.append("choices")
                if "min" in entry and isinstance(value, (int, float)) and value < entry["min"]:
                    problems.append("min")
                if "max" in entry and isinstance(value, (int, float)) and value > entry["max"]:
                    problems.append("max")
            for p in problems:
                counts[p] += 1
                if len(bad[p]) < MAX_EXAMPLES:
                    bad[p].append({"row": i, "value": value.isoformat() if hasattr(value, "isoformat") else value})
        for rule, n in counts.items():
            if n:
                violations.append({"column": entry["column"], "concept": concept, "rule": rule,
                                   "allowed": entry.get(rule if rule != "not_null" else "not_null"),
                                   "violations": n, "examples": bad[rule]})
    return violations
