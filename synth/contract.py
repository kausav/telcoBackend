"""The output contract implied by the variable definitions themselves.

Wherever a variable comes from - a person's curated definition, a field of an industry source document, a row of
an uploaded CSV definition - its definition says more than its name: ``choices`` are the only values allowed,
``min``/``max`` bound a number, ``pattern`` and ``min_length``/``max_length`` shape a string, ``nullable: false``
forbids gaps. The generation spec decides *how* values are produced (distributions, causality, a language
model's judgement); this module checks that what it produced still stays inside what the definitions allow, and
reports every breach. Nothing is silently "repaired", and nothing a model wrote can widen the contract: this is
how the industry source documents stay authoritative for what a valid value is.

Deliberately not contractual: ``weights``, ``days_back`` and ``source_examples``. They are sampling hints of the
generic generators, so the spec's own distributions and history window replace them.
"""
from __future__ import annotations

import re
from typing import Any

MAX_EXAMPLES = 3
_NUMERIC = {"integer", "int", "float", "decimal", "number", "numeric"}


def _definition_params(variable: dict[str, Any]) -> dict[str, Any]:
    return variable.get("params") if isinstance(variable.get("params"), dict) else {}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_contract(variables: list[dict[str, Any]], columns: dict[str, str]) -> list[dict[str, Any]]:
    """One entry per delivered column that its definition constrains. ``columns``: spec column id -> column name."""
    by_column = {str(v.get("name")): v for v in variables}
    contract: list[dict[str, Any]] = []
    for cid, column in columns.items():
        v = by_column.get(column)
        if not v:
            continue
        params = _definition_params(v)
        entry: dict[str, Any] = {"column": cid, "name": column}
        if isinstance(params.get("choices"), list) and params["choices"] and str(v.get("gen") or "") != "dependent_choice":
            entry["choices"] = list(params["choices"])
        if str(v.get("dtype") or "").lower() in _NUMERIC:
            for key in ("min", "max"):
                if _number(params.get(key)):
                    entry[key] = params[key]
        if isinstance(params.get("pattern"), str) and params["pattern"]:
            try:
                re.compile(params["pattern"])
                entry["pattern"] = params["pattern"]
            except re.error:
                pass
        for key in ("min_length", "max_length"):
            if _number(params.get(key)):
                entry[key] = int(params[key])
        if v.get("nullable") is False:
            entry["not_null"] = True
        if len(entry) > 2:
            contract.append(entry)
    return contract


def _allowed(value: Any, choices: list[Any]) -> bool:
    if isinstance(value, bool):
        return value in [c for c in choices if isinstance(c, bool)]
    return str(value).casefold() in {str(c).casefold() for c in choices if not isinstance(c, bool)}


def check_contract(rows: list[dict[str, Any]], contract: list[dict[str, Any]],
                   bad_rows: set[int] | None = None) -> list[dict[str, Any]]:
    """Violations of ``contract`` over ``rows`` (``column id -> value`` dicts); ``bad_rows`` collects the offending row indexes."""
    violations: list[dict[str, Any]] = []
    for entry in contract:
        cid = entry["column"]
        pattern = re.compile(entry["pattern"]) if "pattern" in entry else None
        rules = ("not_null", "choices", "min", "max", "pattern", "min_length", "max_length")
        bad: dict[str, list[Any]] = {r: [] for r in rules}
        counts = dict.fromkeys(rules, 0)
        for i, row in enumerate(rows):
            value = row.get(cid)
            problems: list[str] = []
            if value is None:
                if entry.get("not_null"):
                    problems.append("not_null")
            else:
                if "choices" in entry and not _allowed(value, entry["choices"]):
                    problems.append("choices")
                if "min" in entry and _number(value) and value < entry["min"]:
                    problems.append("min")
                if "max" in entry and _number(value) and value > entry["max"]:
                    problems.append("max")
                if isinstance(value, str):
                    if pattern is not None and pattern.fullmatch(value) is None:
                        problems.append("pattern")
                    if "min_length" in entry and len(value) < entry["min_length"]:
                        problems.append("min_length")
                    if "max_length" in entry and len(value) > entry["max_length"]:
                        problems.append("max_length")
            if problems and bad_rows is not None:
                bad_rows.add(i)
            for p in problems:
                counts[p] += 1
                if len(bad[p]) < MAX_EXAMPLES:
                    bad[p].append({"row": i, "value": value.isoformat() if hasattr(value, "isoformat") else value})
        for rule, n in counts.items():
            if n:
                violations.append({"column": entry.get("name", cid), "id": cid, "rule": rule,
                                   "allowed": entry.get(rule), "violations": n, "examples": bad[rule]})
    return violations
