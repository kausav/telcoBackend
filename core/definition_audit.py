"""Audit of curated variable definitions that cannot describe a realistic value on their own."""
from __future__ import annotations

from typing import Any

_RANGE_GENERATORS = {"uniform", "uniform_int"}


def incomplete_definitions(variables: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Curated numeric ranges that declare no upper bound.

    Without a behaviour pack such a column can only be drawn from a neutral open-ended span, so the proposal says
    so and the owner can complete the definition (or author a behaviour pack) for realistic values.
    """
    found = []
    for var in variables:
        params = var.get("params") or {}
        generator = str(var.get("gen") or "").strip().lower()
        if var.get("formula") or generator not in _RANGE_GENERATORS:
            continue
        if params.get("max") is None and params.get("hi") is None:
            found.append({
                "name": str(var.get("name") or ""),
                "reason": "The numeric range declares no upper bound, so its values are drawn from a neutral open-ended span "
                          "(add `max` to the definition or a behaviour pack for realistic values).",
            })
    return found
