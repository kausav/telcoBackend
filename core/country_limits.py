"""Operating limits of a country's payment rails: the floors, caps, fees and cash denomination that real systems enforce.

They are data about a country, so they live in MongoDB (collection ``country_operating_limits``) and the application ships only a
starting set in ``seed/country_limits.json`` that it inserts when a country has no document (``core.seed``). Nothing here names a
country or a rail. The limits of the scenario's country are given to the model that designs the generation spec and are measured on
the simulated rows (``synth.compiler.rail_limit_advice``).

A database that cannot be reached falls back to the last limits read for that country, then to the bundled set; it never fails a request.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from config import constants as _c

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_LAST_GOOD: dict[str, dict[str, Any]] = {}
_RAIL_KEYS = ("id", "label", "aliases", "kind", "min_amount", "max_amount", "fee", "note")


def country_code(country: Any) -> str:
    """The country code (``IN``) for a code or a name (``India``); '' when it cannot be told."""
    text = str(country or "").strip()
    if not text:
        return ""
    upper = text.upper()
    from config.country_metadata import COUNTRY_BASE

    if upper in COUNTRY_BASE:
        return upper
    lowered = text.casefold()
    for code, meta in COUNTRY_BASE.items():
        if str(meta.get("country_name") or "").casefold() == lowered:
            return code
    return upper if len(upper) == 2 and upper.isalpha() else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean(document: Any) -> dict[str, Any]:
    """The usable part of a stored document: rails with valid bounds, in a plain JSON shape. {} when nothing usable remains."""
    if not isinstance(document, dict):
        return {}
    rails: list[dict[str, Any]] = []
    for raw in document.get("rails") or []:
        if not isinstance(raw, dict):
            continue
        aliases = [str(a).strip().casefold() for a in (raw.get("aliases") or []) if str(a).strip()]
        if not aliases:
            continue
        rail: dict[str, Any] = {"id": str(raw.get("id") or aliases[0]), "label": str(raw.get("label") or raw.get("id") or aliases[0]), "aliases": aliases}
        for key in ("min_amount", "max_amount", "fee"):
            value = _number(raw.get(key))
            if value is not None:
                rail[key] = value
        if raw.get("kind"):
            rail["kind"] = str(raw["kind"])
        if raw.get("note"):
            rail["note"] = str(raw["note"])[:240]
        if "min_amount" in rail and "max_amount" in rail and rail["min_amount"] > rail["max_amount"]:
            continue
        rails.append(rail)
    out: dict[str, Any] = {}
    denomination = _number(document.get("cash_denomination"))
    if denomination is not None and denomination > 0:
        out["cash_denomination"] = denomination
    if not rails and not out:
        return {}
    out.update(country=str(document.get("country") or ""), currency=str(document.get("currency") or ""), rails=rails)
    if document.get("note"):
        out["note"] = str(document["note"])[:300]
    return out


def for_country(country: Any) -> dict[str, Any]:
    """The cleaned limits of ``country`` ({} when none are stored or bundled)."""
    code = country_code(country)
    if not code:
        return {}
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(code)
    if hit is not None and now - hit[0] < _c.COUNTRY_LIMITS_CACHE_SECONDS:
        return hit[1]
    found: dict[str, Any] | None = None
    reachable = False
    try:
        from core.seed import ensure_seeded
        from models.country_limit import CountryLimitModel

        ensure_seeded()
        found = CountryLimitModel.get(code)
        reachable = True
    except Exception as exc:
        logger.warning("country limits for %s not read from MongoDB: %s: %s", code, type(exc).__name__, exc)
    if reachable:
        result = clean(found)
    else:
        with _LOCK:
            result = _LAST_GOOD.get(code)
        if result is None:
            from core.seed import bundled

            result = next((clean(d) for d in bundled("country_operating_limits") if d.get("country") == code), {})
    with _LOCK:
        _CACHE[code] = (now if reachable else now - max(_c.COUNTRY_LIMITS_CACHE_SECONDS - 5.0, 0.0), result)
        if reachable:
            _LAST_GOOD[code] = result
    return result


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _LAST_GOOD.clear()
