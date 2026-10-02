"""Draw a column the way its own definition says (the ``definition`` sampler).

A variable arrives from one of three places - a persisted scenario variable, a field of an industry source
document, or a row of an uploaded CSV definition - and in every case it has already been reduced to the same
executable contract: ``dtype`` plus a generator (``gen``) and its ``params`` (allowed choices, bounds, id
prefixes, patterns, formulas, ...). This module executes that contract inside the simulator, so a column the
language model has no reason to change keeps behaving exactly as its owner defined it.

Time is the one thing handled here rather than delegated: generators that depend on "now" use the run's
reference time (``as_of``) and return real datetimes, so a seed and an ``asOf`` reproduce a dataset exactly and
the simulator can do arithmetic on them.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

TEMPORAL_GENS = frozenset({"recent_datetime", "recent_date", "ts_offset", "ts_add_field", "date_offset", "date_offset_range"})
# Generators that can only produce a neutral token unless the definition also carries values/examples/patterns.
FREE_TEXT_GENS = frozenset({"semantic_string", "generic", "semantic_event"})
_TIME_DTYPES = {"datetime", "date"}


def _legacy():
    from agents import data_generation_agent as legacy

    return legacy


def has_values(params: dict[str, Any]) -> bool:
    """Whether a free-text definition carries something real to draw from."""
    p = params or {}
    return any(p.get(k) for k in ("choices", "values", "pattern", "source_examples")) or p.get("value") is not None \
        or str(p.get("format") or "").lower() in {"uuid", "uuid4", "email", "idn-email", "uri", "uri-reference", "url", "ipv4", "ipv6"}


def is_placeholder_only(variable: dict[str, Any]) -> bool:
    """True when the definition can only yield a neutral placeholder token (``SYN_...``)."""
    gen = str(variable.get("gen") or "")
    dtype = str(variable.get("dtype") or "").lower()
    if variable.get("formula") or dtype in _TIME_DTYPES or dtype in {"int", "integer", "float", "decimal", "number", "numeric", "boolean"}:
        return False
    if gen in FREE_TEXT_GENS:
        name = str(variable.get("name") or "").lower()
        description = str(variable.get("description") or "").lower()
        params = variable.get("params") or {}
        if has_values(params):
            return False
        # Identifier / phone / address-like columns get structured values from the legacy name semantics.
        if name.endswith("_id") or name == "id" or name.endswith("_key") or "phone" in name or "msisdn" in name \
                or "email" in name or "href" in name or "url" in name or "identifier" in description:
            return False
        if re.search(r"\b(such as|for example|possible values|valid values)\b", description):
            return False
        return True
    return False


def dependencies(variable: dict[str, Any]) -> set[str]:
    """Original column names the definition reads (depends_on, formula names, base/source/lo/hi fields)."""
    legacy = _legacy()
    deps = {str(d) for d in (variable.get("depends_on") or []) if d}
    params = variable.get("params") or {}
    for key in ("base_field", "source_field", "depends_on_field", "add_seconds_field", "lo_field", "hi_field", "field", "segment_field"):
        if isinstance(params.get(key), str) and params[key]:
            deps.add(params[key])
    if variable.get("formula"):
        deps |= legacy._formula_dependencies(str(variable["formula"]))
    deps.discard(str(variable.get("name") or ""))
    return deps


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    parsed = _legacy()._qa_parse_dt(value)
    return parsed


def _num(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _temporal(gen: str, params: dict[str, Any], rec: dict[str, Any], rt: Any) -> datetime | None:
    rng, as_of = rt.ctx.rng, rt.ctx.as_of
    if gen in {"recent_datetime", "recent_date"}:
        back = max(0, _num(params.get("days_back"), 0))
        stamp = as_of - timedelta(days=rng.randint(0, back), hours=rng.randint(0, 23), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
        return min(stamp, as_of).replace(microsecond=0)
    base_field = str(params.get("base_field") or params.get("source_field") or "").strip()
    base = _as_dt(rec.get(base_field)) if base_field else None
    if base is None:
        return None
    if gen == "ts_offset":
        lo = _num(params.get("min_sec", params.get("min_seconds")), 0)
        hi = _num(params.get("max_sec", params.get("max_seconds")), lo)
        return base + timedelta(seconds=rng.randint(min(lo, hi), max(lo, hi)))
    if gen == "ts_add_field":
        return base + timedelta(seconds=_num(rec.get(str(params.get("add_seconds_field"))), 60))
    if gen == "date_offset":
        return base + timedelta(days=_num(params.get("days"), 0))
    lo, hi = _num(params.get("min_days"), 0), _num(params.get("max_days"), _num(params.get("min_days"), 0))
    return base + timedelta(days=rng.randint(min(lo, hi), max(lo, hi)))


def draw(spec: dict[str, Any], env: dict[str, Any], rt: Any) -> Any:
    """One value for the definition ``spec`` (``gen``, ``params``, ``dtype``, ``formula``, ``name``, ``description``)."""
    legacy = _legacy()
    gen = str(spec.get("gen") or "generic")
    params = dict(spec.get("params") or {})
    dtype = str(spec.get("dtype") or "string").lower()
    names = (rt.reference or {}).get("columns") or {}
    rec: dict[str, Any] = {original: env[alias] for alias, original in names.items() if alias in env}
    rec["__country__"] = (rt.reference or {}).get("country")
    rec["__current_field__"] = spec.get("name")
    if gen in TEMPORAL_GENS:
        return _temporal(gen, params, rec, rt)
    if gen == "generic" and dtype in _TIME_DTYPES and not spec.get("formula"):
        return _temporal("recent_datetime", params, rec, rt)
    var = {"name": spec.get("name"), "dtype": dtype, "params": params, "formula": spec.get("formula"),
           "description": spec.get("description", ""), "gen": gen}
    token = legacy._GENERATION_RNG.set(rt.ctx.rng)          # the run's own seeded stream, so a seed reproduces the data
    try:
        if spec.get("formula") or gen == "formula":
            value = legacy._formula(var, rec)
        else:
            generator = legacy._GENERATORS.get(gen) or legacy._GENERATORS["generic"]
            value = generator(var, rec)
    finally:
        legacy._GENERATION_RNG.reset(token)
    return _typed(value, dtype)


def _typed(value: Any, dtype: str) -> Any:
    if value is None or isinstance(value, datetime):
        return value
    if dtype in _TIME_DTYPES:
        return _as_dt(value)
    if dtype in {"int", "integer"}:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None
    if dtype in {"float", "decimal", "number", "numeric"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if dtype == "boolean":
        flag = _legacy()._boolean_semantic(value)
        return flag
    return value if isinstance(value, str) else str(value)
