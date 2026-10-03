"""The baseline spec: what can be said about a scenario from its variable definitions alone.

Every variable - persisted scenario variable, industry source document field or uploaded CSV row - carries an
executable definition (dtype, generator, params, dependencies). The baseline turns a list of them into a runnable
``GenerationSpec`` in which every column is drawn the way its own definition says, in dependency order, with:

* entity-level columns drawn once per entity and event-level columns once per row;
* one column acting as the event clock (when the definitions name one);
* timestamps that the definitions relate to each other (explicit dependency, or a same-resource lifecycle
  pair) drawn at or after their parents.

It deliberately knows nothing about the scenario's *behaviour* - which outcomes happen, what is conditional,
what scenario type changes - because that is not in a definition. ``synth.compiler`` asks a language model for
exactly that and layers it over this baseline; this is also the spec a scenario is generated from when no model
is reachable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from synth import definitions
from synth.expr import Expr
from synth.samplers import is_rollup, sampler_names
from synth.spec import GenerationSpec, SpecColumn, safe_identifier

_DTYPE = {
    "int": "integer", "integer": "integer", "long": "integer", "bigint": "integer",
    "float": "float", "double": "float", "decimal": "float", "number": "float", "numeric": "float",
    "bool": "boolean", "boolean": "boolean",
    "datetime": "datetime", "timestamp": "datetime", "date": "date",
    "categorical": "categorical", "enum": "categorical",
}
UNSUPPORTED_DTYPES = frozenset({"object", "array"})
_UNIQUE_GENS = frozenset({"prefixed_int", "prefixed_uuid", "uuid_string", "tx_id"})
_FREE_TIME_GENS = frozenset({"recent_datetime", "recent_date", "generic", "timestamp", "semantic_string"})
DEFAULT_HISTORY_DAYS = 180
DEFAULT_MIN_GAP_MINUTES = 60


class SpecBuildError(ValueError):
    pass


def spec_dtype(dtype: Any) -> str:
    return _DTYPE.get(str(dtype or "").strip().lower(), "string")


def order_entries(entries: list[dict[str, Any]], *, soft: bool = False) -> list[dict[str, Any]]:
    """Order emit entries so every entry comes after what it reads: entity-scope ones first, then event-scope ones, then rollups.

    ``entries`` are dicts with ``column``, ``scope``, ``when``, ``expr``, ``sample``. Entries are sorted by
    dependency and, among independent ones, kept in their given order, so the result is stable. An entity-scope
    entry may not read an event-scope column, and a cycle is an error - unless ``soft``, where the edge that closes
    a cycle is dropped (its ``reads`` entry too) and the entry simply sees no value for that column.

    A rollup (an entity fact computed from the entity's own events) is the one entity-scope entry that reads event
    columns. It is computed after the events, so nothing but another rollup may read it.
    """
    by_col = {e["column"]: e for e in entries}
    position = {e["column"]: i for i, e in enumerate(entries)}
    rollups = {c for c, e in by_col.items() if is_rollup(e.get("sample"))}

    def reads(e: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for src in (e.get("when"), e.get("expr")):
            if src:
                names |= Expr(src).names
        if e.get("sample") is not None:
            names |= sampler_names(e["sample"])
        return {n for n in names if n in by_col and n != e["column"]}

    deps = {c: reads(e) for c, e in by_col.items()}
    for c, e in by_col.items():
        if c not in rollups:
            summary = sorted(d for d in deps[c] if d in rollups)
            if summary and not soft:
                raise SpecBuildError(f"column '{c}' reads rollup column(s) {summary}; a rollup summarises the entity's events and is "
                                     "computed after them, so only another rollup may read it")
            deps[c] -= set(summary)
        if e["scope"] == "entity" and c not in rollups:
            bad = sorted(d for d in deps[c] if by_col[d]["scope"] == "event")
            if bad:
                if not soft:
                    raise SpecBuildError(f"entity-scope column '{c}' reads event-scope column(s) {bad}")
                deps[c] -= set(bad)
                _drop_reads(e, set(bad))
    ordered: list[str] = []
    placed: set[str] = set()

    def rank(c: str) -> int:
        return 2 if c in rollups else (0 if by_col[c]["scope"] == "entity" else 1)

    remaining = sorted(by_col, key=lambda c: (rank(c), position[c]))
    while remaining:
        progress = False
        for c in list(remaining):
            if deps[c] <= placed:
                ordered.append(c)
                placed.add(c)
                remaining.remove(c)
                progress = True
                break
        if not progress:
            if not soft:
                raise SpecBuildError("columns depend on each other in a cycle: " + ", ".join(sorted(remaining)[:8]))
            c = remaining[0]
            cut = deps[c] - placed
            deps[c] -= cut
            _drop_reads(by_col[c], cut)
    # entity-scope entries first, event-scope next, rollups last (a stable sort keeps dependency order inside each group)
    ordered.sort(key=rank)
    return [by_col[c] for c in ordered]


def _drop_reads(entry: dict[str, Any], names: set[str]) -> None:
    sample = entry.get("sample")
    if isinstance(sample, dict) and sample.get("type") == "definition":
        sample["reads"] = [r for r in sample.get("reads", []) if r not in names]


def _entity_key(variables: list[dict[str, Any]], entity_key: str | None) -> str | None:
    names = [str(v.get("name")) for v in variables if v.get("name")]
    if entity_key:
        match = next((n for n in names if n.lower() == str(entity_key).strip().lower()), None)
        if match:
            return match
    return None


@dataclass
class Baseline:
    """A baseline spec plus what the compiler needs to override parts of it without losing the definitions."""
    spec: GenerationSpec
    definition_entries: dict[str, dict[str, Any]]    # column id -> emit that draws it from its own definition
    variables: dict[str, dict[str, Any]]             # column id -> the variable definition
    clock_candidate: str | None


def build_baseline(variables: list[dict[str, Any]], *, brief: dict[str, Any]) -> GenerationSpec:
    return build(variables, brief=brief).spec


def build(variables: list[dict[str, Any]], *, brief: dict[str, Any]) -> Baseline:
    """A runnable spec from definitions alone. ``brief``: scenario facts (``type_of_data``, ``entity_key``, ...)."""
    from agents.data_generation_agent import _infer_temporal_relationships, _pick_timestamp_field
    from core.compiled_schema import infer_history_field_sets

    usable = [dict(v) for v in variables
              if isinstance(v, dict) and v.get("name") and str(v.get("dtype") or "").lower() not in UNSUPPORTED_DTYPES]
    if not usable:
        raise SpecBuildError("no usable variables")
    transactional = str(brief.get("type_of_data") or "transactional").lower() == "transactional"
    entity_name = _entity_key(usable, brief.get("entity_key")) if transactional else None

    taken: set[str] = set()
    alias: dict[str, str] = {}
    for v in usable:
        alias[str(v["name"])] = safe_identifier(str(v["name"]), taken)
        taken.add(alias[str(v["name"])])
    by_name = {str(v["name"]): v for v in usable}

    # ---- scope -------------------------------------------------------------------------------------------------
    if transactional:
        user_fields, _ = infer_history_field_sets(usable, entity_name)
        entity_scope = set(user_fields)
        if entity_name:
            entity_scope.add(entity_name)
    else:
        entity_scope = set()
    deps = {name: definitions.dependencies(v) & set(by_name) for name, v in by_name.items()}
    changed = True
    while changed:                                  # an entity-level column cannot read an event-level one
        changed = False
        for name in list(entity_scope):
            if name != entity_name and deps[name] - entity_scope:
                entity_scope.discard(name)
                changed = True

    # ---- clock and temporal relations --------------------------------------------------------------------------
    clock = None
    if transactional:
        pick = _pick_timestamp_field(usable)
        candidate = by_name.get(pick) if pick else None
        if candidate and spec_dtype(candidate.get("dtype")) == "datetime" and pick not in entity_scope \
                and not candidate.get("formula") and str(candidate.get("gen") or "") in _FREE_TIME_GENS | {"constant"}:
            clock = pick
    relations: dict[str, list[tuple[str, int | None, int]]] = defaultdict(list)
    try:
        for parent, child, max_gap, min_gap in _infer_temporal_relationships(usable):
            relations[child].append((parent, max_gap, min_gap))
    except Exception:       # a relation that cannot be inferred is simply not enforced in the baseline
        relations.clear()

    # ---- columns + emits ---------------------------------------------------------------------------------------
    columns: dict[str, SpecColumn] = {}
    entries: list[dict[str, Any]] = []
    definition_entries: dict[str, dict[str, Any]] = {}
    for v in usable:
        name = str(v["name"])
        cid = alias[name]
        dtype = spec_dtype(v.get("dtype"))
        params = v.get("params") if isinstance(v.get("params"), dict) else {}
        scope = "entity" if name in entity_scope else "event"
        fmt = params.get("timestamp_format") or params.get("format") if dtype in {"datetime", "date"} else None
        precision = params.get("precision") if dtype == "float" and isinstance(params.get("precision"), int) else None
        columns[cid] = SpecColumn(
            kind=scope, dtype=dtype, column=name, description=str(v.get("description") or "")[:300], precision=precision,
            placeholder=definitions.is_placeholder_only(v), timestamp_format=str(fmt) if fmt else None)
        gen = str(v.get("gen") or "generic")
        sample_def = {"type": "definition", "gen": gen, "params": params, "dtype": str(v.get("dtype") or "string"),
                      "formula": v.get("formula") or None, "name": name, "description": str(v.get("description") or ""),
                      "reads": sorted(alias[d] for d in deps[name])}
        if gen in _UNIQUE_GENS or name == entity_name:
            sample_def["unique"], sample_def["registry"] = True, cid
        definition_entries[cid] = {"column": cid, "scope": scope, "when": None, "expr": None, "sample": sample_def}
        if name == clock:
            entries.append({"column": cid, "scope": scope, "when": None, "expr": "event_at", "sample": None})
            continue
        parents = [(p, mx, mn) for p, mx, mn in relations.get(name, []) if p in alias and p != name]
        if parents and dtype == "datetime" and not v.get("formula") and gen in _FREE_TIME_GENS:
            sample = {"type": "after_parents", "parents": [{"var": alias[p], "min_secs": mn, "max_secs": mx} for p, mx, mn in parents]}
            entries.append({"column": cid, "scope": scope, "when": None, "expr": None, "sample": sample})
            continue
        entries.append(dict(definition_entries[cid]))

    ordered = order_entries(entries, soft=True)
    spec = GenerationSpec(
        source="definitions",
        scenario={k: brief.get(k) for k in ("industry", "domain", "country", "use_case", "scenario_type", "type_of_data") if brief.get(k)},
        entity_column=alias[entity_name] if entity_name else None,
        clock=alias[clock] if clock else None,
        columns=columns,
        emit=[{"column": e["column"], "scope": e["scope"], "when": e["when"], "expr": e["expr"], "sample": e["sample"]} for e in ordered],
        timeline={"history_days": DEFAULT_HISTORY_DAYS, "min_gap_minutes": DEFAULT_MIN_GAP_MINUTES},
        assumptions=["Behaviour is not modelled: every column is drawn from its own definition, and only timestamps the "
                     "definitions relate to each other are ordered."],
        output={"order_by": alias[clock]} if clock else {},
    )
    return Baseline(spec, definition_entries, {alias[str(v["name"])]: v for v in usable}, alias[clock] if clock else None)
