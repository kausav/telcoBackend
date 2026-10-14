"""Declarative value samplers for generation specs.

Any sampler may carry ``"cast": "int" | "float"`` (``int`` rounds to the nearest integer).

A spec says *how a column is produced* in data, never in code. Each ``emit`` entry is either an
expression over engine variables / earlier columns, or a sampler from the small, closed catalogue below.
This module holds mechanism only: there are no vocabularies, prefixes, numbers or formats in it.

    {"type": "const",     "value": ...}
    {"type": "expr",      "expr": "<expression>"}
    {"type": "choice",    "weights": {label: w}}                      # or "from_ref": "<REF key>"
                          | "by": "<expression>", "tables": {key: {label: w} | null}   # or "tables_ref"
                          | optional "filter": "<expression over _choice>" (per-table: {"weights":..,"filter":..})
                          (labels are JSON keys, i.e. strings: use "cast" to get numbers)
    {"type": "id",        "prefix": "", "digits": 8, "registry": "<name>",             # unique per run
                          "stable_by": "<expression>", "format_hint": true,            # one id per entity and key
                          "unique": false}                                              # ids may repeat across rows
    {"type": "pool",      "name": "<pool>", "prefix": "", "digits": 8,                  # ids shared by many entities
                          "ratio": 0.3, "min_size": 3, "skew": 0.6}                    # members = entities x ratio; skew > 0: a few hubs
    {"type": "template",  "parts": [{"const": ..} | {"choice": [..]} | {"choice_ref": ".."} | {"digits": n}],
                          "registry": "<name>"}
    {"type": "lognormal", "median": m, "sigma": s, "min": a, "max": b}
    {"type": "beta_scaled", "alpha": a, "beta": b, "scale": "<expr>", "round": "<expr>"}
    {"type": "uniform_int", "low": a, "high": b, "p": prob}                   # a, b: number or "<expression>"
    {"type": "uniform",   "low": a, "high": b, "round": digits}               # a, b: number or "<expression>"
    {"type": "bernoulli", "p": prob}                                          # prob: number or "<expression>"
    {"type": "time_shift", "anchor": "<variable>", "direction": "before|after",
                           "days": {lognormal spec} | "minutes": {lognormal spec} |
                           "seconds": [lo, hi], "extra_seconds": [lo, hi]}    # bounds may be expressions
    {"type": "switch",    "by": "<expression>", "cases": {key: <sampler>}}
    {"type": "after_parents", "parents": [{"var": "<variable>", "min_secs": 0, "max_secs": null}]}   # at/after every parent
    {"type": "definition", "gen": "<generator>", "params": {...}, "dtype": "..", "formula": "..", "name": "<column>",
                           "reads": ["<variable>", ..]}                      # the column's own definition (see synth.definitions)
    {"type": "rollup",    "fn": "first|last|min|max|sum|mean|count|any|all|distinct", "of": "<expression per row>",
                          "where": "<expression per row>"}                  # a fact about the ENTITY computed from its own events
"""
from __future__ import annotations

import math
import random
from datetime import timedelta
from typing import Any

from synth.expr import Expr

TYPES = {"const", "expr", "choice", "id", "pool", "template", "lognormal", "beta_scaled", "uniform_int", "uniform", "bernoulli",
         "time_shift", "switch", "definition", "after_parents", "rollup"}
ROLLUP_FNS = ("first", "last", "min", "max", "sum", "mean", "count", "any", "all", "distinct")
_EXPR_KEYS = {"expr", "by", "filter", "scale", "round", "stable_by", "of", "where"}


def is_rollup(sample: Any) -> bool:
    """True for a sampler that summarises the entity's own events (computed after them, never drawn)."""
    return isinstance(sample, dict) and sample.get("type") == "rollup"
_CASTS = {"int": lambda v: int(round(float(v))), "float": float}
_NUMERIC_KEYS = {"uniform_int": ("low", "high"), "uniform": ("low", "high", "round"), "bernoulli": ("p",), "pool": ("ratio", "min_size", "skew")}


def sampler_names(spec: dict[str, Any]) -> set[str]:
    """Free variables a sampler reads (what the engine must have defined before it runs)."""
    names: set[str] = set()
    kind = spec.get("type")
    for key in _EXPR_KEYS | set(_NUMERIC_KEYS.get(kind, ())):
        if isinstance(spec.get(key), str):
            names |= Expr(spec[key]).names
    if kind == "definition":
        names |= {str(n) for n in spec.get("reads") or ()}
    if kind == "after_parents":
        names |= {str(p["var"]) for p in spec.get("parents") or ()}
    if kind == "time_shift":
        names.add(spec["anchor"])
        for key in ("seconds", "extra_seconds"):
            for bound in spec.get(key) or ():
                if isinstance(bound, str):
                    names |= Expr(bound).names
    for table in (spec.get("tables") or {}).values():
        if isinstance(table, dict) and isinstance(table.get("filter"), str):
            names |= Expr(table["filter"]).names
    for case in (spec.get("cases") or {}).values():
        names |= sampler_names(case)
    return names - {"_choice"}


def case_for(table: dict[str, Any], value: Any) -> Any:
    """The entry of ``table`` that belongs to ``value``, or None.

    Keys are the text of the value. A true/false value is written ``True``/``False`` by the engine but ``true``/``false`` in JSON,
    so either spelling selects the case; without that, a switch on a flag silently matched nothing and its column came out empty.
    """
    if not isinstance(table, dict):
        return None
    hit = table.get(str(value))
    if hit is None and isinstance(value, bool):
        hit = table.get(str(value).lower())
    return hit


class SamplerError(ValueError):
    pass


def validate(spec: Any, where: str) -> None:
    """Fail fast (when a spec is loaded) on unknown types, bad expressions and missing keys."""
    if not isinstance(spec, dict) or spec.get("type") not in TYPES:
        raise SamplerError(f"{where}: sampler needs a 'type' in {sorted(TYPES)}")
    kind = spec["type"]
    need = {
        "const": ["value"], "expr": ["expr"], "id": ["digits"], "pool": ["name", "digits"], "template": ["parts"], "lognormal": ["median", "sigma"],
        "beta_scaled": ["alpha", "beta", "scale"], "uniform_int": ["low", "high"], "uniform": ["low", "high"],
        "bernoulli": ["p"], "time_shift": ["anchor", "direction"],
        "switch": ["by", "cases"], "choice": [], "definition": ["gen"], "after_parents": ["parents"], "rollup": ["fn"],
    }[kind]
    for key in need:
        if key not in spec:
            raise SamplerError(f"{where}: '{kind}' sampler needs '{key}'")
    for key in _EXPR_KEYS & spec.keys():
        if isinstance(spec[key], str):
            Expr(spec[key])
    for key in _NUMERIC_KEYS.get(kind, ()):
        if isinstance(spec.get(key), str):
            Expr(spec[key])
    if kind == "rollup":
        if spec["fn"] not in ROLLUP_FNS:
            raise SamplerError(f"{where}: rollup fn must be one of {list(ROLLUP_FNS)}")
        if spec["fn"] != "count" and not spec.get("of"):
            raise SamplerError(f"{where}: rollup '{spec['fn']}' needs 'of' (an expression evaluated on each event)")
    if spec.get("cast") is not None and spec["cast"] not in _CASTS:
        raise SamplerError(f"{where}: cast must be one of {sorted(_CASTS)}")
    if kind == "choice":
        if not any(k in spec for k in ("weights", "from_ref", "tables", "tables_ref")):
            raise SamplerError(f"{where}: 'choice' needs weights, from_ref, tables or tables_ref")
        if ("tables" in spec or "tables_ref" in spec) and "by" not in spec:
            raise SamplerError(f"{where}: table choices need 'by'")
        weight_tables = [spec["weights"]] if isinstance(spec.get("weights"), dict) else []
        for table in (spec.get("tables") or {}).values():
            if isinstance(table, dict) and isinstance(table.get("filter"), str):
                Expr(table["filter"])
            if isinstance(table, dict):
                weight_tables.append(table.get("weights") if isinstance(table.get("weights"), dict) else table)
        for table in weight_tables:
            for label, weight in table.items():
                if isinstance(weight, str):
                    Expr(weight)                  # an expression weight must parse now, not fail while the data is drawn
                elif isinstance(weight, bool) or not isinstance(weight, (int, float, list, dict)):
                    raise SamplerError(f"{where}: the weight of {label!r} must be a number or an expression")
    if kind == "time_shift":
        if spec["direction"] not in ("before", "after"):
            raise SamplerError(f"{where}: direction must be before|after")
        if not {"days", "minutes", "seconds"} & spec.keys():
            raise SamplerError(f"{where}: time_shift needs 'days', 'minutes' or 'seconds'")
        for key in ("seconds", "extra_seconds"):
            for bound in spec.get(key) or ():
                if isinstance(bound, str):
                    Expr(bound)
    if kind == "switch":
        for key, case in spec["cases"].items():
            validate(case, f"{where}.cases[{key}]")


class Sampler:
    """Compiled sampler bound to one run (``rt``: RunContext, reference view, hints, memo)."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self._exprs: dict[str, Expr] = {}

    def _expr(self, src: str) -> Expr:
        e = self._exprs.get(src)
        if e is None:
            e = self._exprs[src] = Expr(src)
        return e

    # ------------------------------------------------------------------------------------------
    def draw(self, env: dict[str, Any], rt: "Runtime", concept: str) -> Any:
        value = self._draw(self.spec, env, rt, concept)
        cast = _CASTS.get(self.spec.get("cast"))
        return value if cast is None or value is None else cast(value)

    def _draw(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime", concept: str) -> Any:
        kind = spec["type"]
        rng, ctx = rt.ctx.rng, rt.ctx
        if kind == "const":
            return spec["value"]
        if kind == "expr":
            return self._expr(spec["expr"])(env)
        if kind == "choice":
            return self._choice(spec, env, rt)
        if kind == "id":
            return self._id(spec, env, rt, concept)
        if kind == "pool":
            return self._pool(spec, env, rt)
        if kind == "template":
            return ctx.unique(spec.get("registry", concept), lambda: "".join(self._part(p, rt) for p in spec["parts"]))
        if kind == "lognormal":
            return _lognormal(ctx, spec)
        if kind == "beta_scaled":
            value = rng.betavariate(spec["alpha"], spec["beta"]) * float(self._expr(spec["scale"])(env))
            digits = self._expr(spec["round"])(env) if isinstance(spec.get("round"), str) else spec.get("round")
            return round(value, int(digits)) if digits is not None else value
        if kind == "uniform_int":
            if "p" in spec and not ctx.bernoulli(spec["p"]):
                return None
            return rng.randint(int(self._num(spec["low"], env)), int(self._num(spec["high"], env)))
        if kind == "uniform":
            value = rng.uniform(float(self._num(spec["low"], env)), float(self._num(spec["high"], env)))
            digits = self._num(spec.get("round"), env)
            return round(value, int(digits)) if digits is not None else value
        if kind == "bernoulli":
            return ctx.bernoulli(float(self._num(spec["p"], env)))
        if kind == "time_shift":
            return self._shift(spec, env, rt)
        if kind == "definition":
            from synth import definitions

            if spec.get("unique"):
                return ctx.unique(spec.get("registry") or concept, lambda: definitions.draw(spec, env, rt))
            return definitions.draw(spec, env, rt)
        if kind == "after_parents":
            return self._after(spec, env, rt)
        if kind == "switch":
            case = case_for(spec["cases"], self._expr(spec["by"])(env))
            if case is None:
                return None
            value = self._draw(case, env, rt, concept)
            cast = _CASTS.get(case.get("cast"))          # a case is a sampler in its own right: its cast applies too
            return value if cast is None or value is None else cast(value)
        if kind == "rollup":
            raise SamplerError("a rollup is computed from the entity's events by the engine; it is never drawn")
        raise SamplerError(f"unknown sampler type {kind!r}")

    # ------------------------------------------------------------------------------------------
    def _num(self, value: Any, env: dict[str, Any]) -> Any:
        """A literal number, or the value of an expression over the current variables."""
        return self._expr(value)(env) if isinstance(value, str) else value

    def _choice(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime") -> Any:
        if "by" in spec:
            tables = spec["tables"] if "tables" in spec else rt.ref_path(spec["tables_ref"])
            table = case_for(tables, self._expr(spec["by"])(env))
            if table is None:
                return None
            flt = None
            if isinstance(table, dict) and "weights" in table:
                table, flt = table["weights"], table.get("filter")
        else:
            table, flt = (spec["weights"] if "weights" in spec else rt.ref_path(spec["from_ref"])), spec.get("filter")
        labels = list(table)
        if flt:
            expr = self._expr(flt)
            labels = [c for c in labels if expr({**env, "_choice": c})]
            if not labels:
                raise SamplerError(f"choice filter {flt!r} excluded every option for {env.get('event_type')!r}")
        # a weight is a number or an expression over the current variables and the model parameters ("P['p_retry']")
        weights = [float(self._num(table[c], env) or 0.0) if not isinstance(table[c], (list, dict)) else 1.0 for c in labels]
        return rt.ctx.weighted(labels, weights)

    def _id(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime", concept: str) -> str:
        prefix, digits = str(spec.get("prefix", "")), int(spec["digits"])
        if spec.get("format_hint"):
            hint = rt.hints.get(concept) or {}
            prefix, digits = str(hint.get("prefix", prefix)), int(hint.get("digits", digits))
        hi = 10 ** digits

        def make() -> str:
            return f"{prefix}{rt.ctx.rng.randrange(hi // 10, hi):0{digits}d}"

        if spec.get("stable_by"):
            key = (rt.entity_index, concept, str(self._expr(spec["stable_by"])(env)))
            if key not in rt.memo:
                rt.memo[key] = rt.ctx.unique(spec.get("registry", concept), make)
            return rt.memo[key]
        return rt.ctx.unique(spec.get("registry", concept), make)

    def _pool(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime") -> str:
        """An identifier from a pool that every entity of the run draws from, so several entities share the same party.

        Used for facts that link entities (a common owner, controlling party, counterparty, beneficiary, device or address). The pool
        has ``ratio`` members per entity (at least ``min_size``) and is built once per run, from the run's seed, so it is the same
        for every draw and independent of the order the entities are drawn in. ``skew`` > 0 makes a few members much more popular.
        """
        name = str(spec["name"])
        pool = rt.pools.get(name)
        if pool is None:
            ratio = float(self._num(spec.get("ratio", 0.3), env) or 0.3)
            floor = int(self._num(spec.get("min_size", 3), env) or 3)
            size = max(floor, 2, int(math.ceil(max(1, rt.entities) * max(ratio, 0.0))))
            prefix, digits = str(spec.get("prefix", "")), int(spec["digits"])
            private = random.Random(f"{rt.ctx.seed}|pool|{name}")
            lo, hi = (10 ** (digits - 1) if digits > 1 else 0), 10 ** digits
            members: list[str] = []
            seen: set[str] = set()
            while len(members) < size and len(seen) < hi - lo:
                member = f"{prefix}{private.randrange(lo, hi):0{digits}d}"
                if member not in seen:
                    seen.add(member)
                    members.append(member)
            skew = max(0.0, float(self._num(spec.get("skew", 0.6), env) or 0.0))
            pool = rt.pools[name] = (members, [1.0 / ((i + 1) ** skew) for i in range(len(members))])
        return rt.ctx.weighted(pool[0], pool[1])

    def _part(self, part: dict[str, Any], rt: "Runtime") -> str:
        if "const" in part:
            return str(part["const"])
        if "choice" in part:
            return str(rt.ctx.rng.choice(part["choice"]))
        if "choice_ref" in part:
            return str(rt.ctx.rng.choice(list(rt.ref_path(part["choice_ref"]))))
        if "digits" in part:
            n = int(part["digits"])
            return f"{rt.ctx.rng.randrange(0, 10 ** n):0{n}d}"
        raise SamplerError(f"bad template part {part!r}")

    def _after(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime") -> Any:
        """A time at or after every parent that has one (within each parent's optional ceiling), else None."""
        lowers, uppers = [], []
        for parent in spec["parents"]:
            at = env.get(parent["var"])
            if at is None:
                continue
            lowers.append(at + timedelta(seconds=float(parent.get("min_secs") or 0)))
            if parent.get("max_secs") is not None:
                uppers.append(at + timedelta(seconds=float(parent["max_secs"])))
        if not lowers:
            return None
        lower = max(lowers)
        upper = min(uppers) if uppers else None
        if upper is None:
            return lower + timedelta(seconds=rt.ctx.rng.randint(1, 3600))
        span = int((upper - lower).total_seconds())
        return lower if span <= 0 else lower + timedelta(seconds=rt.ctx.rng.randint(0, span))

    def _shift(self, spec: dict[str, Any], env: dict[str, Any], rt: "Runtime") -> Any:
        anchor = env.get(spec["anchor"])
        if anchor is None:
            return None
        rng = rt.ctx.rng
        delta = timedelta()
        if "days" in spec:
            delta += timedelta(days=_lognormal(rt.ctx, spec["days"]))
        if "minutes" in spec:
            delta += timedelta(minutes=_lognormal(rt.ctx, spec["minutes"]))
        for key in ("seconds", "extra_seconds"):
            if key in spec:
                lo, hi = (float(self._num(b, env)) for b in spec[key])
                delta += timedelta(seconds=rng.uniform(lo, hi))
        return anchor - delta if spec["direction"] == "before" else anchor + delta


def _lognormal(ctx: Any, spec: dict[str, Any]) -> float:
    value = ctx.lognormal_median(spec["median"], spec["sigma"])
    if spec.get("min") is not None:
        value = max(value, spec["min"])
    if spec.get("max") is not None:
        value = min(value, spec["max"])
    return value


class Runtime:
    """What samplers may touch: the seeded context, the reference tables and per-run memo/hints."""

    def __init__(self, ctx: Any, reference: dict[str, Any], hints: dict[str, Any]):
        self.ctx, self.reference, self.hints = ctx, reference, hints
        self.memo: dict[tuple, Any] = {}
        self.pools: dict[str, tuple[list[str], list[float]]] = {}      # shared-party pools, built on first use
        self.entities = 1                                              # how many entities this run draws (sizes the pools)
        self.entity_index = 0

    def ref_path(self, dotted: str) -> Any:
        node: Any = self.reference
        for part in dotted.split("."):
            node = node[part]
        return node

