"""Event-rows engine (``event_rows/v1``).

Mechanism for any domain where each row is one *event* of an entity (a notification, a case, a claim, a
transaction, an alert ...) and what matters is that the columns of a row are logically consistent: an offer
cannot be accepted before it was presented, a suppressed contact has a reason and no delivery time, a
resolution never precedes the case that it resolves, and so on.

The engine owns only the time axis; everything else is declared by the generation spec's ``emit`` list:

* every entity gets ``per_entity`` event times, strictly increasing, at least ``timeline.min_gap_minutes``
  apart, inside the last ``timeline.history_days`` days before the reference time ``as_of``
  (optionally shaped by a ``timeline.hour_weights`` curve over the local hour of day);
* ``timeline.earliest_next`` (optional expression over ``prev``) pushes an event later when the previous event
  of the same entity forbids another one yet - for example while a cooldown or suppression window is running.
  The event then happens a little after that time (a natural delay of up to a quarter of the usual spacing).
  An entity whose events no longer fit before ``as_of`` is simulated again, so rows never break the rule;
* ``scope: entity`` columns are drawn once per entity, in declaration order;
* ``scope: event`` columns are drawn per row, in declaration order, and may read every column declared
  before them (latent columns, which are never delivered, are how a spec expresses the hidden decisions that
  make the visible columns agree with each other);
* a ``rollup`` (an entity-scope fact such as "when did this customer last accept an offer" or "how many top-ups
  did it make") is computed after the entity's events from those events, so a fact about the entity can never
  contradict the events it summarises; the same value is on every row of the entity.
* the timeline is designed for ``timeline.events`` events per entity (the history length that spaces them
  realistically); asking for more or fewer events per entity stretches or shrinks the history in proportion, so the
  spacing between consecutive events - and everything that depends on it, such as overlapping windows - stays as designed.

Variables available to expressions and samplers (engine vocabulary, domain-neutral):

  entity scope : first_event_at, as_of
  event scope  : event_at, event_index, ``prev`` (the previous row of the same entity as ``column -> value``,
                 or None for the first row), plus the entity-scope variables
  both         : every column emitted earlier, ``REF`` (the spec's reference tables), ``P`` (model parameters)

This module contains no domain data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from synth.clock import RunContext
from synth.expr import Expr
from synth.projection import time_resolution
from synth.spec import GenerationSpec
from synth.samplers import Runtime, Sampler, is_rollup, sampler_names

ENTITY_VARS = frozenset({"first_event_at", "as_of"})
EVENT_VARS = frozenset({"event_at", "event_index", "prev"})
_MAX_ATTEMPTS = 40
MAX_HISTORY_DAYS = 730.0
_NAMESPACES = frozenset({"REF", "P"})


class SpecRuntimeError(ValueError):
    """A spec entry failed while it was being drawn; carries the column so a repair can target it."""

    def __init__(self, column: str, cause: Exception):
        super().__init__(f"column '{column}': {type(cause).__name__}: {cause}")
        self.column, self.cause = column, cause


def _hour_weights(curve: Any) -> dict[int, float]:
    """Hour-of-day weights from a 24-item list or an ``{hour: weight}`` mapping."""
    items = enumerate(curve) if isinstance(curve, list) else curve.items()
    return {int(h): float(w) for h, w in items}


class EventRows:
    engine_id = "event_rows/v1"

    # ------------------------------------------------------------------ public -------------
    def validate(self, spec: GenerationSpec) -> None:
        """Fail at load when an emit reads something that is not defined at that point."""
        seen_event = False
        for emit in spec.emit:
            if is_rollup(emit.sample):
                continue
            if emit.scope == "event":
                seen_event = True
            elif seen_event:
                raise ValueError(f"emit '{emit.column}': entity-scope columns must be declared before event-scope ones")
        known = set(ENTITY_VARS)
        rolled: set[str] = set()
        for emit in spec.emit:
            rollup = is_rollup(emit.sample)
            if not rollup and rolled:
                raise ValueError(f"emit '{emit.column}' is declared after a rollup; rollups come last")
            available = known | (EVENT_VARS if emit.scope == "event" and not rollup else set())
            used: set[str] = set()
            for src in (emit.when, emit.expr):
                if src:
                    used |= Expr(src).names
            if emit.sample is not None:
                used |= sampler_names(emit.sample)
            unknown = used - available - _NAMESPACES
            if unknown:
                raise ValueError(
                    f"emit '{emit.column}' reads {sorted(unknown)}, which are not defined before it "
                    f"(available: engine variables and earlier columns)"
                )
            if rollup:
                if emit.scope != "entity":
                    raise ValueError(f"emit '{emit.column}': a rollup describes the entity and must be entity-scoped")
                rolled.add(emit.column)
            known.add(emit.column)
        timeline = spec.timeline or {}
        if "history_days" not in timeline:
            raise ValueError("timeline needs history_days (engine event_rows/v1)")
        if timeline.get("earliest_next"):
            unknown = Expr(timeline["earliest_next"]).names - {"prev", "as_of"}
            if unknown:
                raise ValueError(f"timeline.earliest_next reads {sorted(unknown)}; only prev and as_of are available")

    def reference_view(self, spec: GenerationSpec, ctx: RunContext) -> dict[str, Any]:
        offset = ctx.tz.utcoffset(ctx.as_of)
        return {
            **spec.reference,
            "currency": spec.currency,
            "country": (spec.scenario or {}).get("country"),
            "columns": spec.delivered,       # id -> delivered column name (what ``definition`` samplers read their record by)
            "model": spec.model,             # what ``P`` is in emit expressions
            "as_of": ctx.as_of,
            "tz_offset_min": (offset.total_seconds() / 60.0) if offset else 0.0,
        }

    def simulate(self, spec: GenerationSpec, ctx: RunContext, *,
                 entities: int, per_entity: int, hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        run = _Run(spec, ctx, self.reference_view(spec, ctx), hints or {})
        rows: list[dict[str, Any]] = []
        for index in range(entities):
            run.rt.entity_index = index
            rows.extend(run.entity_rows(per_entity))
        return rows


class _Run:
    def __init__(self, spec: GenerationSpec, ctx: RunContext, view: dict[str, Any], hints: dict[str, Any]):
        self.ctx, self.P, self.view = ctx, spec.model, view
        self.rt = Runtime(ctx, view, hints)
        timeline = spec.timeline
        self.history_days = float(timeline["history_days"])
        self.events_ref = max(1.0, float(timeline.get("events") or 0)) if timeline.get("events") else None
        self.gap = float(timeline.get("min_gap_minutes", 0)) * 60.0
        self.margin = timedelta(minutes=float(timeline.get("margin_minutes", 0)))
        self.earliest = Expr(timeline["earliest_next"]) if timeline.get("earliest_next") else None
        curve = timeline.get("hour_weights") or (self.rt.ref_path(timeline["hour_weights_ref"]) if timeline.get("hour_weights_ref") else None)
        self.hour_weights = _hour_weights(curve) if curve else None
        self.offset_min = view["tz_offset_min"]
        resolution = time_resolution(spec)
        self.entity_emits = [_Step(e, resolution) for e in spec.emit if e.scope == "entity" and not is_rollup(e.sample)]
        self.event_emits = [_Step(e, resolution) for e in spec.emit if e.scope == "event"]
        self.rollups = [_Rollup(e, resolution) for e in spec.emit if is_rollup(e.sample)]
        self.resolution = resolution

    # ---- time axis ---------------------------------------------------------------------------
    def window_days(self, n: int) -> float:
        """Days of history for ``n`` events: the designed length, scaled with the number of events asked for."""
        days = self.history_days * (n / self.events_ref) if self.events_ref else self.history_days
        return min(MAX_HISTORY_DAYS, max(1.0, days))

    def _anchors(self, n: int) -> list[datetime]:
        end = self.ctx.as_of - self.margin
        window = self.window_days(n) * 86400.0
        slot = window / n
        # Events are at least ``gap`` apart, but never so far apart that ``n`` of them cannot fit the history.
        pad = min(self.gap / 2.0, slot * 0.45)
        start = end - timedelta(seconds=window)
        out: list[datetime] = []
        for k in range(n):
            lo, hi = k * slot + pad, (k + 1) * slot - pad
            t = start + timedelta(seconds=self.ctx.rng.uniform(lo, hi))
            if self.hour_weights:
                top = max(self.hour_weights.values())
                for _ in range(40):
                    hour = (t + timedelta(minutes=self.offset_min)).hour
                    if self.ctx.rng.random() * top <= self.hour_weights.get(hour, 0.0):
                        break
                    t = start + timedelta(seconds=self.ctx.rng.uniform(lo, hi))
            out.append(t.replace(microsecond=0))
        return out

    # ---- rows --------------------------------------------------------------------------------
    def entity_rows(self, n: int) -> list[dict[str, Any]]:
        for _ in range(_MAX_ATTEMPTS):
            rows = self._simulate_entity(n)
            if rows is not None:
                return rows
        raise ValueError(
            f"recordsPerUser={n} cannot be simulated: the spec's spacing rule (timeline.earliest_next) pushes the events "
            f"past the reference time within the {self.window_days(n):.0f}-day history. Lower recordsPerUser or widen timeline.history_days."
        )

    def _simulate_entity(self, n: int) -> list[dict[str, Any]] | None:
        anchors = self._anchors(n)
        end = self.ctx.as_of - self.margin
        gap = min(self.gap, self.window_days(n) * 86400.0 / n * 0.9)
        env: dict[str, Any] = {"REF": self.view, "P": self.P, "as_of": self.ctx.as_of, "first_event_at": anchors[0]}
        for step in self.entity_emits:
            env[step.column] = step.draw(env, self.rt)
        rows: list[dict[str, Any]] = []
        prev: dict[str, Any] | None = None
        at: datetime | None = None
        for i, nominal in enumerate(anchors):
            earliest = nominal
            if at is not None:
                earliest = max(earliest, at + timedelta(seconds=gap))
            if self.earliest is not None and prev is not None:
                floor = self.earliest({"prev": prev, "as_of": self.ctx.as_of, "REF": self.view, "P": self.P})
                if floor is not None and floor > earliest:
                    # pushed by the rule: the next event follows the end of what the previous one opened after some natural
                    # delay (up to a quarter of the usual spacing), not at the very instant it becomes possible
                    earliest = floor + timedelta(seconds=self.ctx.rng.uniform(0.0, 0.25 * self.window_days(n) * 86400.0 / n))
            if earliest > end:
                return None
            at = earliest
            row_env = dict(env)
            row_env.update(event_at=at, event_index=i, prev=prev)
            for step in self.event_emits:
                row_env[step.column] = step.draw(row_env, self.rt)
            prev = {s.column: row_env[s.column] for s in (*self.entity_emits, *self.event_emits)}
            rows.append(prev)
        for rollup in self.rollups:
            value = rollup.compute(rows, env, self.rt)
            env[rollup.column] = value
            for row in rows:
                row[rollup.column] = value
        return rows


class _Step:
    def __init__(self, emit: Any, resolution: int = 1):
        self.column = emit.column
        self.resolution = resolution
        self.when = Expr(emit.when) if emit.when else None
        self.expr = Expr(emit.expr) if emit.expr else None
        self.sampler = Sampler(emit.sample) if emit.sample is not None else None

    def draw(self, env: dict[str, Any], rt: Runtime) -> Any:
        try:
            if self.when is not None and not self.when(env):
                return None
            value = self.expr(env) if self.expr is not None else self.sampler.draw(env, rt, self.column)  # type: ignore[union-attr]
        except SpecRuntimeError:
            raise
        except Exception as exc:
            raise SpecRuntimeError(self.column, exc) from exc
        # Timestamps are delivered at the resolution of their layout; keeping them at it here means a rule such as
        # "elapsed <= window" is decided on exactly the values the consumer will see.
        if isinstance(value, datetime):
            return value.replace(second=0, microsecond=0) if self.resolution >= 60 else value.replace(microsecond=0)
        return value


class _Rollup:
    """An entity fact computed from the entity's own events (oldest first): first/last/min/max/sum/mean/count/any/all/distinct."""

    def __init__(self, emit: Any, resolution: int = 1):
        spec = emit.sample
        self.column, self.fn, self.resolution = emit.column, spec["fn"], resolution
        self.of = Expr(spec["of"]) if spec.get("of") else None
        self.where = Expr(spec["where"]) if spec.get("where") else None
        self.when = Expr(emit.when) if emit.when else None
        self.cast = spec.get("cast")

    def compute(self, rows: list[dict[str, Any]], entity_env: dict[str, Any], rt: Runtime) -> Any:
        try:
            if self.when is not None and not self.when(entity_env):
                return None
            values: list[Any] = []
            hits = 0
            for row in rows:
                env = {**entity_env, **row}
                if self.where is not None and not self.where(env):
                    continue
                hits += 1
                if self.of is not None:
                    values.append(self.of(env))
            return self._finish(values, hits)
        except SpecRuntimeError:
            raise
        except Exception as exc:
            raise SpecRuntimeError(self.column, exc) from exc

    def _finish(self, values: list[Any], hits: int) -> Any:
        fn = self.fn
        if fn == "count":
            return hits if self.of is None else sum(1 for v in values if v is not None and v is not False)
        if fn == "any":
            return any(bool(v) for v in values) if self.of is not None else hits > 0
        if fn == "all":
            return all(bool(v) for v in values) if self.of is not None else True
        present = [v for v in values if v is not None]
        if not present:
            return 0 if fn == "sum" else None
        if fn == "first":
            out: Any = present[0]
        elif fn == "last":
            out = present[-1]
        elif fn == "min":
            out = min(present)
        elif fn == "max":
            out = max(present)
        elif fn == "sum":
            out = sum(present)
        elif fn == "mean":
            out = sum(present) / len(present)
        else:                                                   # distinct
            out = len({str(v) for v in present})
        if isinstance(out, datetime):
            return out.replace(second=0, microsecond=0) if self.resolution >= 60 else out.replace(microsecond=0)
        if self.cast == "int" and isinstance(out, (int, float)):
            return int(round(out))
        if self.cast == "float" and isinstance(out, (int, float)):
            return float(out)
        return out
