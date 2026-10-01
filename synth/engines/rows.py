"""Event-rows engine (``event_rows/v1``).

Mechanism for any domain where each row is one *event* of an entity (a notification, a case, a claim, a
transaction, an alert ...) and what matters is that the columns of a row are logically consistent: an offer
cannot be accepted before it was presented, a suppressed contact has a reason and no delivery time, a
resolution never precedes the case that it resolves, and so on.

The engine owns only the time axis; everything else is declared by the behaviour pack's ``emit`` list:

* every entity gets ``per_entity`` event times, strictly increasing, at least ``timeline.min_gap_minutes``
  apart, inside the last ``timeline.history_days`` days before the reference time ``as_of``
  (optionally shaped by a ``timeline.hour_weights_ref`` curve over the local hour of day);
* ``timeline.earliest_next`` (optional expression over ``prev``) pushes an event later when the previous event
  of the same entity forbids another one yet - for example while a cooldown or suppression window is running.
  An entity whose events no longer fit before ``as_of`` is simulated again, so rows never break the rule;
* ``scope: entity`` concepts are drawn once per entity, in declaration order;
* ``scope: event`` concepts are drawn per row, in declaration order, and may read every concept declared
  before them (latent concepts that no column is bound to are allowed and are how a pack expresses the
  hidden decisions that make the visible columns agree with each other).

Variables available to expressions and samplers (engine vocabulary, domain-neutral):

  entity scope : first_event_at, as_of
  event scope  : event_at, event_index, ``prev`` (the previous row of the same entity as ``concept -> value``,
                 or None for the first row), plus the entity-scope variables
  both         : every concept emitted earlier, ``REF`` (the pack's reference tables), ``P`` (model parameters)

This module contains no domain data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from synth.clock import RunContext
from synth.expr import Expr
from synth.pack import BehaviorPack
from synth.samplers import Runtime, Sampler, sampler_names

ENTITY_VARS = frozenset({"first_event_at", "as_of"})
EVENT_VARS = frozenset({"event_at", "event_index", "prev"})
_MAX_ATTEMPTS = 40
_NAMESPACES = frozenset({"REF", "P"})


def _need(tree: dict[str, Any], *path: str) -> Any:
    node: Any = tree
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"behaviour pack is missing model.{'.'.join(path)} (required by engine event_rows/v1)")
        node = node[key]
    return node


def _hour_weights(curve: Any) -> dict[int, float]:
    """Hour-of-day weights from a 24-item list or an ``{hour: weight}`` mapping."""
    items = enumerate(curve) if isinstance(curve, list) else curve.items()
    return {int(h): float(w) for h, w in items}


class EventRows:
    engine_id = "event_rows/v1"

    # ------------------------------------------------------------------ public -------------
    def validate(self, pack: BehaviorPack) -> None:
        """Fail at pack load when an emit reads something that is not defined at that point."""
        seen_event = False
        for emit in pack.emit:
            if emit.scope == "event":
                seen_event = True
            elif seen_event:
                raise ValueError(f"emit '{emit.concept}': entity-scope concepts must be declared before event-scope ones")
        order_known = set(ENTITY_VARS)
        for emit in pack.emit:
            available = order_known | (EVENT_VARS if emit.scope == "event" else set())
            used: set[str] = set()
            for src in (emit.when, emit.expr):
                if src:
                    used |= Expr(src).names
            if emit.sample is not None:
                used |= sampler_names(emit.sample)
            unknown = used - available - _NAMESPACES
            if unknown:
                raise ValueError(
                    f"emit '{emit.concept}' reads {sorted(unknown)}, which are not defined before it "
                    f"(available: engine variables and earlier concepts)"
                )
            order_known.add(emit.concept)
        timeline = (pack.model or {}).get("timeline")
        if not isinstance(timeline, dict) or "history_days" not in timeline or "min_gap_minutes" not in timeline:
            raise ValueError("model.timeline needs history_days and min_gap_minutes (engine event_rows/v1)")
        if timeline.get("earliest_next"):
            unknown = Expr(timeline["earliest_next"]).names - {"prev", "as_of"}
            if unknown:
                raise ValueError(f"model.timeline.earliest_next reads {sorted(unknown)}; only prev and as_of are available")

    def reference_view(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        offset = ctx.tz.utcoffset(ctx.as_of)
        return {
            **pack.reference,
            "currency": pack.currency,
            "model": params,                 # the mode's model parameters (what ``P`` is in emit expressions)
            "as_of": ctx.as_of,
            "tz_offset_min": (offset.total_seconds() / 60.0) if offset else 0.0,
            "min_gap_seconds": _need(params, "timeline", "min_gap_minutes") * 60,
        }

    def simulate(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext, *,
                 entities: int, per_entity: int, hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        run = _Run(pack, params, ctx, self.reference_view(pack, params, ctx), hints or {})
        rows: list[dict[str, Any]] = []
        for index in range(entities):
            run.rt.entity_index = index
            rows.extend(run.entity_rows(per_entity))
        return rows


class _Run:
    def __init__(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext,
                 view: dict[str, Any], hints: dict[str, Any]):
        self.ctx, self.P, self.view = ctx, params, view
        self.rt = Runtime(ctx, view, hints)
        timeline = _need(params, "timeline")
        self.history = timedelta(days=float(_need(timeline, "history_days")))
        self.gap = float(_need(timeline, "min_gap_minutes")) * 60.0
        self.margin = timedelta(minutes=float(timeline.get("margin_minutes", 0)))
        self.earliest = Expr(timeline["earliest_next"]) if timeline.get("earliest_next") else None
        curve = timeline.get("hour_weights_ref")
        self.hour_weights = _hour_weights(self.rt.ref_path(curve)) if curve else None
        self.offset_min = view["tz_offset_min"]
        self.entity_emits = [_Step(e) for e in pack.emit if e.scope == "entity"]
        self.event_emits = [_Step(e) for e in pack.emit if e.scope == "event"]

    # ---- time axis ---------------------------------------------------------------------------
    def _anchors(self, n: int) -> list[datetime]:
        end = self.ctx.as_of - self.margin
        window = self.history.total_seconds()
        slot = window / n
        pad = self.gap / 2.0
        if slot < 2 * pad:
            raise ValueError(
                f"recordsPerUser={n} does not fit in a {window / 86400:.0f}-day history with events at least "
                f"{self.gap / 60:.0f} minutes apart; lower recordsPerUser or raise model.timeline.history_days."
            )
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
            f"recordsPerUser={n} cannot be simulated: the pack's spacing rules (timeline.earliest_next) push the events "
            f"past the reference time within the {self.history.days}-day history. Lower recordsPerUser or widen model.timeline.history_days."
        )

    def _simulate_entity(self, n: int) -> list[dict[str, Any]] | None:
        anchors = self._anchors(n)
        end = self.ctx.as_of - self.margin
        env: dict[str, Any] = {"REF": self.view, "P": self.P, "as_of": self.ctx.as_of, "first_event_at": anchors[0]}
        for step in self.entity_emits:
            env[step.concept] = step.draw(env, self.rt)
        rows: list[dict[str, Any]] = []
        prev: dict[str, Any] | None = None
        at: datetime | None = None
        for i, nominal in enumerate(anchors):
            earliest = nominal
            if at is not None:
                earliest = max(earliest, at + timedelta(seconds=self.gap))
            if self.earliest is not None and prev is not None:
                floor = self.earliest({"prev": prev, "as_of": self.ctx.as_of, "REF": self.view, "P": self.P})
                if floor is not None:
                    earliest = max(earliest, floor)
            if earliest > end:
                return None
            at = earliest
            row_env = dict(env)
            row_env.update(event_at=at, event_index=i, prev=prev)
            for step in self.event_emits:
                row_env[step.concept] = step.draw(row_env, self.rt)
            prev = {s.concept: row_env[s.concept] for s in (*self.entity_emits, *self.event_emits)}
            rows.append(prev)
        return rows


class _Step:
    def __init__(self, emit: Any):
        self.concept = emit.concept
        self.when = Expr(emit.when) if emit.when else None
        self.expr = Expr(emit.expr) if emit.expr else None
        self.sampler = Sampler(emit.sample) if emit.sample is not None else None

    def draw(self, env: dict[str, Any], rt: Runtime) -> Any:
        if self.when is not None and not self.when(env):
            return None
        value = self.expr(env) if self.expr is not None else self.sampler.draw(env, rt, self.concept)  # type: ignore[union-attr]
        # Timestamps are delivered with whole-second precision; keeping them whole here means a rule such as
        # "elapsed <= window" is decided on exactly the values the consumer will see.
        return value.replace(microsecond=0) if isinstance(value, datetime) else value
