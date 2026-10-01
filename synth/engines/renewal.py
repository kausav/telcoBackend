"""Renewal-journey engine (``renewal_journey/v1``).

Mechanism for any domain where an entity keeps coming back to renew or replenish something:
prepaid balance, insurance premium, utility bill, subscription, loan instalment, card limit ...

Each entity is simulated as a *renewal process* running forward in time up to the reference time
``as_of``; the last ``per_entity`` episodes are returned. An *episode* is::

    need arises  ->  [operator offer -> customer response]  ->  request  ->  confirmation / failure / abandonment

Consistency is by construction, not by repair: every column of a row is derived from the same latent
decisions (segment, archetype, offer, acceptance, status), so a confirmation only exists for a
successful request, an accepted offer's amount *is* the purchased product, a retry always follows a
failed attempt, and so on.

This module contains **no domain data**. Everything a domain needs is in the behaviour pack:

* ``reference``: segments, archetypes, event types, product catalogs, channels, status labels, diurnal curve;
* ``model``: probabilities, delays and timing rules (``history``, ``timing``, ``retry``, ``renewal_lag``,
  ``intervene``, ``offer``, ``accept``, ``status``, ``addon``);
* ``emit``: which columns exist and how each is derived from the variables below.

Variables available to ``emit`` expressions and samplers (engine vocabulary, domain-neutral):

  entity scope : segment, archetype, first_event_at, last_event_at
  event scope  : event_type, usage, bucket, catalog, segment, archetype, is_auto, is_retry, status (pack label),
                 status_role (ok|fail|abandon|pending), channel, method, product_id, product_amount,
                 product_duration_days, offer_made, offer_channel, offer_amount, offer_accepted,
                 request_at, trigger_at, response_at, confirmed_at, threshold
  both         : every concept emitted earlier (entity concepts first), ``REF`` (reference view), ``P`` (model)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from synth.clock import RunContext
from synth.expr import Expr
from synth.pack import BehaviorPack
from synth.samplers import Runtime, Sampler

_MIN = 60.0
_DAY = 86400.0
OK, FAIL, ABANDON, PENDING = "ok", "fail", "abandon", "pending"      # engine roles; labels come from the pack


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _Profile:
    segment: str
    archetype: str
    usual: dict[str, str]                 # catalog -> preferred product id
    responsiveness: float
    auto: bool = False


@dataclass
class _Event:
    type: str
    anchor: datetime                       # coarse schedule time (day-level)
    retry: bool
    is_auto: bool
    product: dict[str, Any]
    channel: str
    method: str
    status: str                            # role
    intervened: bool = False
    offer_channel: str | None = None
    offer_product: dict[str, Any] | None = None
    accepted: bool = False
    decline_explicit: bool = False
    retry_of: "_Event | None" = None       # the failed/abandoned attempt this episode retries
    t_request: datetime | None = None
    trigger_at: datetime | None = None     # finalised by ``_finalise``
    response_at: datetime | None = None
    confirmed_at: datetime | None = None
    inflight: bool = False


class RenewalJourney:
    engine_id = "renewal_journey/v1"

    # ------------------------------------------------------------------ public -------------
    def reference_view(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        """The ``REF`` dictionary: the pack's reference tables plus views derived from them and the run."""
        ref = pack.reference
        usages: dict[str, set] = {}
        for spec in ref["event_types"].values():
            usages.setdefault(spec["usage"], set()).update(p["amount"] for p in ref["catalog"][spec["catalog"]])
        offset = ctx.tz.utcoffset(ctx.as_of)
        return {
            **ref,
            "currency": pack.currency,
            "amounts": {u: sorted(v) for u, v in usages.items()},
            "max_threshold": {u: max(s["threshold"][u] for s in ref["segments"].values()) for u in usages},
            "as_of": ctx.as_of,
            "max_history_days": _need(pack.output, "max_history_days"),
            "min_gap_seconds": _need(params, "history", "min_gap_minutes") * 60,
            "tz_offset_min": (offset.total_seconds() / 60.0) if offset else 0.0,
        }

    def simulate(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext, *,
                 entities: int, per_entity: int, hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        run = _Run(pack, params, ctx, per_entity, hints or {}, self.reference_view(pack, params, ctx))
        rows: list[dict[str, Any]] = []
        for index in range(entities):
            run.rt.entity_index = index
            rows.extend(run.entity_rows())
        return rows


def _need(tree: dict[str, Any], *path: str) -> Any:
    node: Any = tree
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"behaviour pack is missing '{'.'.join(path)}' (required by engine renewal_journey/v1)")
        node = node[key]
    return node


class _Run:
    """One simulation run (holds pack, parameters, RNG and lookup tables)."""

    def __init__(self, pack: BehaviorPack, P: dict[str, Any], ctx: RunContext, per_entity: int,
                 hints: dict[str, Any], view: dict[str, Any]):
        self.pack, self.P, self.ctx, self.N, self.view = pack, P, ctx, per_entity, view
        self.rng = ctx.rng
        self.rt = Runtime(ctx, view, hints)
        ref = pack.reference
        self.segments: dict[str, Any] = ref["segments"]
        self.catalog: dict[str, list[dict[str, Any]]] = ref["catalog"]
        self.event_types: dict[str, Any] = ref["event_types"]
        self.archetypes: dict[str, Any] = ref["archetypes"]
        self.labels: dict[str, str] = ref["status_labels"]
        self.products = {p["id"]: p for items in self.catalog.values() for p in items}
        self.max_days = float(_need(pack.output, "max_history_days"))
        self.H, self.T = _need(P, "history"), _need(P, "timing")
        self.min_gap = float(self.H["min_gap_minutes"]) * _MIN
        self.pad = float(self.T["spacing_pad_minutes"]) * _MIN
        self.diurnal = list(ref["diurnal_weights_local_hour"])
        self._seg_names = list(self.segments)
        self._seg_weights = [self.segments[s]["weight"] for s in self._seg_names]
        channels: dict[str, Any] = ref["channels"]
        self._organic = {c: v for c, v in channels.items() if not v.get("auto")}
        autos = [c for c, v in channels.items() if v.get("auto")]
        self._auto_channel = autos[0] if autos else None
        self._channels = channels
        self._org_names = list(self._organic)
        self._org_weights = [self._organic[c]["weight"] for c in self._org_names]
        self._entity_emit = [self._compile(e) for e in pack.emit if e.scope == "entity"]
        self._event_emit = [self._compile(e) for e in pack.emit if e.scope == "event"]

    def _compile(self, e: Any) -> tuple[str, Expr | None, Expr | None, Sampler | None]:
        return (e.concept, Expr(e.when) if e.when else None, Expr(e.expr) if e.expr else None,
                Sampler(e.sample) if e.sample else None)

    # ------------------------------------------------------------------ helpers ------------
    def _w(self, mapping: dict[Any, float]) -> Any:
        keys = list(mapping)
        return self.ctx.weighted(keys, [mapping[k] for k in keys])

    def _lognorm(self, spec: dict[str, Any], unit: float = 1.0) -> float:
        """Sample ``spec = {median, sigma, [min], [max]}``; ``unit`` converts the spec's unit to seconds."""
        value = self.ctx.lognormal_median(spec["median"], spec["sigma"]) * unit
        if spec.get("min") is not None:
            value = max(value, spec["min"] * unit)
        if spec.get("max") is not None:
            value = min(value, spec["max"] * unit)
        return value

    def _seg_products(self, seg: str, catalog: str) -> dict[str, float]:
        products = self.segments[seg]["products"].get(catalog)
        if products:
            return products
        return self.segments[_need(self.P, "catalog_fallback_segment")]["products"][catalog]

    def _arche(self, name: str) -> dict[str, Any]:
        return self.archetypes[name]

    # ------------------------------------------------------------------ profile ------------
    def _profile(self) -> _Profile:
        ctx = self.ctx
        seg = ctx.weighted(self._seg_names, self._seg_weights)
        archetype = self._w(self.segments[seg]["archetypes"])
        arche = self._arche(archetype)
        usual = {c: self._w(self._seg_products(seg, c)) for c in self.catalog}
        if arche.get("auto"):
            catalog = self.event_types[arche["event"]]["catalog"]
            allowed = set(arche.get("allowed_duration_days") or [])
            options = {pid: w for pid, w in self._seg_products(seg, catalog).items()
                       if not allowed or self.products[pid]["duration_days"] in allowed}
            usual[catalog] = self._w(options or self._seg_products(seg, catalog))
        acc = _need(self.P, "accept")
        mean, shape = float(acc["responsiveness_mean"]), float(acc["responsiveness_shape"])
        resp = self.rng.betavariate(shape, shape * (1.0 - mean) / mean)
        return _Profile(segment=seg, archetype=archetype, usual=usual, responsiveness=resp,
                        auto=bool(arche.get("auto")))

    # ------------------------------------------------------------------ one event ----------
    def _event(self, prof: _Profile, etype: str, anchor: datetime, retry: bool) -> _Event:
        P, ctx = self.P, self.ctx
        spec = self.event_types[etype]
        catalog = spec["catalog"]
        seg = prof.segment
        is_auto = prof.auto and bool(spec.get("auto_eligible"))
        intervened = False
        offer_channel = None
        offer_product = None
        accepted = False
        decline = False
        stick = float(self.H["pack_stickiness"])

        if is_auto:
            if self._auto_channel is None:
                raise ValueError("behaviour pack has auto-eligible events but no channel flagged 'auto'")
            allowed = set(self._arche(prof.archetype).get("allowed_duration_days") or [])
            usual = prof.usual[catalog]
            if ctx.bernoulli(stick):
                product = self.products[usual]
            else:
                options = {pid: w for pid, w in self._seg_products(seg, catalog).items()
                           if not allowed or self.products[pid]["duration_days"] in allowed}
                product = self.products[self._w(options or self._seg_products(seg, catalog))]
            channel = self._auto_channel
        else:
            usual_id = prof.usual[catalog]
            organic_id = usual_id if ctx.bernoulli(stick) else self._w(self._seg_products(seg, catalog))
            product = self.products[organic_id]
            channel = ctx.weighted(self._org_names, self._org_weights)
            iv = P["intervene"]
            p_int = min(1.0, iv["p_by_segment"][seg] * float(iv["p_multiplier"]))
            if retry:
                p_int *= float(P["retry"]["intervene_mult"])
            if ctx.bernoulli(p_int) and ctx.bernoulli(iv["consent_by_segment"][seg]):
                intervened = True
                offer_channel = self._w(iv["channel_weights"][seg])
                offer_product = self._offer(prof, catalog)
                acc = P["accept"]
                logit = (acc["base_logit"] + acc["logit_shift"]
                         + acc["responsiveness_coef"] * (prof.responsiveness - acc["responsiveness_mean"])
                         + (acc["match_bonus"] if offer_product["id"] == usual_id else 0.0)
                         + acc["channel_shift"].get(offer_channel, 0.0)
                         + acc["segment_shift"].get(seg, 0.0))
                accepted = ctx.bernoulli(_sigmoid(logit))
                if accepted:
                    product = offer_product
                else:
                    decline = ctx.bernoulli(acc["decline_share"])

        method = self._w(self._channels[channel]["methods"])
        status = self._status(self._channels[channel]["fail"], accepted, is_auto)
        return _Event(type=etype, anchor=anchor, retry=retry, is_auto=is_auto, product=product,
                      channel=channel, method=method, status=status, intervened=intervened,
                      offer_channel=offer_channel, offer_product=offer_product, accepted=accepted,
                      decline_explicit=decline)

    def _offer(self, prof: _Profile, catalog: str) -> dict[str, Any]:
        """What the operator recommends: usually the customer's usual product, sometimes an upsell."""
        o = self.P["offer"]
        seg_items = sorted((self.products[p] for p in self._seg_products(prof.segment, catalog)), key=lambda p: p["amount"])
        usual = self.products[prof.usual[catalog]]
        u = self.rng.random()
        if u < o["recommend_usual_p"]:
            return usual
        if u < o["recommend_usual_p"] + o["recommend_upsell_p"]:
            higher = [p for p in seg_items if p["amount"] > usual["amount"]]
            return higher[0] if higher else usual
        others = [p for p in seg_items if p["id"] != usual["id"]]
        return self.rng.choice(others) if others else usual

    def _status(self, chan_fail: float, accepted: bool, is_auto: bool) -> str:
        S = self.P["status"]
        fail = S["fail_abs"] if S.get("fail_abs") is not None else chan_fail * float(S["fail_mult"])
        abandon = (S["abandon_abs"] if S.get("abandon_abs") is not None
                   else float(S["abandon_p"]) * float(S["abandon_mult"]))
        if accepted and S.get("abandon_abs") is None:
            abandon *= float(S["accepted_abandon_mult"])
        if is_auto and S.get("abandon_abs") is None:
            abandon = 0.0          # an automatic request has no user session to abandon
        u = self.rng.random()
        if u < fail:
            return FAIL
        if u < fail + abandon:
            return ABANDON
        return OK

    # ------------------------------------------------------------------ one entity ---------
    def entity_rows(self) -> list[dict[str, Any]]:
        ctx, H = self.ctx, self.H
        prof = self._profile()
        bound = ctx.as_of - timedelta(minutes=float(self.T["as_of_margin_minutes"]))
        est = self._cycle_estimate(prof)
        span_days = min(self.max_days, max(float(H["span_min_days"]),
                                           (self.N + H["span_extra_episodes"]) * est * float(H["span_margin"])))
        while True:
            events = self._forward(prof, span_days, bound)
            if len(events) >= self.N or span_days >= self.max_days:
                break
            span_days = min(self.max_days, span_days * float(H["span_growth"]))
        if len(events) < self.N:
            # Sparse long-cycle entity: fall back to the shortest products so the requested history fits.
            base = self._arche(prof.archetype)["event"]
            catalog = self.event_types[base]["catalog"]
            prof.usual[catalog] = min(self.catalog[catalog], key=lambda p: p["duration_days"] or 0)["id"]
            events = self._forward(prof, self.max_days, bound, force_short=True)
        if len(events) < self.N:
            raise RuntimeError(
                f"Could not simulate {self.N} episodes within {self.max_days:.0f} days for archetype "
                f"'{prof.archetype}'; raise output.max_history_days or lower recordsPerUser."
            )
        chosen_from = len(events) - self.N
        last = events[-1]
        # Only a request that is recent anyway can still be in flight at the reference time; an old one
        # (e.g. a retry that happened weeks ago) is never stretched forward to look pending.
        recent = (ctx.as_of - last.t_request) <= timedelta(hours=float(H["inflight_max_age_hours"]))
        if recent and ctx.bernoulli(H["in_flight_p"]):
            last.inflight = True
            lo, hi = H["inflight_offset_minutes"]
            last.t_request = max(last.t_request, ctx.as_of - timedelta(minutes=self.rng.uniform(lo, hi)))
            last.status = PENDING
        chosen = events[chosen_from:]
        for i, ev in enumerate(chosen, start=chosen_from):
            self._finalise(ev, events[i - 1].t_request if i > 0 else None)
        entity = self._emit_entity(prof, chosen)
        return [self._emit_event(prof, ev, entity) for ev in chosen]

    def _cycle_estimate(self, prof: _Profile) -> float:
        arche = self._arche(prof.archetype)
        spec = self.event_types[arche["event"]]
        if spec["schedule"] == "interval":
            return float(arche["interval_days"][prof.segment])
        days = [self.products[p]["duration_days"] for p in self._seg_products(prof.segment, spec["catalog"])]
        return sum(days) / len(days)

    def _forward(self, prof: _Profile, span_days: float, bound: datetime, force_short: bool = False) -> list[_Event]:
        """Run the renewal process from ``as_of - span`` to ``as_of``; events with t <= bound are kept."""
        ctx, rng = self.ctx, self.rng
        arche = self._arche(prof.archetype)
        base_type = arche["event"]
        base = self.event_types[base_type]
        lag_cfg, retry_cfg = self.P["renewal_lag"], self.P["retry"]
        start = ctx.as_of - timedelta(days=span_days)
        first_cycle = max(float(self.H["interval_min_days"]), self._cycle_estimate(prof))
        t = start + timedelta(days=rng.uniform(0, first_cycle))
        out: list[_Event] = []
        retry = False
        failed_prev: _Event | None = None
        guard = 0
        while t <= bound and guard < 5000:
            guard += 1
            ev = self._event(prof, base_type, t, retry)
            ev.retry_of = failed_prev if retry else None
            if force_short and base["schedule"] == "product_validity" and not ev.is_auto:
                ev.product = min(self.catalog[base["catalog"]], key=lambda p: p["duration_days"])
                if ev.accepted:
                    ev.offer_product = ev.product      # an accepted offer *is* the purchased product
            out.append(ev)

            if ev.status != OK:
                t_next, retry, failed_prev = t + timedelta(seconds=self._retry_gap(retry_cfg)), True, ev
            else:
                retry, failed_prev = False, None
                if base["schedule"] == "interval":
                    days = ctx.lognormal_median(arche["interval_days"][prof.segment], arche["sigma"])
                    t_next = t + timedelta(days=max(float(self.H["interval_min_days"]), days))
                else:
                    days = ev.product["duration_days"]
                    if ev.is_auto:
                        lag = rng.uniform(0.0, lag_cfg["auto_max_days"])
                    elif ctx.bernoulli(lag_cfg["early_p"]):
                        lag = -rng.uniform(0.0, lag_cfg["early_max_days"])
                    else:
                        lag = min(lag_cfg["late_max_days"], ctx.lognormal_median(lag_cfg["late_median_days"], lag_cfg["late_sigma"]))
                    t_next = t + timedelta(days=max(float(lag_cfg["min_days"]), days + lag))
                    self._addons(prof, arche, t, t_next, bound, out)
            t = t_next
        out.sort(key=lambda e: e.anchor)
        self._place_all(out, bound)
        return [e for e in out if e.t_request <= bound]

    def _retry_gap(self, retry_cfg: dict[str, Any]) -> float:
        gap = self._lognorm({"median": retry_cfg["median_h"], "sigma": retry_cfg["sigma"]}, 3600.0)
        return min(max(gap, retry_cfg["min_minutes"] * _MIN), retry_cfg["max_h"] * 3600.0)

    def _addons(self, prof: _Profile, arche: dict[str, Any], t0: datetime, t1: datetime,
                bound: datetime, out: list[_Event]) -> None:
        rate = arche.get("addon_per_cycle") or 0.0
        if isinstance(rate, dict):
            rate = rate.get(prof.segment, 0.0)
        margin = float(_need(self.P, "addon", "margin_days")) * _DAY
        window = (t1 - t0).total_seconds() - margin
        if not rate or window <= 0 or not self.ctx.bernoulli(rate):
            return
        count = 1 + (1 if self.ctx.bernoulli(arche.get("second_addon") or 0.0) else 0)
        retry_cfg = self.P["retry"]
        for _ in range(count):
            at = t0 + timedelta(seconds=margin / 2 + self.rng.uniform(0, window))
            retry = False
            failed_prev = None
            for _attempt in range(int(retry_cfg["addon_attempts"])):
                if at > bound:
                    break
                ev = self._event(prof, arche["addon_event"], at, retry)
                ev.retry_of = failed_prev
                out.append(ev)
                # a failed/abandoned add-on purchase is usually retried shortly afterwards
                if ev.status == OK or not self.ctx.bernoulli(retry_cfg["addon_retry_p"]):
                    break
                at, retry, failed_prev = at + timedelta(seconds=self._retry_gap(retry_cfg)), True, ev

    # ------------------------------------------------------------------ timing -------------
    def _place_all(self, events: list[_Event], bound: datetime) -> None:
        """Give every episode a request time, strictly after the previous one.

        New needs are placed on the diurnal curve; a retry happens a realistic delay after the attempt
        that failed (people retry within hours, and rarely in the small hours of the night).
        """
        gap = timedelta(seconds=self.min_gap + self.pad)
        prev: datetime | None = None
        for ev in events:
            lower = (prev + gap) if prev else None
            if ev.retry_of is not None and ev.retry_of.t_request is not None:
                ev.t_request = self._place_retry(ev, lower)
            else:
                ev.t_request = self._place(ev.anchor, lower, bound)
            prev = ev.t_request

    def _place_retry(self, ev: _Event, lower: datetime | None) -> datetime:
        T = self.T
        origin = ev.retry_of
        cand = origin.t_request + (ev.anchor - origin.anchor)
        if lower is not None:
            cand = max(cand, lower)
        local = cand.astimezone(self.ctx.tz)
        if local.hour < int(T["night_end_hour"]) and self.ctx.bernoulli(T["night_defer_p"]):   # try again after the night
            cand = (local.replace(hour=int(T["night_end_hour"]), minute=0, second=0, microsecond=0)
                    + timedelta(minutes=self.rng.uniform(0, T["night_defer_window_minutes"]))).astimezone(self.ctx.as_of.tzinfo)
        return cand

    def _place(self, anchor: datetime, lower: datetime | None, bound: datetime) -> datetime:
        """Pick a time of day on the anchor's local date from the diurnal curve, within [lower, bound]."""
        ctx = self.ctx
        if lower is not None and lower > bound:
            return lower                                   # beyond the reference time: the caller drops it
        local = anchor.astimezone(ctx.tz)
        bound_local = bound.astimezone(ctx.tz)
        for _ in range(40):
            weights = list(self.diurnal)
            if local.date() == bound_local.date():         # today: only hours that have already happened
                weights = [w if h < bound_local.hour else 0.0 for h, w in enumerate(weights)]
            if sum(weights) <= 0:
                break
            hour = ctx.weighted(list(range(24)), weights)
            cand = (local.replace(hour=hour, minute=0, second=0, microsecond=0)
                    + timedelta(seconds=self.rng.randrange(0, 3600))).astimezone(ctx.as_of.tzinfo)
            if cand <= bound and (lower is None or cand >= lower):
                return cand
        if lower is not None:                              # the day is already taken: continue right after the previous one
            return lower + timedelta(seconds=self.rng.uniform(*self.T["overflow_seconds"]))
        return min(anchor, bound)

    # ------------------------------------------------------------------ finalise ----------
    def _finalise(self, ev: _Event, prev_t: datetime | None) -> None:
        """Derive every timestamp of the episode from the request time, bounded by the previous episode."""
        rng, P = self.rng, self.P
        t_r = ev.t_request
        room = ((t_r - prev_t).total_seconds() - self.pad) if prev_t else float("inf")   # seconds before t_r we may use
        acc, iv = P["accept"], P["intervene"]
        dispatch_spec = iv["dispatch_delay_s"]
        margin = float(self.T["response_margin_seconds"])

        if ev.accepted:
            ok = False
            for _ in range(12):
                lag = rng.uniform(*acc["request_lag_s"])
                resp = self._lognorm(acc["response_delay_min"][ev.offer_channel], 60.0)
                disp = self._lognorm(dispatch_spec)
                if lag + resp + disp <= room:
                    ok = True
                    break
            if ok:
                ev.response_at = t_r - timedelta(seconds=lag)
                ev.trigger_at = t_r - timedelta(seconds=lag + resp + disp)
            else:   # the previous episode is too recent for a full offer round-trip: the offer did not convert
                ev.accepted = False
                ev.decline_explicit = False
        if not ev.accepted:
            o = P["organic_delay_h"]
            spec = {**o[ev.type], "max": o["max"]}
            delay = self._lognorm(spec, 3600.0)
            delay = max(o["min_minutes"] * _MIN, min(delay, room))
            ev.trigger_at = t_r - timedelta(seconds=delay)
            if ev.intervened and ev.decline_explicit:
                sent = ev.trigger_at + timedelta(seconds=self._lognorm(dispatch_spec))
                resp_at = sent + timedelta(seconds=self._lognorm(acc["response_delay_min"][ev.offer_channel], 60.0))
                ev.response_at = resp_at if resp_at < t_r - timedelta(seconds=margin) else None

        if ev.status == OK:
            ev.confirmed_at = t_r + timedelta(seconds=self._lognorm(P["status"]["confirm_lag_s"]))

    # ------------------------------------------------------------------ emit ---------------
    def _run_emit(self, specs: list, env: dict[str, Any], out: dict[str, Any]) -> None:
        for concept, when, expr, sampler in specs:
            try:
                if when is not None and not when(env):
                    value = None
                elif expr is not None:
                    value = expr(env)
                else:
                    value = sampler.draw(env, self.rt, concept)
            except Exception as exc:
                raise ValueError(f"behaviour pack emit '{concept}' failed: {type(exc).__name__}: {exc}") from exc
            env[concept] = out[concept] = value

    def _emit_entity(self, prof: _Profile, events: list[_Event]) -> dict[str, Any]:
        env: dict[str, Any] = {
            "REF": self.view, "P": self.P, "segment": prof.segment, "archetype": prof.archetype,
            "first_event_at": events[0].t_request, "last_event_at": events[-1].t_request,
        }
        out: dict[str, Any] = {}
        self._run_emit(self._entity_emit, env, out)
        return out

    def _emit_event(self, prof: _Profile, ev: _Event, entity: dict[str, Any]) -> dict[str, Any]:
        spec = self.event_types[ev.type]
        usage = spec["usage"]
        product = ev.product
        env: dict[str, Any] = {
            "REF": self.view, "P": self.P, **entity,
            "event_type": ev.type, "usage": usage, "bucket": spec["bucket"], "catalog": spec["catalog"],
            "segment": prof.segment, "archetype": prof.archetype, "is_auto": ev.is_auto, "is_retry": ev.retry,
            "status": self.labels[ev.status], "status_role": ev.status, "channel": ev.channel, "method": ev.method,
            "product_id": product["id"], "product_amount": float(product["amount"]),
            "product_duration_days": product.get("duration_days"),
            "offer_made": ev.intervened, "offer_channel": ev.offer_channel if ev.intervened else None,
            "offer_amount": float(ev.offer_product["amount"]) if ev.intervened else None,
            "offer_accepted": ev.accepted if ev.intervened else None,
            "request_at": ev.t_request, "trigger_at": ev.trigger_at, "response_at": ev.response_at,
            "confirmed_at": ev.confirmed_at, "threshold": float(self.segments[prof.segment]["threshold"][usage]),
        }
        row = dict(entity)
        self._run_emit(self._event_emit, env, row)
        return row
