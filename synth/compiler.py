"""Compile one scenario's behaviour: variable definitions + scenario  ->  a verified ``GenerationSpec``.

Why a language model, and why it is safe
----------------------------------------
Industry standards and curated definitions describe columns; they do not say how columns behave together for a
given scenario. That is judgement a model can make from the column descriptions and the scenario text - for any
industry, with no per-industry code or data. But a model's judgement must never reach the dataset unchecked, so:

1. the model never writes rows; it writes a *spec* in a closed, validated vocabulary (``synth.spec`` /
   ``synth.samplers`` / ``synth.expr``) - data, not code;
2. the spec is layered over the *baseline* (``synth.baseline``): every column it does not mention keeps drawing
   from its own definition, and nothing it writes can widen what a definition allows - the definition-derived
   contract (allowed values, ranges, patterns, nullability: ``synth.contract``) is checked on the simulated rows;
3. the spec is simulated and scored before it is accepted: structure, runtime errors, the contract, the model's own
   invariants and targets, placeholder tokens and impossible future timestamps. Problems go back to the model
   (a bounded number of repair rounds); a spec that still fails is rejected, never used;
4. an accepted spec is stored and pinned to the scenario, so the same seed and ``asOf`` always reproduce the same
   dataset - the model is consulted when a scenario is defined, not every time data is generated.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import logging
import math
import re
import statistics
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from synth.baseline import Baseline, SpecBuildError, build, order_entries, spec_dtype
from synth.clock import RunContext, get_tz
from synth.contract import build_contract
from synth.engines import get_engine
from synth import expectations as expect
from synth.expr import Expr
from synth.projection import parse_columns, project, style_of, time_resolution
from synth.samplers import is_rollup, sampler_names
from synth.scorer import score_rows
from synth.spec import EVAL_VARS, SPEC_VERSION, Emit, GenerationSpec, Invariant, SpecColumn, Target, safe_identifier

logger = logging.getLogger(__name__)

COMPILER_VERSION = 7
MAX_REPAIR_ROUNDS = 2
MAX_REFINE_ROUNDS = 6
DRAFT_REPAIRED_WARNINGS = frozenset({"shared_identifier", "near_constant_numeric_column", "mirrored_columns", "part_exceeds_bound",
                                     "near_constant_state", "entity_stamp_inside_history", "expected_state_share"})
DRAFT_REPAIR_ROUNDS = 1                  # patch rounds for measured errors before a spec is first served (the refinement does the rest)
REJECTED_PATCH_ALLOWANCE = 2
GOOD_ENOUGH_WEIGHT = 1                   # refinement stops once at most one minor finding is left
SCENARIO_KEYS = ("industry", "domain", "country", "use_case", "scenario_type", "business_scenario", "business_response",
                 "expected_outcome", "type_of_data", "entity_key")
REVIEW_ENTITIES = 5
REVIEW_EVENTS = 8
VERIFY_ENTITIES = 60
VERIFY_ROWS_PER_ENTITY = 6
VERIFY_MAX_EVENTS = 12
MODEL_ATTEMPTS = 3                       # one model call is tried this often when it fails for a reason that may pass
MODEL_RETRY_PAUSE = (2.0, 6.0)
_VERIFY_AS_OF = "2026-01-01T00:00:00+00:00"
_EVAL_HINT = " (invariants and targets can read only the supplied column ids, your latents, as_of, REF and P - not event_at, event_index, prev or first_event_at; use a column that records what you need)"
_PLACEHOLDER = re.compile(r"^(SYN|EVENT)_[A-Z0-9_]+_\d{3,6}$")

SYSTEM_PROMPT = """You design how the synthetic data of ONE business scenario is generated.
A deterministic simulator executes what you write. You never write data rows. You write a JSON behaviour spec that makes
the generated columns agree with each other and with the scenario, the way the records of a real system would.

WHAT YOU RECEIVE
- scenario: industry, domain, country, use case, scenario type, business scenario text, data type and entity key. The scenario type is a hard signal: a different scenario type must change
  which outcomes happen and which facts exist (an offer nobody answers has no response time; a suppressed contact has no
  delivery time). Never write behaviour that ignores it.
- timestamp_precision: delivered timestamps are rounded to this unit, so two moments less than one unit apart read as equal.
  Where the order of two events matters, model a delay of at least one unit.
- events_per_entity: how many events every entity has in the delivered data (history.days must fit exactly that many).
- source_resources (when present): what the industry standard says each resource is, so a column is read in the meaning of the
  resource it belongs to (a "history" resource describes earlier actions on the main resource; it is not an unrelated record).
- rollup_candidates (when present): entity columns whose names suggest they summarise an event column (an average, total, count,
  last or first of something each row records). A hint matched on words only: decide from the descriptions, and where it is
  right compute the column with a rollup.
- columns: the exact, final list of columns of the dataset. Each card carries the definition its owner gave it (an
  industry standard, a curator or an uploaded CSV): dtype, description, allowed values, range, pattern, id format,
  nullability, formula, dependencies. THESE DEFINITIONS ARE A CONTRACT checked automatically on the simulated rows: a value
  outside "allowed", outside the range, not matching the pattern, or empty in a column marked non-nullable is rejected.
- expectations (when present): acceptance criteria written independently of you, by someone who saw only the scenario and the
  column definitions. "entity_facts" stay the same on every event of an entity (give them entity scope); "constants" are the only
  columns allowed one single value on every row; "rules" are relations every row obeys (present_iff: the column has a value
  exactly when the condition holds; order: later is not before earlier; at_most: a quantity does not exceed its bound;
  determined_by: the column is a function of those columns, so draw the item once and read it; holds: the condition is true on
  every row; separate: two different facts, never copies, relabellings or exact complements); "state_shares" are the expected
  share bands of outcome values for this scenario type. They are measured on the simulated rows and a break is sent back to you:
  build the behaviour so that they hold; do not copy them into invariants.
- Each column is, by default, drawn independently from its own definition (entity columns once per entity, event columns
  once per row). You override only what needs behaviour.

WHAT YOU DECIDE
1. Scope. "entity" = a property of the customer/account/subject that never changes over its history; "event" = changes
   every row. Correct the scope where the default is wrong.
2. The clock. Which datetime column is the time at which each row's event happened ("clock"). Every other timestamp that
   follows from another event is drawn relative to it with a realistic delay.
3. Outcomes through hidden drivers. When several columns must agree (status, flags, timestamps, amounts, reasons), draw ONE
   hidden "latent" decision first (for example whether the customer responded, whether the payment failed) and derive every
   visible column from it. Put the probabilities in "model" so they are visible and tunable, and choose them for THIS
   scenario type. Do not draw the correlated columns independently.
4. Presence. A fact that only exists in some situations is empty (null) otherwise: use "when". A fact that applies must be
   present: never leave a column empty when the situation has it, and never fill a column that cannot exist (a completion
   time for something that did not complete). Columns marked non-nullable may not be empty at all - model the situation so
   they always have a meaningful value.
5. Causality and arithmetic. A timestamp that happens because of another happens after it. Amounts, balances, totals,
   durations, counts and ratios that are related are computed from each other, not drawn separately. States and the
   timestamps/flags that imply them agree. Windows (validity, cooldown, suppression, dedup, eligibility) start from the
   event that grants them, and a window that forbids the entity's NEXT event must really hold: enforce it with
   history.earliest_next (an expression over prev, e.g. "prev['window_end']") and keep history.min_gap_minutes shorter than
   the windows it must not hide; then prove it with an entity-level invariant (all_after). A permission or eligibility flag
   (consent, do-not-disturb, opt-out, blocked, ineligible) and the action it governs agree - an action that the flag forbids
   does not happen - unless the scenario is explicitly about violations. A quantity measured against a limit, capacity or
   size (consumed vs allowance, paid vs due, used vs balance) does not exceed it unless the column means an overage. Facts
   about an entity stay the same across its rows; a measure that is defined by other columns (the time until something runs out =
   what remains / the rate of use; a total = its parts; a ratio, rate or difference) is an expression of those columns with
   matching units (state each unit in "assumptions" and convert), never a separate draw, and the magnitude of the result must be
   realistic for the domain (a time until depletion of an hour on every row, or an average of 0, shows the units do not match); a recommendation, upgrade or "better fit" is better than what the entity has now in
   the dimension that matters (larger, higher tier, longer, cheaper per unit), never the same or smaller; the state or flag of
   a related subject (usage, order, provisioning, a benefit being active) follows the outcome of the transaction that causes
   it: nothing is activated, allocated or finished for a transaction that failed, was cancelled or is still in flight. Facts
   about an entity stay the same across its rows; a fact that carries over from the previous row uses prev. A status that
   means "still in flight" (created, pending, initialised, in progress) is only possible for an event recent enough that it
   could still be in flight; older events have reached a final state. A boolean that says a step happened (presented, sent,
   viewed, accepted, converted, resolved) agrees with everything that records that step: its timestamp, amount and status exist
   exactly when the flag is true, and a later step (accepted, converted) is never true when the step before it (presented,
   offered) is false. An automatic, scheduled or system-initiated flag agrees with who initiates the row (its requestor, role
   and channel): an automatic action is requested by the system, a customer-initiated one is not. An automatic or recurring action
   also uses a means that can run unattended (a stored mandate, card, wallet, bank account), never a physical or assisted one (cash,
   counter, in person, agent). People and points of sale that act (agents, retailers, staff, devices) are many: their identifiers
   vary across entities; only an automated or system actor has one shared identifier.
6. Realistic values. Where a definition lists no values (marked needs_values), derive realistic values for the scenario's
   industry, country and currency from the column's description and the other columns (names, reasons, plan or product
   names, channels, categories, amounts in local currency and realistic magnitudes) and put them in "reference". Identifiers and contact values follow the real format of the scenario's country
   unless the definition gives one (phone numbers in international E.164 form with the leading "+", for example), and the
   same kind of value always has the same format. Never use
   placeholders such as "Type A", "Sample", "Test", "Product_1", "SYN_..." or "N/A". Values a definition lists are the ONLY
   allowed values; choose among them, weighted realistically, never invent others.
7. Time. event_at, first_event_at and as_of are timezone-aware datetimes. Rows are in the past relative to as_of. A fact
   that records something that already happened (a response, a completion, a delivery, a confirmation) must not be later than
   as_of: bound the delay (for example seconds [lo, "min(hi, secs(as_of, event_at))"]) or make the fact absent. Planned or
   future facts (a validity end, a next eligible time, a due date) may be after as_of; list those column ids in "future_ok".
   Set history.hour_weights (24 weights, local hour) to the realistic daily pattern of what the clock records: customer-facing
   messages and customer-initiated actions are mostly daytime and evening, batch or system jobs may run at night.
8. One fact, one value. Every column in the list is delivered - none can be removed - so columns that record the same fact
   (the same identifier under two names, the same amount, state or flag in two places, a record and the history entry that
   describes it) must agree: draw the fact once (a latent or the first column) and derive the others from it, or make the
   difference real (an earlier event, a different party). Columns that look unrelated to the scenario are still modelled
   realistically for it; they are never left to contradict the columns that matter.
9. Self-checks. Write at least one invariant for every causal, presence, state/flag, arithmetic and window rule you modelled
   (ordering of timestamps, state/flag agreement, presence rules, windows as entity-level invariants) - a rule without an
   invariant is not verified - and targets for the shares/medians your parameters imply for this scenario type. They are
   evaluated on the simulated rows; a failing one is sent back to you.
10. Cadence. An entity has events_per_entity events inside history.days, so the average gap between its events is
   history.days / events_per_entity. Choose history.days from the real rhythm of what the rows record (a recharge, renewal,
   billing, claim, visit or alert cycle) so the gaps match it. A window an event opens (validity, term, cooldown, coverage) ends
   before the same entity's next event of that kind starts, unless the scenario says it is renewed early. Never squeeze the
   events into a history far shorter than events_per_entity x the typical gap.
11. Entity facts that summarise events. A fact about the entity that is defined by its events (first/last/latest/total/count/
   average of something, "has ever ...", the latest response or outcome, a current status that results from the last event) is
   never drawn on its own: compute it with a rollup (see the samplers) or derive both it and the events from one entity-level
   latent. The entity fact and the events it summarises always agree. The names give it away: avg_/average_/typical_, total_/sum_,
   count_/number_of_, last_/latest_/current_, first_/earliest_, max_/min_ placed on an entity column next to an event column
   that records the same thing (rollup_candidates lists them). A yes/no or status about the entity as a whole (assistance
   required, issue resolved, eligible, response) either comes from a rollup of its events (any, all, last) or the events depend
   on it: an entity flagged as needing no assistance has no assistance events, and an entity whose issue is resolved does not
   have a latest unresolved event.
12. Items, parties and variety. The attributes of one product, plan, pack, offer or tariff (name, price, size, validity, type)
   are properties of ONE catalogue item: draw the item once into a latent (weights in "model"/"reference") and read every
   attribute from it with a "switch" on that latent or a REF number table indexed by it (REF['price'][lat_item]); never draw
   a name, a price and a validity separately. A column that identifies ANOTHER record (a parent, a referenced resource, a
   counterpart) holds that other record's identifier, never this record's own id. Who acted (role, party identifier, channel)
   agree: an agent or retailer role carries an agent's identifier, a self-service channel has the subscriber as the actor.
   Statuses of the same subject (account, customer, subscription) and their reasons agree with each other. Numeric facts about
   entities (usage, balances, scores, counts) vary continuously from entity to entity: use a distribution whose parameters a
   persona or segment may shift; do not give every entity one of two or three fixed sets of values. A completion, confirmation or
   settlement time, and every amount, window or reference that only exists once something succeeded, is absent when it failed, was
   cancelled, was rejected, expired or is still in flight.
13. Steps, snapshots and independent measures. The timestamps of different steps of one process (requested, confirmed,
   notified, shown, answered, converted) are different moments: each is drawn from the step before it with its own delay and is
   present only when that step happened; none is a single value kept for the entity's whole history, and a "last/latest X"
   moment read at an event is derived from the entity's earlier events (prev) and is never later than that event. Two
   dates of one entity (a profile date, a validity start) differ unless they really are one record. Two scores, ratings or
   measures about one subject (a credit score and a credit risk, an affinity and a propensity) are separate facts: each has its
   own latent and variation, never a constant minus the other; a label derived from a number (a frequency band from a count) is
   computed from it, and counts over nested windows (30 days inside 90 days) are not fixed multiples of each other. The status of
   an event keeps a realistic spread of the stages it can reach and does not repeat the outcome that another column records
   (a status "completed_discrepancy" beside a verification result), nor stays one value on nearly every row. A recommended
   action agrees with the flag that governs it (a retry action only where the flag says retryable) and with the cause of
   the failure. A remainder, reserved or used quantity of a record follows what happened (usage, time), not a fixed share of
   the amount. The kind of what an item delivers (data, voice, money) follows the item.

THE LANGUAGE
Expressions (python-like, safe subset). Variables: any column id or latent you declare (order is resolved automatically;
no cycles); event_at (this row's time), event_index (0-based position in the entity's history), first_event_at, as_of;
prev (the previous row of the same entity as a dict: prev['column'], or None on the first row); P['name'] (your "model"
parameters); REF['table'] (your "reference" tables).
Operators: + - * /  == != < <= > >=  and or not  is None / is not None  in / not in  (a if cond else b)  [lists]  x[key].
Functions: add_days(dt,n) add_secs(dt,n) secs(a,b)=seconds(a-b) days(a,b) text(x) lower(x) upper(x) int(x) float(x) abs min max
round(x,n) len sigmoid(x) matches(regex,str)=whole-string regex match is_none(x) hour_of(dt,offset_minutes).
Invariants and targets read the column ids, as_of, REF[...] and P[...] only (not event_at, event_index or prev).
Entity-level invariants only (every column becomes a LIST over the entity's rows, oldest first): distinct(list)
non_decreasing(list) strictly_increasing(list) min_gap_secs(list) and all_after(starts, ends[, applies]) = every row starts at or
after the PREVIOUS row's end. starts and ends are the columns themselves (lists; an empty end is skipped); the optional third
argument is a list of booleans, one per row (for example a column that is true when the row's window exists), never a single
boolean: "all_after(requested_at, valid_until)" is right, "all_after(a, b, x is not None)" is wrong because x is a list.
NOT available: ** % //  attribute access (x.y)  comprehensions  lambda  f-strings  string methods  imports  keyword args.
Guard None before arithmetic or comparison: "x is None or x > y". An undefined name is an error.

Samplers - an emit gives either "expr" or "sample":
{"type":"const","value":V}
{"type":"expr","expr":"..."}
{"type":"choice","weights":{"label":w,...}}                      labels are strings; add "cast":"int"|"float" for numbers
{"type":"choice","from_ref":"table"}                             weights from REF['table'] (a {label: weight} mapping)
{"type":"choice","by":"<expr>","tables":{"key":{"label":w},...}} weights chosen by another value
{"type":"bernoulli","p":0.3}                                     p: number or an expression, e.g. "P['p_fail']"
{"type":"uniform","low":a,"high":b,"round":2}                    a, b: number or expression
{"type":"uniform_int","low":a,"high":b}
{"type":"lognormal","median":m,"sigma":s,"min":lo,"max":hi}     skewed positive amounts, delays; add "cast":"int" for integers
{"type":"time_shift","anchor":"<column>","direction":"before"|"after","seconds":[lo,hi]}   also "minutes":{lognormal spec} "days":{lognormal spec}
{"type":"id","prefix":"AB","digits":8,"registry":"name"}        unique ids (use the id format the definition gives)
{"type":"template","parts":[{"const":"TX-"},{"digits":6}],"registry":"name"}
{"type":"switch","by":"<expr>","cases":{"key":<sampler>,...}}   a different sampler per value of another column
{"type":"rollup","fn":"first|last|min|max|sum|mean|count|any|all|distinct","of":"<expr on one event row>","where":"<expr>|null"}
   an ENTITY column (scope entity) computed from the entity's own events once they exist: "of" and "where" read the event
   columns of each row; count and any may omit "of". It is the way to state a fact about the entity that its events decide
   (for example fn "last", of "closed_at", where "closed_at is not None"). Only another rollup may read a rollup.
A sampler may carry "cast". Probabilities, thresholds, delays that matter belong in "model" and are read as P['name'].

OUTPUT: one JSON object, nothing else:
{
 "timezone": "<IANA zone of the scenario's country>", "currency": "<ISO 4217>",
 "assumptions": ["each modelling assumption in one sentence, including every invented value list and parameter"],
 "history": {"days": 120, "min_gap_minutes": 60, "margin_minutes": 20, "earliest_next": null, "hour_weights": null},
 "clock": "<datetime column id>",
 "model": {"name": number-or-text, ...},
 "reference": {"table": {"label": weight, ...}},
 "latents": [ {"name":"lat_x","dtype":"boolean|string|integer|float|datetime","scope":"entity|event","when":null,"sample":{...}} ],
 "columns": [ {"name":"<column id>","scope":"entity|event","when":"<expr>|null","sample":{...}} or {"name":..,"expr":".."} ],
 "future_ok": ["<column id>"],
 "invariants": [ {"id":"snake_id","level":"record|entity","expr":"...","message":"..."} ],
 "targets": [ {"id":"snake_id","stat":"share|mean|median","condition":"<expr>","column":"<id for mean/median>","where":"<expr>|null","min":0.0,"max":1.0,"description":"..."} ]
}
history.earliest_next (optional) is an expression over prev and as_of giving the earliest time the next event of the same
entity may happen (for example the end of a cooldown: "prev['cooldown_end']"). hour_weights (optional) is 24 relative weights
for the local hour of day. Use only the supplied column ids and your own latents. Mention only columns that need behaviour.

A TINY EXAMPLE OF THE FORMAT ONLY (an unrelated delivery scenario; never copy its names):
{"timezone":"Europe/Paris","currency":"EUR","assumptions":["Late delivery rate is 12% for this scenario."],
 "history":{"days":90,"min_gap_minutes":120,"margin_minutes":20},"clock":"ordered_at",
 "model":{"p_late":0.12},"latents":[{"name":"lat_late","dtype":"boolean","sample":{"type":"bernoulli","p":"P['p_late']"}}],
 "columns":[{"name":"promised_at","sample":{"type":"time_shift","anchor":"ordered_at","direction":"after","seconds":[172800,259200]}},
  {"name":"delivered_at","when":"secs(as_of, ordered_at) > 345600","sample":{"type":"time_shift","anchor":"promised_at","direction":"after","seconds":["3600 if lat_late else -86400","172800 if lat_late else -3600"]}}],
 "invariants":[{"id":"delivery_after_order","expr":"delivered_at is None or delivered_at > ordered_at","message":"Delivery follows the order."}],
 "targets":[{"id":"late_share","condition":"delivered_at is not None and delivered_at > promised_at","where":"delivered_at is not None","min":0.05,"max":0.2}]}
"""


# ------------------------------------------------------------------------------------------------------------------
@dataclass
class CompileResult:
    spec: GenerationSpec | None
    status: str                                   # compiled | rejected | unavailable
    problems: list[str] = field(default_factory=list)
    rounds: int = 0
    overlay: dict[str, Any] | None = None
    report: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)   # what the checks and the reviewer still consider wrong or implausible


def _short(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit - 1] + "…"


DEFAULT_EVENTS = 10
MAX_DESIGN_EVENTS = 50


def events_per_entity(brief: dict[str, Any]) -> int:
    """Events per entity the dataset will have (the scenario's records per user), which the history is designed to fit."""
    try:
        n = int(brief.get("events_per_entity") or DEFAULT_EVENTS)
    except (TypeError, ValueError):
        n = DEFAULT_EVENTS
    return 1 if str(brief.get("type_of_data") or "").lower() == "aggregational" else max(1, min(MAX_DESIGN_EVENTS, n))


def column_cards(base: Baseline, notes: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Compact, model-facing description of every column, taken from its own definition (and its source standard)."""
    cards: list[dict[str, Any]] = []
    for cid, col in base.spec.columns.items():
        v = base.variables[cid]
        params = v.get("params") if isinstance(v.get("params"), dict) else {}
        card: dict[str, Any] = {"id": cid, "dtype": col.dtype, "scope": col.kind}
        if col.column and col.column != cid:
            card["column_name"] = col.column
        if col.description:
            card["description"] = col.description
        source = str(v.get("source") or "").upper()
        if source:
            card["source"] = source
        prov = v.get("provenance") if isinstance(v.get("provenance"), dict) else {}
        model = prov.get("source_json_model") or prov.get("model")
        if model:
            card["resource"] = str(model)
        allowed = params.get("choices") if isinstance(params.get("choices"), list) else None
        if allowed:
            card["allowed"] = allowed[:40] + ([f"... ({len(allowed)} in total)"] if len(allowed) > 40 else [])
        for key in ("min", "max", "pattern", "min_length", "max_length", "prefix", "digits", "precision", "timestamp_format"):
            if params.get(key) is not None:
                card[key] = params[key]
        if v.get("formula"):
            card["formula"] = str(v["formula"])
        if v.get("depends_on"):
            card["depends_on"] = list(v["depends_on"])
        card["nullable"] = bool(v.get("nullable"))
        if col.placeholder:
            card["needs_values"] = True
        note = (notes or {}).get(str(col.column or cid))
        if note:
            card.update(note)
        cards.append(card)
    return cards


_AGGREGATES = {"avg": "mean", "average": "mean", "mean": "mean", "total": "sum", "sum": "sum", "count": "count",
               "number": "count", "last": "last", "latest": "last", "first": "first", "earliest": "first",
               "max": "max", "maximum": "max", "highest": "max", "min": "min", "minimum": "min", "lowest": "min"}
_GENERIC_TOKENS = frozenset({"id", "date", "time", "timestamp", "value", "amount", "flag", "name", "status", "type", "of", "to", "the",
                             "in", "for", "per", "is", "at", "by", "on", "and", "last", "days", "day", "total", "number", "count"})


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def rollup_candidates(base: Baseline) -> list[dict[str, Any]]:
    """Entity columns whose names say they summarise an event column (avg_x beside x, last_y beside a y of each row).

    Only a hint for the author, who decides from the descriptions whether a rollup is right: the columns are matched by the
    words of their names, nothing about the domain is assumed.
    """
    columns = base.spec.columns
    events = {cid: _tokens(cid) - _GENERIC_TOKENS for cid, col in columns.items() if col.kind == "event" and col.column}
    out = []
    for cid, col in columns.items():
        if col.kind != "entity" or not col.column or cid == base.spec.entity_column:
            continue
        parts = [w for w in re.split(r"[^a-z0-9]+", cid.lower()) if w]
        found = [(i, _AGGREGATES[w]) for i, w in enumerate(parts) if w in _AGGREGATES
                 and not (i + 1 < len(parts) and parts[i + 1].isdigit())]          # "last_30_days" is a period, not "the last one"
        if not found:
            continue
        fn = found[0][1]
        own = _tokens(cid) - _GENERIC_TOKENS - set(_AGGREGATES)
        scored = sorted(((len(own & t) / len(own | t), y) for y, t in events.items() if own and own & t), reverse=True)
        scored = [(j, y) for j, y in scored if j >= 0.3]
        if scored:
            out.append({"column": cid, "looks_like": fn, "of_event_columns": [y for _, y in scored[:3]]})
    return out[:12]


def build_prompt(brief: dict[str, Any], base: Baseline, notes: dict[str, dict[str, Any]] | None = None,
                 resources: dict[str, str] | None = None, expectations: dict[str, Any] | None = None) -> str:
    scenario = {k: v for k, v in brief.items() if v not in (None, "") and k in SCENARIO_KEYS}
    spec = base.spec
    context: dict[str, Any] = {
        "scenario": scenario,
        "entity_column": spec.entity_column,
        "default_clock": spec.clock,
        "timestamp_precision": "minute" if time_resolution(spec) == 60 else "second",
        "events_per_entity": events_per_entity(brief),
    }
    if resources:
        context["source_resources"] = resources
    candidates = rollup_candidates(base)
    if candidates:
        context["rollup_candidates"] = candidates
    rendered = expect.render(expectations)
    if rendered:
        context["expectations"] = rendered
    context["columns"] = column_cards(base, notes)
    return ("Design the behaviour spec for this scenario and these columns.\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":")))


def repair_prompt(first: str, overlay: dict[str, Any], problems: list[str]) -> str:
    return (first + "\n\nYOUR CURRENT SPEC:\n" + json.dumps({k: v for k, v in overlay.items() if k not in ("healed", "expectations")}, ensure_ascii=False, separators=(",", ":"))
            + "\n\nIT WAS SIMULATED AND CHECKED. THESE PROBLEMS MUST BE FIXED (the first ones are the most serious):\n- "
            + "\n- ".join(problems[:14])
            + "\n\nFix the root cause in the behaviour; do not remove an invariant or target merely to make it pass unless it is itself wrong. "
              "Return ONLY THE CHANGES as one JSON object, a patch to your current spec; everything you leave out stays exactly as it is:\n"
              ' - "columns", "latents", "invariants", "targets": entries to add or replace - a column or latent is replaced as a whole '
              '(identified by "name"), an invariant or target as a whole (identified by "id");\n'
              ' - "model", "reference", "history": the keys to set (a reference table is replaced as a whole);\n'
              ' - "timezone", "currency", "clock", "assumptions", "future_ok": their new values (assumptions replace the old list: keep the true ones);\n'
              ' - "remove": {"columns": [names whose behaviour override should be dropped], "latents": [...], "invariants": [ids], '
              '"targets": [ids], "model": [keys], "reference": [keys]}.')


_KEYED = {"columns": "name", "latents": "name", "invariants": "id", "targets": "id"}
_MERGED = ("model", "reference", "history")
_REPLACED = ("timezone", "currency", "clock", "assumptions", "future_ok")


def apply_patch(overlay: dict[str, Any], patch: Any) -> dict[str, Any]:
    """``overlay`` with the changes of ``patch`` applied (see ``repair_prompt``); the input is not modified.

    A reply that restates the whole spec instead of a patch is handled the same way: every entry it carries replaces
    the entry of the same name or id, and what it leaves out is kept.
    """
    out = json.loads(json.dumps(overlay, default=str))
    if not isinstance(patch, dict):
        return out
    ignored: list[str] = []
    for section, key in _KEYED.items():
        current = _entries(out.get(section), section, ignored)
        index = {str(e.get("name") if key == "name" else e.get("id") or e.get("name") or ""): i for i, e in enumerate(current)}
        for item in _entries(patch.get(section), section, ignored):
            ident = str(item.get(key) or item.get("column") or item.get("name") or "")
            if ident in index:
                current[index[ident]] = item
            else:
                index[ident] = len(current)
                current.append(item)
        out[section] = current
    for section in _MERGED:
        incoming = patch.get(section)
        if isinstance(incoming, dict):
            base = out.get(section) if isinstance(out.get(section), dict) else {}
            out[section] = {**base, **incoming}
    for section in _REPLACED:
        if section in patch and patch[section] not in (None, ""):
            out[section] = patch[section]
    removals = patch.get("remove") if isinstance(patch.get("remove"), dict) else {}
    for section, key in _KEYED.items():
        gone = {str(x) for x in removals.get(section) or [] if x}
        if gone:
            out[section] = [e for e in _entries(out.get(section), section, ignored)
                            if str(e.get(key) or e.get("column") or e.get("name") or "") not in gone]
    for section in ("model", "reference"):
        for name in removals.get(section) or []:
            if isinstance(out.get(section), dict):
                out[section].pop(str(name), None)
    return out


# ------------------------------------------------------------------------------------------------------------------
def _entries(raw: Any, what: str, errors: list[str]) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if isinstance(raw, dict):
        raw = [{"name": k, **(v if isinstance(v, dict) else {"expr": v})} for k, v in raw.items()]
    if not isinstance(raw, list):
        errors.append(f"'{what}' must be a list")
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        else:
            errors.append(f"'{what}' entries must be objects, got {_short(item, 60)}")
    return out


def _entry(item: dict[str, Any], default_scope: str, errors: list[str]) -> dict[str, Any] | None:
    name = str(item.get("name") or item.get("column") or "").strip()
    scope = str(item.get("scope") or default_scope).strip().lower()
    if is_rollup(item.get("sample")):
        scope = "entity"                     # a rollup is a fact about the entity, whatever scope the author wrote
    if scope not in {"entity", "event"}:
        errors.append(f"column '{name}': scope must be entity or event")
        return None
    sample, expr = item.get("sample"), item.get("expr")
    if isinstance(sample, dict) and sample.get("type") == "expr" and expr is None:
        expr, sample = sample.get("expr"), None
    when = item.get("when")
    entry = {"column": name, "scope": scope, "when": str(when) if when not in (None, "") else None,
             "expr": str(expr) if expr not in (None, "") else None, "sample": sample if isinstance(sample, dict) else None}
    try:
        Emit.model_validate(entry)
    except (ValidationError, ValueError) as exc:
        errors.append(f"column '{name}': {_first_error(exc)}")
        return None
    return entry


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        err = exc.errors()[0]
        return f"{err.get('msg', 'invalid')} ({'.'.join(str(p) for p in err.get('loc', ()))})".replace("Value error, ", "")
    return str(exc).replace("Value error, ", "")


def merge(base: Baseline, overlay: dict[str, Any], *, brief: dict[str, Any]) -> tuple[GenerationSpec | None, list[str]]:
    """Layer a model-written overlay over the baseline. Returns ``(spec, errors)``; ``spec`` is None when errors remain."""
    errors: list[str] = []
    if not isinstance(overlay, dict):
        return None, ["the spec must be one JSON object"]
    columns = {cid: col.model_copy() for cid, col in base.spec.columns.items()}
    entries: dict[str, dict[str, Any]] = {e.column: e.model_dump() for e in base.spec.emit}
    taken = set(columns)

    # latents
    for item in _entries(overlay.get("latents"), "latents", errors):
        name = str(item.get("name") or "").strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) or safe_identifier(name, set()) != name:
            errors.append(f"latent '{name}': needs a plain identifier that is not a reserved word or function name")
            continue
        if name in taken:
            errors.append(f"latent '{name}' collides with a column id; choose another name")
            continue
        dtype = str(item.get("dtype") or "string").lower()
        entry = _entry({**item, "name": name}, "event", errors)
        if entry is None:
            continue
        columns[name] = SpecColumn(kind="latent", dtype=spec_dtype(dtype) if spec_dtype(dtype) in {"string", "integer", "float", "boolean", "datetime", "date", "categorical"} else "string",
                                   description=str(item.get("description") or "")[:200], origin="llm")
        entries[name] = entry
        taken.add(name)

    # column overrides
    for item in _entries(overlay.get("columns"), "columns", errors):
        name = str(item.get("name") or item.get("column") or "").strip()
        if name not in columns or columns[name].kind == "latent":
            errors.append(f"column '{name}' is not one of the supplied columns (use only the supplied column ids; declare hidden drivers under 'latents')")
            continue
        entry = _entry(item, columns[name].kind if columns[name].kind != "latent" else "event", errors)
        if entry is None:
            continue
        if name == base.spec.entity_column and entry["scope"] != "entity":
            errors.append(f"column '{name}' is the entity key and must stay entity-scoped")
            continue
        entries[name] = entry
        columns[name].kind = entry["scope"]
        columns[name].origin = "llm"
        if columns[name].placeholder:
            columns[name].placeholder = False

    # clock
    clock = overlay.get("clock") or base.spec.clock
    clock = str(clock) if clock else None
    if clock is not None:
        if clock not in columns or columns[clock].kind == "latent" or columns[clock].dtype != "datetime":
            errors.append(f"clock '{clock}' must be a supplied datetime column")
            clock = None
        else:
            explicit = any(str(i.get("name")) == clock for i in _entries(overlay.get("columns"), "columns", []))
            if not explicit:
                entries[clock] = {"column": clock, "scope": "event", "when": None, "expr": "event_at", "sample": None}
                columns[clock].kind = "event"
    old = base.spec.clock
    if old and old != clock and old in entries and entries[old].get("expr") == "event_at" and old in base.definition_entries:
        entries[old] = dict(base.definition_entries[old])      # the former default clock reverts to its own definition

    # timezone / currency / timeline / parameters
    timezone = str(overlay.get("timezone") or base.spec.timezone or "UTC")
    try:
        get_tz(timezone)
    except ValueError:
        errors.append(f"timezone '{timezone}' is not an IANA zone name")
        timezone = "UTC"
    history = overlay.get("history") if isinstance(overlay.get("history"), dict) else {}
    timeline = dict(base.spec.timeline)
    timeline["events"] = events_per_entity(brief)
    try:
        timeline["history_days"] = min(730.0, max(7.0, float(history.get("days", timeline["history_days"]))))
        timeline["min_gap_minutes"] = min(1440.0, max(0.0, float(history.get("min_gap_minutes", timeline["min_gap_minutes"]))))
        if history.get("margin_minutes") is not None:
            timeline["margin_minutes"] = min(100000.0, max(0.0, float(history["margin_minutes"])))
    except (TypeError, ValueError):
        errors.append("history.days / min_gap_minutes / margin_minutes must be numbers")
    if history.get("earliest_next"):
        timeline["earliest_next"] = str(history["earliest_next"])
    weights = history.get("hour_weights")
    if weights:
        if isinstance(weights, list) and len(weights) == 24 and all(isinstance(w, (int, float)) and w >= 0 for w in weights) and sum(weights) > 0:
            timeline["hour_weights"] = [float(w) for w in weights]
        else:
            errors.append("history.hour_weights must be 24 non-negative numbers")
    model = overlay.get("model") if isinstance(overlay.get("model"), dict) else {}
    reference = overlay.get("reference") if isinstance(overlay.get("reference"), dict) else {}
    for key, table in reference.items():
        if not isinstance(table, dict) or not table or any(isinstance(w, (dict, list)) or not isinstance(w, (int, float)) or w < 0 for w in table.values()) \
                or sum(float(w) for w in table.values()) <= 0:
            errors.append(f"reference '{key}' must be a non-empty {{label: non-negative weight}} mapping")

    ordered: list[dict[str, Any]] = []
    try:
        ordered = order_entries(list(entries.values()))
    except SpecBuildError as exc:
        errors.append(str(exc))

    invariants, targets = [], []
    known = set(columns) | EVAL_VARS
    for i, item in enumerate(_entries(overlay.get("invariants"), "invariants", errors)):
        try:
            inv = Invariant.model_validate({"id": str(item.get("id") or f"invariant_{i + 1}"), "level": str(item.get("level") or "record"),
                                            "severity": str(item.get("severity") or "error"), "expr": str(item.get("expr") or ""),
                                            "message": str(item.get("message") or "")})
        except (ValidationError, ValueError) as exc:
            errors.append(f"invariant '{item.get('id')}': {_first_error(exc)}")
            continue
        if inv.level == "dataset":
            errors.append(f"invariant '{inv.id}': level must be record or entity")
        elif Expr(inv.expr).names - known:
            errors.append(f"invariant '{inv.id}' reads unknown column(s) {sorted(Expr(inv.expr).names - known)}{_EVAL_HINT}")
        else:
            invariants.append(inv)
    for i, item in enumerate(_entries(overlay.get("targets"), "targets", errors)):
        try:
            tgt = Target.model_validate({
                "id": str(item.get("id") or f"target_{i + 1}"), "stat": str(item.get("stat") or "share"),
                "column": item.get("column"), "condition": item.get("condition"), "where": item.get("where") or None,
                "min": float(item.get("min")), "max": float(item.get("max")), "description": str(item.get("description") or "")})
        except (ValidationError, ValueError, TypeError) as exc:
            errors.append(f"target '{item.get('id')}': {_first_error(exc)}")
            continue
        used = {n for src in (tgt.where, tgt.condition) if src for n in Expr(src).names} | ({tgt.column} if tgt.column else set())
        if used - known:
            errors.append(f"target '{tgt.id}' reads unknown column(s) {sorted(used - known)}{_EVAL_HINT}")
        else:
            targets.append(tgt)

    future_ok = [str(n) for n in (overlay.get("future_ok") or []) if str(n) in columns] if isinstance(overlay.get("future_ok"), list) else []
    if errors:
        return None, errors
    try:
        spec = GenerationSpec(
            source="llm", scenario=dict(base.spec.scenario),
            currency=str(overlay.get("currency") or base.spec.currency or "USD")[:8], timezone=timezone,
            entity_column=base.spec.entity_column, clock=clock,
            assumptions=[str(a)[:400] for a in (overlay.get("assumptions") or []) if a][:40],
            columns=columns, emit=ordered, timeline=timeline, model=model, reference=reference,
            invariants=invariants, targets=targets,
            output={"order_by": clock, "future_ok": future_ok} if clock else {"future_ok": future_ok})
    except (ValidationError, ValueError) as exc:
        return None, [_first_error(exc)]
    return spec, []


# ------------------------------------------------------------------------------------------------------------------
def verify(spec: GenerationSpec, variables: list[dict[str, Any]], *, entities: int = VERIFY_ENTITIES,
           per_entity: int | None = None, aggregational: bool = False,
           records_out: list[dict[str, Any]] | None = None,
           expectations: dict[str, Any] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Simulate the spec and return ``(problems, report)``; no problems means the spec may be used.

    The simulation has as many events per entity as the spec was designed for (``timeline.events``, at most
    ``VERIFY_MAX_EVENTS``; the history scales with the count, so spacing is the same), because rules about spacing and
    windows only mean something at the real number of events. ``records_out`` (optional) receives the simulated rows as
    a client would see them, for review; ``report["advice"]`` holds what the checks consider implausible without being
    impossible, and ``report["facts"]`` what a reviewer needs to know about the shape of the data. ``expectations`` (see
    ``synth.expectations``) are measured on the same rows and join the advice.
    """
    if per_entity is None:
        per_entity = max(2, min(VERIFY_MAX_EVENTS, int(spec.timeline.get("events") or VERIFY_ROWS_PER_ENTITY)))
    from synth.engines.rows import SpecRuntimeError

    engine = get_engine(spec.engine)
    ctx = RunContext(seed=11, as_of=_VERIFY_AS_OF, tz_name=spec.timezone)
    try:
        rows = engine.simulate(spec, ctx, entities=entities, per_entity=1 if aggregational else per_entity, hints={})
    except SpecRuntimeError as exc:
        return [f"RUNTIME column '{exc.column}' failed while being drawn: {_short(str(exc.cause), 200)}"], {}
    except Exception as exc:
        return [f"RUNTIME the simulation failed: {_short(f'{type(exc).__name__}: {exc}', 240)}"], {}
    delivered = spec.delivered
    records = project(rows, spec, delivered)
    if records_out is not None:
        records_out.extend(records)
    parsed, parse_errors = parse_columns(records, spec, delivered)
    contract = build_contract(variables, delivered)
    report = score_rows(spec, parsed, engine.reference_view(spec, ctx), columns=set(delivered), parse_errors=parse_errors, contract=contract)
    problems: list[str] = []
    for err in report["structure"]["parse_errors"][:3]:
        problems.append(f"FORMAT column '{err['column']}' produced an unparseable value {err['value']!r}: {err['error']}")
    for v in report["structure"]["contract_violations"]:
        problems.append(f"CONTRACT column '{v['column']}': {v['violations']} value(s) break its definition ({v['rule']}"
                        f"{' ' + _short(v['allowed'], 160) if v['allowed'] is not None and v['rule'] != 'not_null' else ''}); e.g. "
                        f"{_short([e['value'] for e in v['examples']], 100)}"
                        + ("  -> the column is non-nullable: it must always have a value" if v["rule"] == "not_null" else ""))
    for f in report["invariants"]["failed"]:
        ex = f["examples"][0] if f.get("examples") else {}
        problems.append(f"INVARIANT '{f['id']}' failed on {f['violations']} of {f['evaluated']} {f['level']}s ({f['message']}); e.g. {_short(ex, 220)}")
    for t in report["targets"]["results"]:
        if t["status"] == "fail":
            problems.append(f"TARGET '{t['id']}' expects {t['min']}..{t['max']} but the simulation gives {t.get('observed')} (n={t.get('n')}); "
                            "either the behaviour or the expectation is wrong")
    for d in report["structure"]["duplicate_columns"]:
        problems.append(f"DUPLICATE columns {d['columns']} carry the same value on every row; draw the fact once and derive the other column from it, or make the difference real")
    # placeholders and impossible future timestamps
    future_ok = set(spec.output.get("future_ok") or [])
    placeholders: dict[str, str] = {}
    future: dict[str, int] = {}
    for cid in delivered:
        col = spec.columns[cid]
        for r in rows:
            v = r.get(cid)
            if isinstance(v, str) and _PLACEHOLDER.match(v):
                placeholders.setdefault(cid, v)
            elif col.dtype == "datetime" and v is not None and v > ctx.as_of and cid not in future_ok:
                future[cid] = future.get(cid, 0) + 1
    total = len(rows)
    empty_ids: list[str] = []
    if total >= 30:
        for cid, name in delivered.items():
            if all(r.get(cid) is None for r in rows):
                empty_ids.append(cid)
                problems.append(f"EMPTY column '{name}' is empty on every simulated row; a delivered column must carry a value whenever its "
                                "situation exists, and that situation must occur in this scenario (drop the condition that never holds, or "
                                "derive the value from the fact it describes)")
    for cid, example in placeholders.items():
        problems.append(f"PLACEHOLDER column '{cid}' still draws neutral tokens such as '{example}'; give it realistic values for this "
                        "scenario (declare them in 'reference')")
    for cid, n in future.items():
        problems.append(f"FUTURE column '{cid}' has {n} value(s) later than as_of, but it records something that has already happened; "
                        "bound its delay, make it absent when it would be in the future, or list it in 'future_ok' if it is a planned/future fact")
    report["problems"] = len(problems)
    report["empty_columns"] = empty_ids
    report["failed_targets"] = [t["id"] for t in report["targets"]["results"] if t["status"] == "fail"]
    if not problems:
        declared = set((expectations or {}).get("constants") or [])
        report["advice"] = (cadence_advice(spec, rows) + constant_advice(spec, rows, variables, declared) + sparse_advice(spec, rows)
                            + bound_advice(spec, rows) + inflight_advice(spec, rows) + shared_identifier_advice(spec, rows)
                            + mirrored_advice(spec, rows, expectations) + counter_advice(spec, rows) + flag_state_advice(spec, rows)
                            + near_constant_state_advice(spec, rows) + entity_stamp_advice(spec, rows)
                            + expect.check(spec, rows, expectations, ctx.as_of)
                            + expect.constant_columns(spec, rows, expectations, _definition_fixed(variables)))
        report["facts"] = shape_facts(spec, rows)
    return problems, report


# ------------------------------------------------------------------------------------------------------------------
MAX_FACT_COLUMNS = 14


def _by_entity(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The simulated rows grouped by entity, each group oldest event first."""
    if not spec.entity_column or not spec.clock:
        return []
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(spec.entity_column), []).append(r)
    return [sorted(g, key=lambda r: r.get(spec.clock) or datetime.min.replace(tzinfo=timezone.utc)) for g in groups.values()]


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def cadence_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings (same shape as a review's) for windows that consecutive events of one entity open on top of each other.

    A pair of event-scope datetime columns (a, b) is a *window* when b never precedes a and spans a day or more. When the
    window of an event is still open at the next event of the same entity, and windows are longer than the typical gap, the
    history is too short for the events it has to hold (or the windows too long). Measured, not guessed.
    """
    groups = _by_entity(spec, rows)
    if not groups or not spec.clock:
        return []
    cols = [c for c, col in spec.columns.items() if col.kind == "event" and col.dtype == "datetime" and col.column and c != spec.clock]
    gaps = [(g[i + 1][spec.clock] - g[i][spec.clock]).total_seconds() for g in groups for i in range(len(g) - 1)]
    median_gap = _median(gaps)
    if median_gap <= 0:
        return []
    found: list[tuple[float, dict[str, Any]]] = []
    for a in cols:
        for b in cols:
            if a == b:
                continue
            spans = [(r[b] - r[a]).total_seconds() for r in rows if r.get(a) is not None and r.get(b) is not None]
            if len(spans) < 30 or sum(1 for x in spans if x >= 0) < 0.98 * len(spans) or _median(spans) < 86400.0:
                continue
            seen = overlapping = 0
            for g in groups:
                for prev, nxt in zip(g, g[1:]):
                    if prev.get(b) is None or nxt.get(a) is None:
                        continue
                    seen += 1
                    overlapping += nxt[a] < prev[b]
            share = overlapping / seen if seen else 0.0
            window = _median(spans)
            if seen >= 20 and share > 0.25 and window > 0.5 * median_gap:
                names = [spec.columns[a].column, spec.columns[b].column]
                ordered = sorted(spans)
                at_clock = [abs((r[b] - r[spec.clock]).total_seconds()) <= 86400.0 for r in rows if r.get(b) is not None and r.get(spec.clock) is not None]
                found.append((share, {
                    "id": "window_overlap", "severity": "warn", "columns": names, "ids": [a, b], "share": round(share, 3),
                    "window_days_mean": round(statistics.fmean(x for x in spans if x >= 0) / 86400.0, 1),
                    "window_days_p95": round(ordered[int(0.95 * (len(ordered) - 1))] / 86400.0, 1),
                    "ends_at_clock": bool(at_clock) and sum(at_clock) >= 0.95 * len(at_clock),
                    "problem": f"The window {names[0]} -> {names[1]} (typically {window / 86400:.0f} days) is still open at the same entity's next event "
                               f"in {share:.0%} of consecutive events, because events are only {median_gap / 86400:.1f} days apart on average.",
                    "evidence": f"history {spec.timeline.get('history_days')} days for {spec.timeline.get('events')} events per entity; "
                                f"median gap {median_gap / 86400:.1f} d, median window {window / 86400:.0f} d",
                    "fix": "Make the gap between an entity's events match the real cycle of what the rows record: raise history.days (about events_per_entity x the "
                           "typical gap), or shorten the window, or let the next event start after the previous window ends (history.earliest_next)."}))
    found.sort(key=lambda t: -t[0])
    out, used_b = [], set()
    for _, issue in found:
        if issue["columns"][1] not in used_b:
            used_b.add(issue["columns"][1])
            out.append(issue)
    return out[:2]


def _definition_fixed(variables: list[dict[str, Any]]) -> set[str]:
    """Names of the columns whose own definition fixes one value (min == max, or a single allowed value)."""
    fixed: set[str] = set()
    for v in variables:
        params = v.get("params") if isinstance(v.get("params"), dict) else {}
        choices = params.get("choices") if isinstance(params.get("choices"), list) else None
        if (params.get("min") is not None and params.get("min") == params.get("max")) or (choices is not None and len(choices) == 1):
            fixed.add(str(v.get("name")))
    return fixed


def constant_advice(spec: GenerationSpec, rows: list[dict[str, Any]], variables: list[dict[str, Any]],
                    declared: set[str] | None = None) -> list[dict[str, Any]]:
    """Findings for numeric and yes/no columns that carry one single value on every row: a measure that never varies measures nothing.

    A column the definition fixes, or that the expectations declare constant, is left alone.
    """
    n = len(rows)
    if n < 100:
        return []
    fixed = _definition_fixed(variables)
    out = []
    for cid, name in spec.delivered.items():
        dtype = spec.columns[cid].dtype
        if dtype not in ("integer", "float", "boolean") or name in fixed or cid in (declared or ()):
            continue
        values = {r.get(cid) for r in rows if r.get(cid) is not None}
        present = [r.get(cid) for r in rows if r.get(cid) is not None]
        if len(values) > 1 and len(present) >= n // 2 and dtype != "boolean":
            top, count = max(((v, present.count(v)) for v in values), key=lambda t: t[1])
            if top != 0 and count >= 0.9 * len(present):         # (a mostly-zero count is a normal shape; a mostly-25.0 amount is not)
                out.append({"id": "near_constant_numeric_column", "severity": "warn", "columns": [name],
                            "problem": f"Column {name} has the value {top!r} on {count / len(present):.0%} of the {len(present)} rows that carry one.",
                            "evidence": f"{len(values)} distinct values, one of them on {count} rows",
                            "fix": "A measure that is the same on nearly every row measures nothing: let it follow what it depends on "
                                   "(the amount, the plan, the cause) so that its magnitude varies the way the real quantity does."})
                continue
        if len(values) == 1 and len(present) >= n // 2:
            out.append({"id": "constant_numeric_column", "severity": "warn" if dtype == "boolean" else "error", "columns": [name],
                        "problem": f"Column {name} has the value {next(iter(values))!r} on every row of every entity.",
                        "evidence": f"{len(rows)} simulated rows, one distinct value",
                        "fix": "A fact varies from entity to entity (and from event to event when it is measured per event): draw it from a "
                               "distribution, or compute it from the columns it is defined by with matching units so its magnitude is realistic."})
    return out[:3]


_BOUND_WORDS = {"total", "limit", "quota", "capacity", "maximum", "max", "cap", "allowance", "ceiling", "entitlement"}
_PART_WORDS = {"remaining", "used", "consumed", "consumption", "spent", "available", "outstanding", "paid", "left"}


def bound_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for a numeric part (remaining, used, consumed...) that exceeds the bound it is part of (total, limit, quota...) on most rows."""
    n = len(rows)
    if n < 100:
        return []
    numeric = {cid: name for cid, name in spec.delivered.items() if spec.columns[cid].dtype in ("integer", "float")}
    found: list[tuple[float, dict[str, Any]]] = []
    for bid, bname in numeric.items():
        if not (_tokens(bname) & _BOUND_WORDS):
            continue
        for xid, xname in numeric.items():
            if xid == bid or (_tokens(xname) & _BOUND_WORDS) or not (_tokens(xname) & _PART_WORDS):
                continue
            pairs = [(r[xid], r[bid]) for r in rows if isinstance(r.get(xid), (int, float)) and isinstance(r.get(bid), (int, float))]
            if len(pairs) < 100:
                continue
            share = sum(1 for x, bound in pairs if x > bound) / len(pairs)
            median_x, median_bound = _median([float(x) for x, _ in pairs]), _median([float(b) for _, b in pairs])
            comparable = median_x > 0 and median_bound > 0 and 0.1 <= median_x / median_bound <= 10.0     # same unit and scale
            if share > 0.5 and comparable:
                found.append((share, {
                    "id": "part_exceeds_bound", "severity": "warn", "columns": [xname, bname],
                    "problem": f"{xname} is larger than {bname} on {share:.0%} of rows.",
                    "evidence": f"{len(pairs)} simulated rows",
                    "fix": f"Unless {xname} is an overage of {bname}, a part of something cannot exceed its total, limit or allowance: "
                           "draw the bound first and derive the part from it (same unit), or derive the bound from its parts."}))
    found.sort(key=lambda t: -t[0])
    return [f for _, f in found[:2]]


_IN_FLIGHT_WORDS = frozenset({"created", "pending", "initiated", "initialised", "initialized", "processing", "queued", "submitted",
                              "inprogress", "awaiting"})
_FINAL_WORDS = frozenset({"completed", "complete", "done", "failed", "cancelled", "canceled", "rejected", "resolved", "closed",
                          "terminated", "verified", "expired", "declined", "abandoned", "aborted", "settled", "succeeded"})


def _value_words(value: str) -> set[str]:
    """The lower-case words of a state value, whatever its spelling (PENDING_VERIFICATION, inProgress, in-progress)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    words = [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]
    return set(words) | ({"inprogress"} if "in" in words and "progress" in words else set())


def _is_in_flight(value: str) -> bool:
    """True for a state that means "still under way": it names an in-flight step and no final outcome."""
    words = _value_words(value)
    return bool(words & _IN_FLIGHT_WORDS) and not (words & _FINAL_WORDS)


STALE_IN_FLIGHT_DAYS = 30.0
_STATE_NAME_WORDS = frozenset({"status", "state", "stage", "phase", "outcome", "result"})   # columns that hold a state, not free wording
_AUTOMATED_ACTOR = re.compile(r"(system|auto|batch|bot|scheduler|platform|api|service|cron|robot)", re.IGNORECASE)


def inflight_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for states that mean "still in flight" (created, pending, processing...) on events long past.

    Such a state is only possible for an event recent enough that it could still be in flight; an old event has reached a
    final state. Measured against the event clock and the simulation's as_of.
    """
    if not spec.clock or len(rows) < 100:
        return []
    as_of = datetime.fromisoformat(_VERIFY_AS_OF)
    entity_level = set(getattr(spec, "entity_columns", []) or [])          # a customer's status is not the state of one event
    out = []
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("string", "categorical") or cid == spec.clock or name in entity_level \
                or not (re.split(r"[^a-z0-9]+", name.lower())[-1] in _STATE_NAME_WORDS):
            continue
        flight = [r for r in rows if isinstance(r.get(cid), str) and _is_in_flight(r[cid]) and r.get(spec.clock) is not None]
        stale = [r for r in flight if (as_of - r[spec.clock]).total_seconds() > STALE_IN_FLIGHT_DAYS * 86400.0]
        if flight and len(stale) >= 0.5 * len(flight) and len(stale) >= 2:
            out.append({"id": "stale_in_flight", "severity": "error", "columns": [name],
                        "problem": f"{len(stale)} of {len(flight)} rows in an in-flight state of {name} are older than {STALE_IN_FLIGHT_DAYS:.0f} days.",
                        "evidence": f"e.g. {sorted({str(r[cid]) for r in stale})[:3]} on events months before as_of",
                        "fix": "Let an in-flight state occur only for events newer than a few days (condition on secs(as_of, event_at)); "
                               "older events get a final state."})
    return out[:2]


def shared_identifier_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for an identifier column in which one human or point-of-sale identifier is shared by many entities.

    People, retailers, agents and devices that act are many; only an automated actor legitimately has one shared identifier.
    """
    entities = {r.get(spec.entity_column) for r in rows} if spec.entity_column else set()
    if len(entities) < 8 or len(rows) < 100:
        return []
    out = []
    for cid, name in spec.delivered.items():
        if cid == spec.entity_column or spec.columns[cid].dtype != "string" or not (_tokens(name) & {"id", "identifier", "key", "code"}):
            continue
        by_value: dict[str, set[Any]] = {}
        for r in rows:
            if isinstance(r.get(cid), str):
                by_value.setdefault(r[cid], set()).add(r.get(spec.entity_column))
        if len(by_value) < len(entities) / 2:
            continue                                  # a code list, not an identifier of individuals
        shared = {v: len(e) for v, e in by_value.items() if len(e) >= 5 and len(e) >= 0.4 * len(entities) and not _AUTOMATED_ACTOR.search(v)}
        if shared:
            value, count = max(shared.items(), key=lambda kv: kv[1])
            out.append({"id": "shared_identifier", "severity": "warn", "columns": [name],
                        "problem": f"The identifier {value!r} in {name} is used by {count} of {len(entities)} entities.",
                        "evidence": f"{len(by_value)} distinct values in {len(rows)} rows",
                        "fix": "An agent, retailer, store, device or other acting party is one of many: draw its identifier from a pool "
                               "(per entity or per event); keep one shared identifier only for an automated or system actor."})
    return out[:2]


_COUNTER_WORDS = {"retry", "retries", "attempt", "attempts", "resend", "resends", "redelivery"}


_STATE_KINDS = {"status", "state", "stage", "phase", "outcome", "result"}


def _kind(name: str) -> str:
    last = re.split(r"[^a-z0-9]+", name.lower())[-1]
    return "state" if last in _STATE_KINDS else last


_GRADE_KINDS = {"score", "level", "rating", "tier", "grade"}
_ID_WORDS = {"id", "identifier", "code", "key", "number"}
_LABEL_WORDS = {"name", "label", "title", "description"}


def mirrored_advice(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Findings for two columns that say the same thing twice.

    Two categorical columns of the same kind (two statuses, two scores, two flags) where one is a function of the other, or two
    columns of any kind that map one-to-one onto each other; two numbers that always add up to (or differ by) one constant, which
    is one measure written twice; two dates that are the same instant on nearly every row. An identifier and its label
    (product_id and plan_name) are one fact written twice by design and are left alone, and so is a pair the expectations
    relate on purpose (a reason that follows its status, a flag that agrees with a state) - while a pair they declare separate is
    judged by ``synth.expectations.check`` as an error.
    """
    n = len(rows)
    if n < 100:
        return []
    delivered = spec.delivered
    related, separate = expect.declared_pairs(expectations)
    exempt = related | separate
    cols = []
    for cid, name in delivered.items():
        dtype = spec.columns[cid].dtype
        if cid == spec.entity_column or dtype not in ("string", "categorical", "boolean", "integer"):
            continue
        counting = dtype == "integer" and _kind(name) not in _GRADE_KINDS          # a count is a label only next to a column named alike
        present = [str(r.get(cid)) for r in rows if r.get(cid) is not None]
        values = set(present)
        # a column that says one thing on nearly every row has no information to repeat (or to be repeated by)
        if 2 <= len(values) <= 12 and len(present) >= 100 and max(present.count(v) for v in values) <= 0.95 * len(present):
            cols.append((cid, name, _kind(name), values, counting))
    found: list[tuple[int, dict[str, Any]]] = []
    for i, (a, na, ka, da, ca) in enumerate(cols):
        for b, nb, kb, db, cb in cols[i + 1:]:
            if frozenset((a, b)) in exempt:
                continue
            ta, tb = _tokens(na), _tokens(nb)
            if (ca or cb) and not {t for t in ta & tb if len(t) >= 5 and t not in _GENERIC_FLAG_WORDS | _GENERIC_TOKENS}:
                continue                                            # a count and a label are tied only when they speak of the same thing
            if (ta & _ID_WORDS and tb & _LABEL_WORDS) or (tb & _ID_WORDS and ta & _LABEL_WORDS):
                continue                                            # an identifier and its label are one fact written twice by design
            if {ka, kb} == {"state", "reason"}:
                continue                                            # a reason elaborates its status; it is meant to follow it
            same_kind = not (ca or cb) and ka == kb and ka in (_STATE_KINDS | {"state", "score", "flag", "type", "level", "rating", "tier", "grade"})
            pairs = [(str(r.get(a)), str(r.get(b))) for r in rows if r.get(a) is not None and r.get(b) is not None]
            if len(pairs) < 100:
                continue
            ab: dict[str, dict[str, int]] = {}
            ba: dict[str, dict[str, int]] = {}
            for x, y in pairs:
                ab.setdefault(x, {})[y] = ab.setdefault(x, {}).get(y, 0) + 1
                ba.setdefault(y, {})[x] = ba.setdefault(y, {}).get(x, 0) + 1
            forward = sum(max(c.values()) for c in ab.values()) / len(pairs)       # share of rows where b is the commonest b for its a
            backward = sum(max(c.values()) for c in ba.values()) / len(pairs)
            one_to_one = forward >= 0.97 and backward >= 0.97 and len(da) == len(db)
            if one_to_one or (same_kind and (forward >= 0.97 or backward >= 0.97)):   # (a count is only ever compared one-to-one)
                follower, leader = (nb, na) if forward >= 0.97 else (na, nb)
                how = (f"map one-to-one onto each other ({len(da)} values each)" if one_to_one else f"{follower} is a function of {leader}")
                found.append((0 if same_kind else 1, _mirrored(na, nb, f"{how} on {min(max(forward, backward), 1.0):.0%} of rows" if not one_to_one
                                                              else f"{how} on {min(forward, backward):.0%} of rows", len(pairs))))
    # numbers that add up to a constant, and dates that are one instant
    entity_level = set(spec.entity_columns)
    measures = [(cid, name, spec.columns[cid].dtype in ("datetime", "date")) for cid, name in delivered.items()
                if spec.columns[cid].dtype in ("integer", "float", "datetime", "date") and cid != spec.entity_column
                and not _tokens(name) & _SHARE_WORDS]
    for i, (a, na, ta_time) in enumerate(measures[:40]):
        for b, nb, tb_time in measures[i + 1:40]:
            if ta_time != tb_time or frozenset((a, b)) in exempt:
                continue
            if ta_time and (_tokens(na) | _tokens(nb)) & _WINDOW_WORDS and not (na in entity_level and nb in entity_level):
                continue                                            # a window opens at the moment of the event that grants it
            pairs = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
            how = expect.same_fact(pairs) if len(pairs) >= 100 else None
            if how:
                said = {"identical": "carry the same value", "sum": "add up to one constant", "difference": "differ by one constant"}.get(how)
                if said:
                    found.append((2, _mirrored(na, nb, f"{said} on nearly every one of {len(pairs)} rows", len(pairs))))
    found.sort(key=lambda t: t[0])
    return [f for _, f in found][:4]


def _mirrored(na: str, nb: str, how: str, n: int) -> dict[str, Any]:
    return {"id": "mirrored_columns", "severity": "warn", "columns": [na, nb], "problem": f"{na} and {nb}: {how}.",
            "evidence": f"{n} simulated rows",
            "fix": "Two columns must describe different things (another stage, party or measure) or differ where reality differs "
                   "(they are recorded at different moments, by different parties); do not relabel, copy or derive one from the other "
                   "one-to-one, and do not make two measures exact complements: give each its own hidden driver (correlated if they are "
                   "related, with its own variation) and each timestamp its own delay from the step before it."}


def counter_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for a counter that runs on from one event to the next although the events are days apart.

    A retry or attempt count belongs to its own event (how many tries this request needed) and may be anything; a count that
    is its predecessor's plus one far more often than chance pairing of the same values would give is a running total across
    unrelated events.
    """
    groups = _by_entity(spec, rows)
    if not groups or not spec.clock:
        return []
    out = []
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype != "integer" or not (_tokens(name) & _COUNTER_WORDS):
            continue
        pairs: list[tuple[int, int]] = []
        spans: list[float] = []
        for g in groups:
            for prev, nxt in zip(g, g[1:]):
                a, b = prev.get(cid), nxt.get(cid)
                if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and b > 0:
                    pairs.append((a, b))
                    spans.append((nxt[spec.clock] - prev[spec.clock]).total_seconds() / 86400.0)
        if len(pairs) < 30 or _median(spans) <= 1.0:
            continue
        observed = sum(b == a + 1 for a, b in pairs) / len(pairs)
        firsts, seconds = [a for a, _ in pairs], [b for _, b in pairs]
        shift = max(1, len(pairs) // 3)                         # the same values paired with other rows: what chance gives
        chance = sum(seconds[i] == firsts[(i + shift) % len(pairs)] + 1 for i in range(len(pairs))) / len(pairs)
        if observed >= 0.5 and observed >= chance + 0.2:
            out.append({"id": "counter_spacing", "severity": "error", "columns": [name],
                        "problem": f"{name} is the previous event's value plus one on {observed:.0%} of {len(pairs)} steps (chance pairing gives {chance:.0%}) "
                                   f"although those events are a median of {_median(spans):.0f} days apart.",
                        "evidence": f"{len(pairs)} consecutive pairs with a positive counter",
                        "fix": "A retry or attempt count belongs to its own event: draw it per event (more tries for failed or retried requests, none "
                               "for a first-time success); do not carry it over from the previous event."})
    return out[:2]


_GENERIC_FLAG_WORDS = frozenset({"flag", "indicator", "required", "customer", "subscriber", "account", "status", "state", "issue", "first",
                                 "contact", "current", "latest", "previous", "total", "count", "value"})
_NEGATING_PREFIXES = ("un", "non", "not", "in", "dis")


_SUFFIXES = ("ation", "ition", "ing", "ed", "ion", "s", "d", "e")


def _stem(word: str) -> str:
    """The word without its ending, so that accepted / accept / accepts meet (but assistance and assisted do not)."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _flag_stems(name: str) -> set[str]:
    return {_stem(t) for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) >= 5 and t not in _GENERIC_FLAG_WORDS}


def _names_step(word: str, stems: set[str]) -> bool:
    """The word is a stem of the flag, or a whole word the flag's longer word begins with (RETRY / is_retryable)."""
    stem = _stem(word)
    return stem in stems or any(len(word) >= 5 and s.startswith(word) for s in stems)


def _value_polarity(value: str, stems: set[str]) -> int:
    """+1 when the value names what the flag says happened (Accepted / accepted_flag), -1 when it names the opposite (UNRESOLVED), else 0."""
    for word in _value_words(value):
        if len(word) >= 5 and _names_step(word, stems):
            return 1
        for prefix in _NEGATING_PREFIXES:
            rest = word[len(prefix):]
            if word.startswith(prefix) and len(rest) >= 5 and _names_step(rest, stems):
                return -1
    return 0


def _shares_subject(first: set[str], second: set[str]) -> bool:
    """Two sets of word stems name the same subject when a stem of one and a stem of the other begin alike (resolv / resolut)."""
    return any(len(os.path.commonprefix([a, b])) >= 5 for a in first for b in second)


def flag_state_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings where a yes/no flag disagrees with a state column that names the same thing.

    ``offer_accepted_flag`` is true where the response is Accepted and false where it is Rejected; ``customer_issue_resolved_flag``
    is true for RESOLVED and false for UNRESOLVED. The words are the only link the two columns have, so they are matched by
    stem; a value that names the flag's step must carry the flag true, one that negates it false.
    """
    n = len(rows)
    if n < 100:
        return []
    flags = [(cid, name, _flag_stems(name)) for cid, name in spec.delivered.items() if spec.columns[cid].dtype == "boolean"]
    flags = [f for f in flags if f[2]]
    if not flags:
        return []
    out = []
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("string", "categorical") or cid == spec.entity_column:
            continue
        values = {r[cid] for r in rows if isinstance(r.get(cid), str)}
        if not 2 <= len(values) <= 12:
            continue
        own = _flag_stems(name)                                  # words the two columns share name the resource, not the step
        for fid, fname, flag_stems in flags:
            stems = flag_stems - own
            if not stems:
                continue
            for value in sorted(values):
                polarity = _value_polarity(value, stems)
                if not polarity:
                    continue
                known = [r for r in rows if r.get(cid) == value and isinstance(r.get(fid), bool)]
                wrong = [r for r in known if r[fid] != (polarity > 0)]
                if len(known) >= 8 and len(wrong) >= 5 and len(wrong) >= 0.1 * len(known):
                    # an error only when the two columns speak of the same subject (a shared word in their names, such as the
                    # offer in offer_accepted_flag / offer_response_status); a state of some other process that happens to
                    # use the same word (a consumption state 'accepted') is a doubt, not a contradiction
                    same_subject = _shares_subject(flag_stems, own) or bool(_tokens(name) & _ACTION_WORDS)   # (an action is what a permission flag governs)
                    out.append({"id": "flag_contradicts_state", "severity": "error" if same_subject else "warn", "columns": [fname, name],
                                "problem": f"{name} is '{value}' but {fname} is {'false' if polarity > 0 else 'true'} on {len(wrong)} of {len(known)} such rows.",
                                "evidence": f"{len(known)} simulated rows with {name} = '{value}'",
                                "fix": f"Derive {fname} from {name} (or both from one draw) so that a '{value}' row always carries "
                                       f"{fname} {'true' if polarity > 0 else 'false'}."})
    return out[:3]


_SHARE_WORDS = frozenset({"share", "percent", "percentage", "pct", "ratio", "proportion", "fraction"})


def _mode_count(values: list[float]) -> tuple[float, int]:
    """The commonest value of a list and how often it occurs."""
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1])
    return top[0], top[1]


def near_constant_state_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for the state of an event that is the same on nearly every row while another state of it varies.

    When a request's status reads "completed" on 99% of its events and a second state (a verification, a settlement) says
    what really happened, the first state carries no information and usually repeats the outcome inside its value.
    """
    n = len(rows)
    if n < 100:
        return []
    entity_level = set(spec.entity_columns)
    states = []
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("string", "categorical") or cid == spec.clock or name in entity_level \
                or re.split(r"[^a-z0-9]+", name.lower())[-1] not in _STATE_NAME_WORDS:
            continue
        present = [str(r[cid]) for r in rows if r.get(cid) is not None]
        if len(present) < 0.9 * n:
            continue
        counts: dict[str, int] = {}
        for v in present:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.items(), key=lambda kv: kv[1])
        states.append((name, top[0], top[1] / len(present), sum(c >= 0.1 * len(present) for c in counts.values()), set(counts)))
    varying = [(s[0], set().union(*(_value_words(v) for v in s[4]))) for s in states if s[3] >= 2]
    out = []
    for name, top, share, _, _ in states:
        if share < 0.97:
            continue
        # the dominant value repeats a word of the outcome another state records ("completed_discrepancy" next to VERIFIED_DISCREPANCY)
        twin = next((v for v, words in varying if v != name and {w for w in _value_words(top) & words if len(w) >= 5}), None)
        if twin:
            out.append({"id": "near_constant_state", "severity": "warn", "columns": [name, twin],
                        "problem": f"{name} is '{top}' on {share:.0%} of the rows and repeats the outcome that {twin} records, which varies.",
                        "evidence": f"{len(rows)} simulated rows",
                        "fix": f"Let {name} follow what happened to the event (the stages it can really reach, in realistic proportions) "
                               f"and keep the outcome that {twin} records out of its value."})
    return out[:2]


_WINDOW_WORDS = frozenset({"valid", "start", "begin", "effective", "from", "end", "expiry", "expires", "until"})


def sparse_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for delivered columns that are present on (almost) no row: they cost a column but say nothing."""
    n = len(rows)
    if n < 100:
        return []
    out = []
    for cid, name in spec.delivered.items():
        present = sum(1 for r in rows if r.get(cid) is not None)
        if 0 < present < 0.01 * n:
            out.append({"id": "near_empty_column", "severity": "warn", "columns": [name],
                        "problem": f"Column {name} has a value on only {present} of {n} simulated rows.",
                        "evidence": f"{present}/{n} rows non-empty",
                        "fix": "Let it carry a value on every row whose situation exists (the situation may be too narrow), or tie it to a more common situation."})
    return out[:2]


def shape_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured facts a reviewer cannot see in a handful of rows: the cadence of events and how emptiness depends on other columns."""
    facts: dict[str, Any] = {"events_per_entity": spec.timeline.get("events"), "history_days": spec.timeline.get("history_days")}
    groups = _by_entity(spec, rows)
    if groups and spec.clock:
        gaps = [(g[i + 1][spec.clock] - g[i][spec.clock]).total_seconds() / 86400.0 for g in groups for i in range(len(g) - 1)]
        if gaps:
            facts["median_days_between_events"] = round(_median(gaps), 1)
    n = len(rows)
    delivered = spec.delivered
    explainers = []
    for cid in delivered:
        col = spec.columns[cid]
        if col.dtype in ("string", "categorical", "boolean") and col.kind != "latent":
            values = {str(r.get(cid)) for r in rows if r.get(cid) is not None}
            if 2 <= len(values) <= 8 and not any(len(str(v)) > 40 for v in values):
                explainers.append(cid)
    presence = []
    for cid, name in delivered.items():
        empty = sum(1 for r in rows if r.get(cid) is None)
        if not n or empty == 0 or empty == n:
            continue
        best = None
        for e in explainers:
            if e == cid:
                continue
            table: dict[str, list[int]] = {}
            for r in rows:
                key = str(r.get(e)) if r.get(e) is not None else "(empty)"
                cell = table.setdefault(key, [0, 0])
                cell[0] += 1
                cell[1] += r.get(cid) is None
            purity = sum(max(c[1], c[0] - c[1]) for c in table.values()) / n
            if best is None or purity > best[0]:
                best = (purity, e, table)
        entry: dict[str, Any] = {"column": name, "empty_share": round(empty / n, 2)}
        if best is not None and best[0] >= 0.9:
            entry["empty_share_by"] = {delivered[best[1]]: {k: round(c[1] / c[0], 2) for k, c in sorted(best[2].items())}}
        presence.append((0 if "empty_share_by" in entry else 1, entry))
    presence.sort(key=lambda t: t[0])
    constant = {name: rows[0].get(cid) for cid, name in delivered.items()
                if n >= 30 and len({str(r.get(cid)) for r in rows}) == 1 and rows[0].get(cid) is not None}
    if constant:
        facts["constant_columns"] = {k: (v if isinstance(v, (int, float, bool)) else str(v)[:40]) for k, v in list(constant.items())[:MAX_FACT_COLUMNS]}
    spread: dict[str, Any] = {}
    for cid, name in delivered.items():
        if spec.columns[cid].dtype in ("integer", "float") and name not in constant:
            vals = sorted(float(r[cid]) for r in rows if isinstance(r.get(cid), (int, float)) and not isinstance(r.get(cid), bool))
            if len(vals) >= 30:
                spread[name] = {"p5": round(vals[int(0.05 * (len(vals) - 1))], 2), "median": round(vals[len(vals) // 2], 2),
                                "p95": round(vals[int(0.95 * (len(vals) - 1))], 2)}
    if spread:
        facts["numeric_spread"] = dict(list(spread.items())[:30])
    if presence:
        facts["sometimes_empty_columns"] = [e for _, e in presence[:MAX_FACT_COLUMNS]]
    flags = _flag_facts(spec, rows)
    if flags:
        facts["flags_vs_timestamps"] = flags
    for key, found in (("item_vs_measure", _item_facts(spec, rows)), ("entity_constant_timestamps", _entity_stamp_facts(spec, rows)),
                       ("flags_vs_columns", _flag_column_facts(spec, rows)), ("constant_ratios", _ratio_facts(spec, rows))):
        if found:
            facts[key] = found
    return facts


_ITEM_WORDS = frozenset({"product", "plan", "offering", "package", "item", "sku", "pack", "bundle"})
_MEASURE_WORDS = frozenset({"amount", "price", "fee", "cost", "charge", "size", "limit", "allocated", "validity", "quota", "volume"})
_ACTION_WORDS = frozenset({"action", "remedy", "decision", "next"})


def _item_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Item names (product, plan, offer) whose price or size is not a function of the item.

    An item has its own price, size and validity; a name that sells at several different amounts is two facts drawn
    independently. Only the weak pairs are handed over; a name whose measure follows it is as it should be.
    """
    n = len(rows)
    items, measures = [], []
    for cid, name in spec.delivered.items():
        dtype = spec.columns[cid].dtype
        if cid == spec.entity_column or not _tokens(name) & (_ITEM_WORDS | _MEASURE_WORDS):
            continue
        present = [r[cid] for r in rows if r.get(cid) is not None and not isinstance(r.get(cid), bool)]
        if len(present) < max(100, n // 2):
            continue
        distinct = {round(v, 2) if isinstance(v, float) else v for v in present}
        if dtype in ("string", "categorical") and _tokens(name) & _ITEM_WORDS and 2 <= len(distinct) <= 12:
            items.append((cid, name))
        elif dtype in ("integer", "float") and _tokens(name) & _MEASURE_WORDS and _kind(name) not in _GRADE_KINDS and 2 <= len(distinct) <= 12:
            measures.append((cid, name))
    kinds = []
    for cid, name in spec.delivered.items():
        if cid != spec.entity_column and spec.columns[cid].dtype in ("string", "categorical") and _kind(name) in {"type", "category", "kind", "class"}:
            present = [str(r[cid]) for r in rows if r.get(cid) is not None]
            if len(present) >= max(100, n // 2) and 2 <= len(set(present)) <= 8:
                kinds.append((cid, name))
    out: dict[str, tuple[float, str]] = {}
    for a, na in items:
        for b, nb in measures:
            table: dict[Any, dict[Any, int]] = {}
            for r in rows:
                x, y = r.get(a), r.get(b)
                if x is not None and y is not None:
                    cell = table.setdefault(x, {})
                    cell[round(y, 2)] = cell.get(round(y, 2), 0) + 1
            total = sum(sum(c.values()) for c in table.values())
            if total < 100:
                continue
            agreement = sum(max(c.values()) for c in table.values()) / total
            if agreement < 0.6:
                spread = sum(len(c) for c in table.values()) / len(table)
                out[f"{na} -> {nb}"] = (agreement, f"{agreement:.0%} of rows carry the item's commonest {nb}; {spread:.1f} different values per item on average")
    for a, na in items:
        for b, nb in kinds:                                     # the kind of thing an item is (data, voice, monetary) follows the item
            table = {}
            for r in rows:
                x, y = r.get(a), r.get(b)
                if x is not None and y is not None:
                    cell = table.setdefault(x, {})
                    cell[str(y)] = cell.get(str(y), 0) + 1
            total = sum(sum(c.values()) for c in table.values())
            if total < 100:
                continue
            overall: dict[str, int] = {}
            for c in table.values():
                for y, k in c.items():
                    overall[y] = overall.get(y, 0) + k
            agreement = sum(max(c.values()) for c in table.values()) / total
            baseline = max(overall.values()) / total
            if len(overall) >= 2 and baseline < 0.95 and agreement < baseline + 0.05:
                out[f"{na} -> {nb}"] = (agreement, f"the item tells nothing about {nb}: {agreement:.0%} of rows carry the item's commonest value, "
                                                   f"{baseline:.0%} carry the commonest value overall")
    return {k: v[1] for k, v in sorted(out.items(), key=lambda kv: kv[1][0])[:5]}


def _ratio_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Numeric columns that are the same fraction of another column on nearly every row (remaining = 90% of the amount).

    That is how a tax or a unit conversion behaves; a balance, a usage or a remainder follows what happened and does not.
    """
    n = len(rows)
    cols = []
    for cid, name in spec.delivered.items():
        if n >= 100 and spec.columns[cid].dtype in ("integer", "float") and not _tokens(name) & _SHARE_WORDS:
            values = [r[cid] for r in rows if isinstance(r.get(cid), (int, float)) and not isinstance(r.get(cid), bool) and r[cid] > 0]
            if len(values) >= 30 and len({round(v, 2) for v in values}) >= 5:
                cols.append((cid, name))
    out: dict[str, str] = {}
    for i, (a, na) in enumerate(cols[:30]):
        for b, nb in cols[i + 1:30]:
            ratios = [float(f"{r[b] / r[a]:.3g}") for r in rows if isinstance(r.get(a), (int, float)) and isinstance(r.get(b), (int, float))
                      and not isinstance(r.get(a), bool) and not isinstance(r.get(b), bool) and r[a] > 0 and r[b] > 0]
            if len(ratios) < 30:
                continue
            top, count = _mode_count(ratios)
            if count >= 0.97 * len(ratios) and abs(top - 1.0) > 0.001:
                out[f"{nb} / {na}"] = f"{top:g} on {count / len(ratios):.0%} of the {len(ratios)} rows where both are above zero"
    return dict(list(out.items())[:4])


def _entity_stamps(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Date columns that hold one value per entity: the share of the entity's events that happen before that date, and how many events were counted."""
    groups = _by_entity(spec, rows)
    if not groups:
        return {}
    out: dict[str, tuple[float, int]] = {}
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("datetime", "date") or cid == spec.clock:
            continue
        events = before = 0
        constant = True
        try:
            for g in groups:
                stamps = {r.get(cid) for r in g}
                if len(stamps) != 1:
                    constant = False
                    break
                stamp = next(iter(stamps))
                if stamp is not None:
                    events += len(g)
                    before += sum(r[spec.clock] < stamp for r in g)
        except TypeError:
            continue
        if constant and events >= 30:
            out[name] = (before / events, events)
    return out


def _entity_stamp_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Date columns that hold one value per entity while the entity's events run on: how many events happen before that date."""
    found = {name: f"one value per entity; {share:.0%} of the entity's events happen before it"
             for name, (share, _) in _entity_stamps(spec, rows).items() if 0.05 <= share <= 0.95}
    return dict(list(found.items())[:6])


def entity_stamp_advice(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for a date that is fixed for a whole entity yet lies in the middle of that entity's events.

    A notification, an impression, a response, a conversion or a "last top-up" belongs to one event (or is the entity's latest
    one as of that event); a single value for the whole history cannot be before some of the events and after the others.
    """
    if len(rows) < 100:
        return []
    return [{"id": "entity_stamp_inside_history", "severity": "warn", "columns": [name],
             "problem": f"{name} has one value per entity but {share:.0%} of that entity's events happen before it.",
             "evidence": f"{events} events of the simulated entities",
             "fix": f"Record {name} per event (present only when that step happened, after the step before it), or, for the latest "
                    "earlier occurrence, derive it from the entity's previous events (prev) so it never lies after the event it is read at."}
            for name, (share, events) in _entity_stamps(spec, rows).items() if 0.05 <= share <= 0.95][:3]


def _flag_column_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For each yes/no flag: what retry counters and next-action columns say on the rows where it is true and where it is false.

    A "retryable" flag that is false should not sit next to attempts or a retry action; the sample alone rarely shows the share.
    """
    delivered = spec.delivered
    counters = [(cid, name) for cid, name in delivered.items() if spec.columns[cid].dtype == "integer" and _tokens(name) & _COUNTER_WORDS]
    actions = [(cid, name) for cid, name in delivered.items()
               if spec.columns[cid].dtype in ("string", "categorical") and _tokens(name) & _ACTION_WORDS
               and 2 <= len({r.get(cid) for r in rows if r.get(cid) is not None}) <= 12]
    if not counters and not actions:
        return {}
    out: dict[str, Any] = {}
    for fid, fname in delivered.items():
        if spec.columns[fid].dtype != "boolean" or len(out) >= 6:
            continue
        yes = [r for r in rows if r.get(fid) is True]
        no = [r for r in rows if r.get(fid) is False]
        if len(yes) < 0.05 * len(rows) or len(no) < 0.05 * len(rows):
            continue
        entry: dict[str, Any] = {}
        for cid, name in counters:
            positive = [sum(isinstance(r.get(cid), int) and r[cid] > 0 for r in side) / len(side) for side in (yes, no)]
            entry[name] = f"above zero on {positive[0]:.0%} of rows where true, {positive[1]:.0%} where false"
        for cid, name in actions:
            sides = []
            for side in (yes, no):
                counts: dict[str, int] = {}
                for r in side:
                    if r.get(cid) is not None:
                        counts[str(r[cid])] = counts.get(str(r[cid]), 0) + 1
                top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
                sides.append({k: round(c / len(side), 2) for k, c in top})
            entry[name] = {"where_true": sides[0], "where_false": sides[1]}
        out[fname] = {"true_share": round(len(yes) / (len(yes) + len(no)), 2), **entry}
    return out


def _flag_facts(spec: GenerationSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For each yes/no flag that is true on some rows and false on others: the timestamps that do not follow it.

    A flag that says a step happened should agree with that step's timestamp; the reviewer cannot see across a sample whether
    a timestamp is present on every row whatever the flag says, so the measurement is handed over.
    """
    delivered = spec.delivered
    stamps = [(cid, name) for cid, name in delivered.items() if spec.columns[cid].dtype in ("datetime", "date") and cid != spec.clock]
    out: dict[str, Any] = {}
    for fid, fname in delivered.items():
        if spec.columns[fid].dtype != "boolean" or len(out) >= 6:
            continue
        yes = [r for r in rows if r.get(fid) is True]
        no = [r for r in rows if r.get(fid) is False]
        if len(yes) < 0.05 * len(rows) or len(no) < 0.05 * len(rows):
            continue
        entries = {}
        for cid, name in stamps:
            when_yes = sum(r.get(cid) is not None for r in yes) / len(yes)
            when_no = sum(r.get(cid) is not None for r in no) / len(no)
            if when_no >= 0.9 or when_yes <= 0.1 or abs(when_yes - when_no) >= 0.5:
                entries[name] = (abs(when_yes - when_no), f"present on {when_yes:.0%} of rows where true, {when_no:.0%} where false")
        if entries:
            ranked = sorted(entries.items(), key=lambda kv: -kv[1][0])[:5]
            out[fname] = {"true_share": round(len(yes) / (len(yes) + len(no)), 2), "timestamps": {k: v[1] for k, v in ranked}}
    return out


# ------------------------------------------------------------------------------------------------------------------
REVIEW_PROMPT = """You are an independent quality reviewer of a synthetic dataset. You did not write the rules that produced it.
You receive the scenario, the columns (with their definitions) and a sample of the generated data: a few entities, each with
the facts that describe the entity once ("entity") and its events in chronological order ("events", oldest first).

Find records that could NOT exist in a real system running this scenario, and facts a domain expert would immediately call wrong.
Look for:
- contradictions between columns of one row (a state that its timestamps, amounts or flags do not support; an effect before
  its cause; a quantity larger than the limit it is measured against; a window that does not contain what it should);
- contradictions between rows of one entity (a fact that should be constant but changes; events that overlap or ignore a
  window set by an earlier event);
- the same fact stored twice with different values (an identifier, amount, state, flag or party under two names);
- facts at the wrong level (an attribute of the entity that changes every row, or the reverse);
- an empty value where the situation requires one, or a value where the situation cannot have one;
- values that are placeholders or meaningless for the scenario's industry and country (generic tokens, nonsensical names,
  magnitudes, formats or scales that no real system would use) and identifiers of inconsistent format;
- relationships that matter in the scenario but are missing (a recommended amount unrelated to what was bought, a response
  unrelated to the offer);
- proportions that are implausible for the scenario type (an outcome that is rare in reality happening on half the rows).
"facts" are measured on a larger simulation of the same rules than the sample: the cadence of events (median days between an
entity's events) and, for columns that are sometimes empty, how the share of empty values depends on another column
("empty_share_by": value -> share of rows where the column is empty) and the columns that carry one single value on every row
("constant_columns": column -> its one value; acceptable only for a fact that is truly fixed for the scenario, such as a currency)
the timestamps that do not follow a yes/no flag ("flags_vs_timestamps": a timestamp present on nearly every row where the flag is false, or absent
where it is true, contradicts a flag that says that step happened), the item names whose price or size is not a function of the item
("item_vs_measure": a product sold at several different prices, or of a kind - data, voice - that does not follow the product, is wrong), the dates fixed for a whole entity although its events go on
before and after them ("entity_constant_timestamps") and what retry counters and next-action columns say when a yes/no flag is true and when it is false
("flags_vs_columns": a retry count or a retry action next to a flag that says "not retryable" is a contradiction), the quantities that are always the same fraction of another
("constant_ratios": right for a tax or a unit conversion, wrong for a balance, a remainder or a usage) and the 5th/50th/95th percentile of every numeric column ("numeric_spread"): judge whether those magnitudes are realistic for what the
column means (a time until depletion of one hour on nearly every row, an average of 0 for a quantity that is never 0, a count that
exceeds what the other columns allow). Judge each against what the column means: a fact that
only exists once something succeeded must be empty for every state that is not a success; a fact that applies must not be
empty; windows (validity, term, cooldown) should not still be open when the same entity's next event of that kind starts
unless the scenario renews them early; items (name, price, size, validity of one product or plan) must agree with each other;
the actor's role, identifier and channel must agree (and with an automatic/scheduled flag: automatic means requested by the system and paid by a
means that can run unattended, not cash or in person; one retailer/agent identifier shared by many customers is wrong);
a flag that says a step happened (presented, viewed, accepted, converted, resolved) agrees with that step's timestamp, amount and status,
and a later step is never true while the step before it is false; statuses of one subject must not contradict each other.
Timestamp layouts ("timestamp_layouts") are fixed by the platform, are written in the scenario's local time and are rounded to the
layout's unit: never report a layout, a missing UTC offset or a rounding effect. Report only what the sample demonstrates,
citing the entity/event and the values. Do not report style, wording, or anything you cannot show. Do not report that a column is "random" unless random values contradict something. At most 8 issues,
most serious first. severity "error" = a record that is impossible or contradictory; "warn" = implausible or weak.

OUTPUT: one JSON object, nothing else:
{"issues":[{"id":"short_snake_id","severity":"error|warn","columns":["column names involved"],
            "problem":"what is wrong, in one or two sentences","evidence":"entity/event and values that show it",
            "fix":"how the generating behaviour should change"}]}
If the sample is sound: {"issues":[]}"""


def review_sample(spec: GenerationSpec, records: list[dict[str, Any]], *, entities: int = REVIEW_ENTITIES,
                  events: int = REVIEW_EVENTS) -> list[dict[str, Any]]:
    """A compact, deterministic sample of ``records`` grouped by entity: entity facts once, events oldest first."""
    entity_cols = [c for c in spec.entity_columns]
    key = spec.columns[spec.entity_column].column if spec.entity_column else None
    groups: dict[Any, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(rec.get(key) if key else len(groups), []).append(rec)
    out = []
    for rows in list(groups.values())[:entities]:
        rows = rows[-events:]               # the engine emits an entity's rows oldest first; a client reads the most recent ones
        first = rows[0]
        out.append({"entity": {c: first.get(c) for c in entity_cols},
                    "events": [{k: v for k, v in r.items() if k not in entity_cols} for r in rows]})
    return out


def build_review_prompt(brief: dict[str, Any], base: Baseline, spec: GenerationSpec, records: list[dict[str, Any]],
                        notes: dict[str, dict[str, Any]] | None = None, resources: dict[str, str] | None = None,
                        facts: dict[str, Any] | None = None) -> str:
    scenario = {k: v for k, v in brief.items() if v not in (None, "") and k in SCENARIO_KEYS}
    cards = column_cards(base, notes)
    for card in cards:                      # the reviewer reads delivered names, as the client sees them
        card.pop("needs_values", None)
    layouts = sorted({style_of(spec, c) or "RFC 3339" for c in spec.columns.values() if c.dtype == "datetime" and c.kind != "latent"})
    context: dict[str, Any] = {"scenario": scenario, "entity_column": spec.columns[spec.entity_column].column if spec.entity_column else None,
                               "clock_column": spec.columns[spec.clock].column if spec.clock else None, "currency": spec.currency,
                               "timezone": spec.timezone, "as_of": _VERIFY_AS_OF,
                               "timestamp_layouts": layouts, "events_order": "oldest first"}
    if resources:
        context["source_resources"] = resources
    context["columns"] = cards
    context["sample"] = review_sample(spec, records)
    return "Review this dataset sample.\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_review(raw: Any) -> list[dict[str, Any]]:
    """The reviewer's issues, normalised; anything that is not a well-formed issue is ignored."""
    issues = raw.get("issues") if isinstance(raw, dict) else None
    out = []
    for item in issues if isinstance(issues, list) else []:
        if not isinstance(item, dict) or not str(item.get("problem") or "").strip():
            continue
        severity = "error" if str(item.get("severity") or "").lower() == "error" else "warn"
        cols = [str(c) for c in item.get("columns") or [] if c] if isinstance(item.get("columns"), list) else []
        out.append({"id": str(item.get("id") or "issue")[:60], "severity": severity, "columns": cols[:8],
                    "problem": str(item["problem"]).strip()[:400], "evidence": str(item.get("evidence") or "").strip()[:400],
                    "fix": str(item.get("fix") or "").strip()[:400]})
    return out[:8]


def review_problem(issue: dict[str, Any]) -> str:
    return (f"REVIEW {issue['id']} on {issue['columns']}: {issue['problem']} Evidence: {issue['evidence']} "
            f"Suggested change: {issue['fix']}")


# ------------------------------------------------------------------------------------------------------------------
def spec_key(brief: dict[str, Any], variables: list[dict[str, Any]]) -> str:
    """Identity of one compile: the scenario facts, the definitions of the final columns, and the compiler itself."""
    keys = SCENARIO_KEYS
    cols = []
    for v in variables:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        cols.append({k: v.get(k) for k in ("name", "dtype", "gen", "params", "formula", "depends_on", "nullable", "scope", "source", "description")})
    payload = {
        "compiler": [COMPILER_VERSION, SPEC_VERSION, hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12], expect.PROMPT_HASH],
        "brief": {k: (brief.get(k) or "") for k in keys},
        "columns": sorted(cols, key=lambda c: str(c["name"])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:40]


MEASURED_FINDINGS = frozenset({"window_overlap", "constant_numeric_column", "near_constant_numeric_column", "constant_column", "near_empty_column",
                               "part_exceeds_bound", "stale_in_flight", "shared_identifier", "mirrored_columns", "counter_spacing",
                               "flag_contradicts_state", "near_constant_state", "entity_stamp_inside_history", "healed_empty_column",
                               "healed_target", "check"})


def is_measured(finding: dict[str, Any]) -> bool:
    """A finding that was measured on the simulation (certain, and the same on every run), as opposed to the reviewer's opinion."""
    fid = str(finding.get("id") or "")
    return fid in MEASURED_FINDINGS or fid.startswith("expected_")


def _measured_weight(findings: list[dict[str, Any]]) -> int:
    """The weight of the findings that were measured on the simulation (the reviewer's findings change from run to run)."""
    return _weight([f for f in findings if is_measured(f)])


def _measured_errors(findings: list[dict[str, Any]]) -> int:
    return sum(1 for f in findings if is_measured(f) and f["severity"] == "error")


def _weight(findings: list[dict[str, Any]]) -> int:
    """How much is still wrong: an impossible record weighs three implausible ones."""
    return 3 * sum(f["severity"] == "error" for f in findings) + sum(f["severity"] != "error" for f in findings)


def _accepts(candidate: list[dict[str, Any]], best: list[dict[str, Any]], *, total: bool = False) -> bool:
    """Whether ``candidate`` findings are an improvement on ``best`` without making anything that was measured worse.

    No new measured error, no heavier measured findings; and then the measured part must be lighter - or, with ``total``, the
    whole (the reviewer's opinions included) must be.
    """
    if _measured_errors(candidate) > _measured_errors(best) or _measured_weight(candidate) > _measured_weight(best):
        return False
    return _measured_weight(candidate) < _measured_weight(best) or (total and _weight(candidate) < _weight(best))


_EXPECTATION_CACHE: dict[str, dict[str, Any]] = {}
_EXPECTATION_CACHE_SIZE = 64
_EXPECTATION_LOCK = threading.Lock()


@dataclass
class _Round:
    spec: GenerationSpec | None
    overlay: dict[str, Any]
    report: dict[str, Any]
    problems: list[str]
    records: list[dict[str, Any]] = field(default_factory=list)


def _permanent(exc: Exception) -> bool:
    """A model failure that trying again cannot fix: no key or library, or the provider rejecting the request itself."""
    if type(exc) in (OSError, ImportError, RuntimeError):        # exactly these: a missing key or library (not a timeout, which is an OSError subclass)
        return True
    return getattr(exc, "status_code", None) in (400, 401, 403, 404)


def finding_text(finding: dict[str, Any]) -> str:
    return f"{finding['severity']}: {finding['problem']} ({', '.join(finding['columns'])})"


class SpecCompiler:
    """Drives the models: author a spec, simulate-and-check it, have it reviewed, repair; accept only a spec that passes.

    Two checks stand between a draft and the dataset. The deterministic one (structure, runtime, the definition
    contract, the author's own invariants and targets, placeholders, impossible future times) must pass for a spec to be
    accepted at all. The second looks for what is implausible rather than impossible: measured advice (windows that overlap
    because events are too close together) and a reviewer model that did not write the spec and reads a sample of the data
    for what a person would see is wrong.

    The two are separate phases so a scenario never waits for the slow one. ``draft`` produces the first spec that passes the
    deterministic checks (usually one model call) and is all that generation needs; ``refine`` then sends what the second
    check finds back to the author as a patch, round by round, and returns a better spec if it found one. ``compile`` runs
    both in sequence.
    """

    def __init__(self, llm: Any | None = None, *, max_repairs: int = MAX_REPAIR_ROUNDS, review: bool | None = None,
                 refine_rounds: int = MAX_REFINE_ROUNDS):
        self._llm = llm
        self.max_repairs = max_repairs
        self.refine_rounds = refine_rounds
        self._review = review

    @staticmethod
    def _budget() -> float:
        from config.runtime import SPEC_COMPILE_BUDGET_SECONDS

        return float(SPEC_COMPILE_BUDGET_SECONDS)

    @staticmethod
    def _repair_window() -> float:
        from config.runtime import SPEC_DRAFT_REPAIR_SECONDS

        return float(SPEC_DRAFT_REPAIR_SECONDS)

    @staticmethod
    def _refine_budget() -> float:
        from config.runtime import SPEC_REFINE_BUDGET_SECONDS

        return float(SPEC_REFINE_BUDGET_SECONDS)

    def reviewing(self) -> bool:
        if self._review is not None:
            return self._review
        from config.runtime import SPEC_REVIEW

        return bool(SPEC_REVIEW)

    def _client(self) -> Any:
        if self._llm is None:
            from core.llm_client import GeminiClient
            from config.runtime import SPEC_LLM_TIMEOUT_MS

            self._llm = GeminiClient(timeout_ms=SPEC_LLM_TIMEOUT_MS)
        return self._llm

    def _review_issues(self, brief: dict[str, Any], base: Baseline, spec: GenerationSpec, records: list[dict[str, Any]],
                       notes: dict[str, dict[str, Any]] | None, resources: dict[str, str] | None,
                       facts: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        """The reviewer's findings, or None when the reviewer could not be consulted (which never blocks a spec)."""
        try:
            raw = self._generate(REVIEW_PROMPT, build_review_prompt(brief, base, spec, records, notes, resources, facts), attempts=2)
        except Exception as exc:
            logger.warning("behaviour spec review unavailable (%s: %s)", type(exc).__name__, exc)
            return None
        return parse_review(raw)

    def _generate(self, system: str, prompt: str, *, attempts: int = MODEL_ATTEMPTS) -> Any:
        """One model call; a transient failure (overload, rate limit, timeout, a reply that is not JSON) is tried again."""
        for attempt in range(1, attempts + 1):
            try:
                return self._client().generate_json(system_instruction=system, user_prompt=prompt, temperature=0.0)
            except Exception as exc:
                if attempt >= attempts or _permanent(exc):
                    raise
                pause = MODEL_RETRY_PAUSE[min(attempt - 1, len(MODEL_RETRY_PAUSE) - 1)]
                logger.warning("behaviour spec model call failed (%s: %s); attempt %d of %d, again in %.0fs",
                               type(exc).__name__, _short(str(exc), 160), attempt, attempts, pause)
                time.sleep(pause)
        raise RuntimeError("unreachable")

    def _verified(self, base: Baseline, brief: dict[str, Any], overlay: dict[str, Any]) -> _Round:
        """Merge ``overlay`` into the baseline and run the deterministic checks on the result."""
        spec, errors = merge(base, overlay, brief=brief)
        if spec is None:
            return _Round(None, overlay, {}, [f"STRUCTURE {e}" for e in errors])
        records: list[dict[str, Any]] = []
        definitions = [base.variables[c] | {"name": spec.columns[c].column} for c in spec.delivered if c in base.variables]
        aggregational = str(brief.get("type_of_data") or "").lower() == "aggregational"
        problems, report = verify(spec, definitions, aggregational=aggregational, records_out=records,
                                  expectations=overlay.get("expectations") if isinstance(overlay.get("expectations"), dict) else None)
        return _Round(spec if not problems else None, overlay, report, problems, records)

    def _round(self, base: Baseline, brief: dict[str, Any], prompt: str, current: dict[str, Any] | None, *,
               final: bool = False, expectations: dict[str, Any] | None = None) -> _Round:
        """One model call (a full spec, or a patch of ``current``), merged, checked and, where that is safe, healed."""
        raw = self._generate(SYSTEM_PROMPT, prompt)
        overlay = apply_patch(current, raw) if current is not None else (dict(raw) if isinstance(raw, dict) else {})
        carried = expectations if expectations is not None else (current or {}).get("expectations")
        overlay.pop("expectations", None)                  # the author does not write the expectations: they are the independent view
        if carried is not None:
            overlay["expectations"] = carried
        rnd = self._verified(base, brief, overlay)
        if rnd.spec is None:
            rnd = self._heal_soft(base, brief, rnd, final=final)
        elif any(f["id"] == "window_overlap" for f in rnd.report.get("advice") or []):
            rnd = self._heal_cadence(base, brief, rnd)
        return rnd

    def _expectations(self, base: Baseline, brief: dict[str, Any], notes: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
        """The independent expectations of the scenario, or None when the model could not be asked (which never blocks a spec).

        What is asked depends only on the scenario and the column definitions, so a design that is repeated (a failed attempt tried
        again, the same columns under another request) reuses the answer instead of asking again.
        """
        columns = {cid for cid, col in base.spec.columns.items() if col.kind != "latent"}
        prompt = expect.build_prompt(brief, column_cards(base, notes), events_per_entity(brief), SCENARIO_KEYS)
        key = hashlib.sha256((expect.PROMPT_HASH + prompt).encode()).hexdigest()
        with _EXPECTATION_LOCK:
            cached = _EXPECTATION_CACHE.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        try:
            raw = self._generate(expect.EXPECTATION_PROMPT, prompt, attempts=2)
        except Exception as exc:
            logger.warning("behaviour expectations unavailable (%s: %s)", type(exc).__name__, _short(str(exc), 160))
            return None
        found = expect.parse(raw, columns, entity=base.spec.entity_column)
        with _EXPECTATION_LOCK:
            _EXPECTATION_CACHE[key] = copy.deepcopy(found)
            while len(_EXPECTATION_CACHE) > _EXPECTATION_CACHE_SIZE:
                _EXPECTATION_CACHE.pop(next(iter(_EXPECTATION_CACHE)))
        logger.info("behaviour expectations: %d rule(s), %d entity fact(s), %d state share(s)", len(found.get("rules") or []),
                    len(found.get("entity_facts") or []), len(found.get("state_shares") or []))
        return found

    def _heal_soft(self, base: Baseline, brief: dict[str, Any], rnd: _Round, *, final: bool) -> _Round:
        """Repair what can be repaired without the model: a column that is empty on every row and a target the data misses.

        A target is the author's expectation of a share or median, a condition that never holds is the author's mistake about
        when a fact exists; neither is a reason to lose the whole design. The column then carries values from its own
        definition, the target is dropped, and the change is reported as a finding so that the refinement can model it properly.
        Only when nothing else is wrong (or the repair rounds are spent) - otherwise the author gets to fix it itself first.
        """
        soft = [p for p in rnd.problems if p.startswith(("EMPTY ", "TARGET "))]
        if not soft or not rnd.overlay or (not final and len(soft) != len(rnd.problems)):
            return rnd
        overlay = copy.deepcopy(rnd.overlay)
        notes: list[dict[str, Any]] = []
        current = rnd
        for step in range(2):
            empty = list(current.report.get("empty_columns") or [])
            failed = set(current.report.get("failed_targets") or [])
            if not empty and not failed:
                break
            columns = overlay.get("columns") if isinstance(overlay.get("columns"), list) else []
            for cid in empty:
                entry = next((c for c in columns if isinstance(c, dict) and c.get("name") == cid), None)
                if entry is None:
                    continue
                if entry.get("when") and step == 0:
                    entry["when"] = None
                    what = "its condition never held in this scenario and was dropped"
                else:
                    columns.remove(entry)
                    what = "its behaviour left it empty and was replaced by its definition"
                notes.append({"id": "healed_empty_column", "severity": "warn", "columns": [cid],
                              "problem": f"Column {cid} was empty on every row; {what}.", "evidence": "",
                              "fix": "Model when this fact really exists and make that situation occur in this scenario."})
            if failed and isinstance(overlay.get("targets"), list):
                overlay["targets"] = [t for t in overlay["targets"] if not (isinstance(t, dict) and t.get("id") in failed)]
                for tid in sorted(failed):
                    notes.append({"id": "healed_target", "severity": "warn", "columns": [],
                                  "problem": f"Target {tid} did not hold on the simulated data and was dropped.", "evidence": "",
                                  "fix": "Either change the behaviour so the expectation holds or state a realistic expectation."})
            current = self._verified(base, brief, overlay)
            if current.spec is not None:
                current.report["advice"] = notes + list(current.report.get("advice") or [])
                current.overlay["healed"] = notes             # kept with the design so the refinement knows what to model properly
                return current
        return rnd

    def _heal_cadence(self, base: Baseline, brief: dict[str, Any], rnd: _Round) -> _Round:
        """Space an entity's events by the windows they open when the measurements show the windows overlap.

        The measured overlap needs no model: the next event may not start before the previous window ends, and the history
        is made long enough to hold that many events of that kind. Used only when the author set no spacing rule of its own
        and only when the simulation shows the result is feasible and clearly better.
        """
        from synth.engines.rows import MAX_HISTORY_DAYS

        issue = next((f for f in rnd.report.get("advice") or [] if f["id"] == "window_overlap" and f.get("ids")), None)
        history = rnd.overlay.get("history") if isinstance(rnd.overlay.get("history"), dict) else {}
        if issue is None or history.get("earliest_next") or not rnd.spec:
            return rnd
        _, end_id = issue["ids"]
        events = int(rnd.spec.timeline.get("events") or events_per_entity(brief))
        if issue.get("ends_at_clock") and rnd.spec.clock:
            # The window ends when its own event happens, so nothing "after the previous end" can be asked of the next
            # event; the next event has to come a whole window later than the previous one.
            clock = rnd.spec.clock
            gap = float(math.ceil(issue.get("window_days_p95") or issue["window_days_mean"]))
            rule = f"add_days(prev['{clock}'], {gap:g}) if prev['{clock}'] is not None else None"
            span = gap
            what = f"at least {gap:g} days after the previous event (the window that ends at each event)"
        else:
            rule = f"prev['{end_id}'] if prev['{end_id}'] is not None else None"
            span = issue["window_days_mean"]
            what = f"after the previous {issue['columns'][1]}"
        days = min(float(MAX_HISTORY_DAYS), max(float(history.get("days") or 0.0), math.ceil(events * span * 1.25)))
        overlay = copy.deepcopy(rnd.overlay)
        overlay["history"] = {**history, "days": days, "earliest_next": rule}
        assumptions = list(overlay.get("assumptions") or [])
        assumptions.append(f"An entity's next event starts {what} (spacing rule added because the "
                           f"measured windows overlapped in {issue['share']:.0%} of consecutive events).")
        overlay["assumptions"] = assumptions[-40:]
        candidate = self._verified(base, brief, overlay)
        if candidate.spec is None:
            return rnd
        left = next((f for f in candidate.report.get("advice") or [] if f["id"] == "window_overlap"
                     and f.get("ids") == issue["ids"]), None)
        if left is not None and left["share"] > 0.5 * issue["share"]:
            return rnd
        return candidate

    def _all_findings(self, brief: dict[str, Any], base: Baseline, rnd: _Round, notes: dict[str, dict[str, Any]] | None,
                      resources: dict[str, str] | None, *, consult_reviewer: bool = True) -> tuple[list[dict[str, Any]], str]:
        """Measured advice plus the reviewer's findings for a spec that passed the deterministic checks.

        ``consult_reviewer=False`` skips the (slow) reviewer call when the measured findings alone already decide what happens next.
        """
        findings = list(rnd.report.get("advice") or [])
        review = "off"
        if self.reviewing() and rnd.spec is not None:
            if not consult_reviewer:
                review = "deferred"
            else:
                reviewed = self._review_issues(brief, base, rnd.spec, rnd.records, notes, resources, rnd.report.get("facts"))
                review = "skipped" if reviewed is None else "clean" if not reviewed else "findings"
                findings += reviewed or []
        findings.sort(key=lambda f: f["severity"] != "error")
        return findings, review

    def _repair_measured(self, base: Baseline, brief: dict[str, Any], first: str, rnd: _Round, started: float) -> _Round:
        """Patch rounds for the errors the checks measured on a spec that passed, before the spec is first served.

        Measured errors (a column that never varies, an in-flight state on old events, a relation the independent expectations
        say must hold) are certain, so waiting for the background review to find them only delays the fix. A patched spec
        replaces the current one only when it passes every check again and is better measured: no more measured errors, no
        more measured weight overall, and less left to repair. A candidate that is worse tells the author what it broke.
        """
        def due(r: _Round) -> list[dict[str, Any]]:
            found = [f for f in r.report.get("advice") or []
                     if is_measured(f) and (f["severity"] == "error" or f["id"] in DRAFT_REPAIRED_WARNINGS)]
            return sorted(found, key=lambda f: f["severity"] != "error")

        expectations = rnd.overlay.get("expectations") if isinstance(rnd.overlay.get("expectations"), dict) else None
        extra: list[str] = []
        for _ in range(DRAFT_REPAIR_ROUNDS):
            before = due(rnd)
            if not before or time.monotonic() - started > min(self._budget(), self._repair_window()):
                break                                      # a slow design is served as it is; the refinement patches it next
            try:
                cand = self._round(base, brief, repair_prompt(first, rnd.overlay, [review_problem(f) for f in before] + extra), rnd.overlay,
                                   expectations=expectations)
            except Exception as exc:
                logger.warning("behaviour spec draft repair skipped (%s: %s)", type(exc).__name__, exc)
                break
            if cand.spec is None:
                extra = [f"The previous attempt was rejected: {p}" for p in cand.problems[:4]]
                logger.info("behaviour spec draft repair: patch rejected (%d problem(s))", len(cand.problems))
                continue
            after = cand.report.get("advice") or []
            if _accepts(after, rnd.report.get("advice") or []) and _weight(due(cand)) < _weight(due(rnd)):
                logger.info("behaviour spec draft repair: %d -> %d finding(s) to repair", len(before), len(due(cand)))
                rnd, extra = cand, []
                continue
            known = {(f["id"], tuple(f["columns"])) for f in before}
            extra = [f"The previous attempt introduced a new problem: {review_problem(f)}" for f in due(cand)
                     if (f["id"], tuple(f["columns"])) not in known][:4]
            logger.info("behaviour spec draft repair kept the spec it had (%d finding(s) to repair)", len(before))
        return rnd

    def draft(self, variables: list[dict[str, Any]], brief: dict[str, Any], *, notes: dict[str, dict[str, Any]] | None = None,
              resources: dict[str, str] | None = None) -> CompileResult:
        """The first spec that passes the deterministic checks (what generation needs), with the measured advice attached."""
        try:
            base = build(variables, brief=brief)
        except SpecBuildError as exc:
            return CompileResult(None, "rejected", [str(exc)])
        expectations = self._expectations(base, brief, notes)
        first = build_prompt(brief, base, notes, resources, expectations)
        prompt, current, problems, rnd = first, None, [], None
        started = time.monotonic()
        for round_no in range(self.max_repairs + 1):
            if round_no and time.monotonic() - started > self._budget():
                logger.warning("behaviour spec draft: time budget spent after %d round(s)", round_no)
                break
            try:
                rnd = self._round(base, brief, prompt, current, final=round_no == self.max_repairs, expectations=expectations)
            except Exception as exc:                      # no key, provider outage, timeout, invalid JSON
                logger.warning("behaviour spec draft: model unavailable (%s: %s)", type(exc).__name__, exc)
                return CompileResult(None, "unavailable", [f"{type(exc).__name__}: {_short(str(exc), 240)}"], round_no)
            if rnd.spec is not None:
                rnd = self._repair_measured(base, brief, first, rnd, started)
                rnd.report.update(rounds=round_no, problems=0)
                findings = list(rnd.report.get("advice") or [])
                return CompileResult(rnd.spec, "compiled", [], round_no, rnd.overlay, rnd.report, findings)
            problems = rnd.problems
            logger.info("behaviour spec draft round %d: %d problem(s)", round_no, len(problems))
            current = rnd.overlay or None            # an empty reply has nothing to patch: ask again from the start
            prompt = repair_prompt(first, rnd.overlay, problems) if current else first
        return CompileResult(None, "rejected", problems, self.max_repairs, rnd.overlay if rnd else None)

    def refine(self, variables: list[dict[str, Any]], brief: dict[str, Any], current: CompileResult, *,
               notes: dict[str, dict[str, Any]] | None = None, resources: dict[str, str] | None = None,
               on_improve: Any = None) -> CompileResult | None:
        """A better spec than ``current`` (fewer and less serious findings), or None when none was found.

        The findings of ``current`` (measured advice, the reviewer's issues) go back to the author as a patch; each candidate
        must pass the deterministic checks again and is judged again. A candidate is kept only when nothing that was measured got
        worse (no new measured error, no heavier measured findings) and the total left wrong is lighter or the measured part is.
        """
        if current.spec is None or current.overlay is None:
            return None
        try:
            base = build(variables, brief=brief)
        except SpecBuildError:
            return None
        started = time.monotonic()
        stored = current.overlay.get("expectations")
        expectations = stored if isinstance(stored, dict) else self._expectations(base, brief, notes)
        start_overlay = dict(current.overlay)
        if expectations is not None:
            start_overlay["expectations"] = expectations
        # the reviewer reads the spec that is being improved, on its own simulation
        probe = self._round_for(base, brief, current.spec, start_overlay)
        # measured findings that call for a patch go to the author at once; the reviewer reads the patched spec instead of this one
        urgent = _weight([f for f in probe.report.get("advice") or []
                          if is_measured(f) and (f["severity"] == "error" or f["id"] in DRAFT_REPAIRED_WARNINGS)]) > GOOD_ENOUGH_WEIGHT
        findings, review = self._all_findings(brief, base, probe, notes, resources, consult_reviewer=not urgent)
        healed = [h for h in current.overlay.get("healed") or [] if isinstance(h, dict) and h.get("problem")]
        findings = sorted(healed + findings, key=lambda f: f["severity"] != "error")
        best = (_weight(findings), current.spec, start_overlay, current.report, findings, review)
        first = build_prompt(brief, base, notes, resources, expectations)
        overlay, rounds = {k: v for k, v in start_overlay.items() if k != "healed"}, 0
        counted = 0                                        # rounds that produced a spec to judge; a rejected patch is a second chance, not a round
        while counted < self.refine_rounds and rounds < self.refine_rounds + REJECTED_PATCH_ALLOWANCE:
            if _weight(findings) <= GOOD_ENOUGH_WEIGHT or time.monotonic() - started > self._refine_budget():
                break
            rounds += 1
            try:
                rnd = self._round(base, brief, repair_prompt(first, overlay, [review_problem(f) for f in findings]), overlay,
                                  expectations=expectations)
            except Exception as exc:
                logger.warning("behaviour spec refinement: model unavailable (%s: %s)", type(exc).__name__, exc)
                break
            if rnd.spec is None:                           # the patch broke a deterministic check: tell the author, keep the old spec
                overlay = rnd.overlay if rnd.overlay else overlay
                findings = [{"id": "check", "severity": "error", "columns": [], "problem": p, "evidence": "", "fix": ""} for p in rnd.problems[:8]] + findings
                logger.info("behaviour spec refinement round %d: patch rejected (%d problem(s))", rounds, len(rnd.problems))
                continue
            counted += 1
            # a candidate that made something measured worse is not going to be kept: the reviewer need not read it
            worse = _measured_errors(rnd.report.get("advice") or []) > _measured_errors(best[4]) \
                or _measured_weight(rnd.report.get("advice") or []) > _measured_weight(best[4])
            candidate, review = self._all_findings(brief, base, rnd, notes, resources, consult_reviewer=not worse)
            overlay = rnd.overlay
            weight = _weight(candidate)
            logger.info("behaviour spec refinement round %d: %d finding(s), weight %d (best %d)", rounds, len(candidate), weight, best[0])
            # The reviewer reads the same data differently on every call, so only what was measured is compared strictly; the
            # reviewer's findings count for the total, which must then be lighter - or the measured part must be.
            if _accepts(candidate, best[4], total=True):
                best = (weight, rnd.spec, rnd.overlay, rnd.report, candidate, review)
                if on_improve is not None:                  # whoever waits for the spec gets each improvement as soon as it exists
                    report = dict(rnd.report)
                    report.update(rounds=int(current.report.get("rounds", 0)) + rounds, review=review)
                    try:
                        on_improve(CompileResult(rnd.spec, "compiled", [], int(report["rounds"]), rnd.overlay, report, candidate))
                    except Exception:
                        logger.exception("could not publish a refined generation spec")
            findings = candidate
        if best[1] is current.spec:
            current.findings = best[4]
            current.report["review"] = best[5]
            return None
        report = dict(best[3])
        report.update(rounds=int(current.report.get("rounds", 0)) + rounds, review=best[5])
        return CompileResult(best[1], "compiled", [], int(report["rounds"]), best[2], report, best[4])

    def _round_for(self, base: Baseline, brief: dict[str, Any], spec: GenerationSpec, overlay: dict[str, Any]) -> _Round:
        """Re-simulate an accepted spec (without a model call) to get the rows and facts a review needs."""
        records: list[dict[str, Any]] = []
        definitions = [base.variables[c] | {"name": spec.columns[c].column} for c in spec.delivered if c in base.variables]
        aggregational = str(brief.get("type_of_data") or "").lower() == "aggregational"
        _, report = verify(spec, definitions, aggregational=aggregational, records_out=records,
                           expectations=overlay.get("expectations") if isinstance(overlay.get("expectations"), dict) else None)
        return _Round(spec, overlay, report, [], records)

    def compile(self, variables: list[dict[str, Any]], brief: dict[str, Any], *, notes: dict[str, dict[str, Any]] | None = None,
                resources: dict[str, str] | None = None) -> CompileResult:
        """``draft`` followed by ``refine``: the best spec this compiler can produce, all in the caller's time."""
        result = self.draft(variables, brief, notes=notes, resources=resources)
        if result.spec is None:
            return result
        better = self.refine(variables, brief, result, notes=notes, resources=resources) if (self.reviewing() or result.findings) else None
        final = better or result
        return finalize(final)


def finalize(result: CompileResult, revision: int = 0) -> CompileResult:
    """The result with its spec carrying the findings that remain (as ``warnings``) and its revision number."""
    if result.spec is None:
        return result
    warnings = [finding_text(f) for f in result.findings]
    result.spec = result.spec.model_copy(update={"warnings": warnings, "revision": revision})
    result.report["review_findings"] = result.findings
    return result


# ------------------------------------------------------------------------------------------------------------------
def _reads(entry: Emit, known: set[str]) -> set[str]:
    names: set[str] = set()
    for src in (entry.when, entry.expr):
        if src:
            names |= Expr(src).names
    if entry.sample is not None:
        names |= sampler_names(entry.sample)
    return names & known


def restrict(spec: GenerationSpec, keep: set[str]) -> GenerationSpec:
    """The same spec for fewer delivered columns (``keep``: column names).

    A removed column that a kept column still reads stays as a hidden driver (simulated, not delivered); every other
    removed column disappears together with the rules that mention it.
    """
    kept = {cid for cid, name in spec.delivered.items() if name in keep}
    kept |= {cid for cid, c in spec.columns.items() if c.kind == "latent"}
    if spec.entity_column:
        kept.add(spec.entity_column)
    emits = {e.column: e for e in spec.emit}
    known = set(spec.columns)
    needed, frontier = set(kept), list(kept)
    while frontier:
        for dep in _reads(emits[frontier.pop()], known):
            if dep not in needed:
                needed.add(dep)
                frontier.append(dep)
    columns: dict[str, SpecColumn] = {}
    for cid, col in spec.columns.items():
        if cid in needed:
            hidden = cid not in kept and col.kind != "latent"
            columns[cid] = col.model_copy(update={"kind": "latent", "column": None}) if hidden else col
    entries = []
    for e in spec.emit:
        if e.column in columns:
            entry = e.model_dump()
            entry["scope"] = e.scope
            entries.append(entry)
    removed = known - set(columns)

    def alive(src: str | None) -> bool:
        return not src or not (Expr(src).names & removed)

    order_by = spec.output.get("order_by")
    return spec.model_copy(update={
        "columns": columns, "emit": [Emit.model_validate(e) for e in order_entries(entries)],
        "invariants": [i for i in spec.invariants if alive(i.expr) and not (set(i.requires) & removed)],
        "targets": [t for t in spec.targets if alive(t.where) and alive(t.condition) and t.column not in removed],
        "clock": spec.clock if spec.clock in columns and columns[spec.clock].kind != "latent" else None,
        "output": {**spec.output, "order_by": order_by if order_by in columns else None},
        "design": {},                       # the designer's own spec no longer describes these columns: nothing to patch
    })
