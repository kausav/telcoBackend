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
import logging
import math
import re
import statistics
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from synth.baseline import Baseline, SpecBuildError, build, order_entries, spec_dtype
from synth.clock import RunContext, get_tz
from synth.contract import build_contract
from synth.engines import get_engine
from synth.expr import Expr
from synth.projection import parse_columns, project, style_of, time_resolution
from synth.samplers import is_rollup, sampler_names
from synth.scorer import score_rows
from synth.spec import EVAL_VARS, SPEC_VERSION, Emit, GenerationSpec, Invariant, SpecColumn, Target, safe_identifier

logger = logging.getLogger(__name__)

COMPILER_VERSION = 2
MAX_REPAIR_ROUNDS = 2
MAX_REFINE_ROUNDS = 4
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
   could still be in flight; older events have reached a final state.
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
                 resources: dict[str, str] | None = None) -> str:
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
    context["columns"] = column_cards(base, notes)
    return ("Design the behaviour spec for this scenario and these columns.\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":")))


def repair_prompt(first: str, overlay: dict[str, Any], problems: list[str]) -> str:
    return (first + "\n\nYOUR CURRENT SPEC:\n" + json.dumps({k: v for k, v in overlay.items() if k != "healed"}, ensure_ascii=False, separators=(",", ":"))
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
           records_out: list[dict[str, Any]] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Simulate the spec and return ``(problems, report)``; no problems means the spec may be used.

    The simulation has as many events per entity as the spec was designed for (``timeline.events``, at most
    ``VERIFY_MAX_EVENTS``; the history scales with the count, so spacing is the same), because rules about spacing and
    windows only mean something at the real number of events. ``records_out`` (optional) receives the simulated rows as
    a client would see them, for review; ``report["advice"]`` holds what the checks consider implausible without being
    impossible, and ``report["facts"]`` what a reviewer needs to know about the shape of the data.
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
        report["advice"] = cadence_advice(spec, rows) + constant_advice(spec, rows, variables) + sparse_advice(spec, rows)
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
                found.append((share, {
                    "id": "window_overlap", "severity": "warn", "columns": names, "ids": [a, b], "share": round(share, 3),
                    "window_days_mean": round(statistics.fmean(x for x in spans if x >= 0) / 86400.0, 1),
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


def constant_advice(spec: GenerationSpec, rows: list[dict[str, Any]], variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings for numeric columns that carry one single value on every row: a measure that never varies measures nothing."""
    n = len(rows)
    if n < 100:
        return []
    fixed = {str(v.get("name")) for v in variables
             if isinstance(v.get("params"), dict) and v["params"].get("min") is not None and v["params"].get("min") == v["params"].get("max")}
    out = []
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("integer", "float") or name in fixed:
            continue
        values = {r.get(cid) for r in rows if r.get(cid) is not None}
        if len(values) == 1 and sum(1 for r in rows if r.get(cid) is not None) >= n // 2:
            out.append({"id": "constant_numeric_column", "severity": "error", "columns": [name],
                        "problem": f"Column {name} has the value {next(iter(values))!r} on every row of every entity.",
                        "evidence": f"{len(rows)} simulated rows, one distinct value",
                        "fix": "A numeric fact varies from entity to entity (and from event to event when it is measured per event): draw it from a "
                               "distribution, or compute it from the columns it is defined by with matching units so its magnitude is realistic."})
    return out[:3]


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
    return facts


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
and the 5th/50th/95th percentile of every numeric column ("numeric_spread"): judge whether those magnitudes are realistic for what the
column means (a time until depletion of one hour on nearly every row, an average of 0 for a quantity that is never 0, a count that
exceeds what the other columns allow). Judge each against what the column means: a fact that
only exists once something succeeded must be empty for every state that is not a success; a fact that applies must not be
empty; windows (validity, term, cooldown) should not still be open when the same entity's next event of that kind starts
unless the scenario renews them early; items (name, price, size, validity of one product or plan) must agree with each other;
the actor's role, identifier and channel must agree; statuses of one subject must not contradict each other.
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
        "compiler": [COMPILER_VERSION, SPEC_VERSION, hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]],
        "brief": {k: (brief.get(k) or "") for k in keys},
        "columns": sorted(cols, key=lambda c: str(c["name"])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:40]


def _weight(findings: list[dict[str, Any]]) -> int:
    """How much is still wrong: an impossible record weighs three implausible ones."""
    return 3 * sum(f["severity"] == "error" for f in findings) + sum(f["severity"] != "error" for f in findings)


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
        problems, report = verify(spec, definitions, aggregational=aggregational, records_out=records)
        return _Round(spec if not problems else None, overlay, report, problems, records)

    def _round(self, base: Baseline, brief: dict[str, Any], prompt: str, current: dict[str, Any] | None, *,
               final: bool = False) -> _Round:
        """One model call (a full spec, or a patch of ``current``), merged, checked and, where that is safe, healed."""
        raw = self._generate(SYSTEM_PROMPT, prompt)
        overlay = apply_patch(current, raw) if current is not None else (raw if isinstance(raw, dict) else {})
        rnd = self._verified(base, brief, overlay)
        if rnd.spec is None:
            rnd = self._heal_soft(base, brief, rnd, final=final)
        elif any(f["id"] == "window_overlap" for f in rnd.report.get("advice") or []):
            rnd = self._heal_cadence(base, brief, rnd)
        return rnd

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
        days = min(float(MAX_HISTORY_DAYS), max(float(history.get("days") or 0.0), math.ceil(events * issue["window_days_mean"] * 1.25)))
        overlay = copy.deepcopy(rnd.overlay)
        overlay["history"] = {**history, "days": days,
                              "earliest_next": f"prev['{end_id}'] if prev['{end_id}'] is not None else None"}
        assumptions = list(overlay.get("assumptions") or [])
        assumptions.append(f"An entity's next event starts after the previous {issue['columns'][1]} (spacing rule added because the "
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
                      resources: dict[str, str] | None) -> tuple[list[dict[str, Any]], str]:
        """Measured advice plus the reviewer's findings for a spec that passed the deterministic checks."""
        findings = list(rnd.report.get("advice") or [])
        review = "off"
        if self.reviewing() and rnd.spec is not None:
            reviewed = self._review_issues(brief, base, rnd.spec, rnd.records, notes, resources, rnd.report.get("facts"))
            review = "skipped" if reviewed is None else "clean" if not reviewed else "findings"
            findings += reviewed or []
        findings.sort(key=lambda f: f["severity"] != "error")
        return findings, review

    def draft(self, variables: list[dict[str, Any]], brief: dict[str, Any], *, notes: dict[str, dict[str, Any]] | None = None,
              resources: dict[str, str] | None = None) -> CompileResult:
        """The first spec that passes the deterministic checks (what generation needs), with the measured advice attached."""
        try:
            base = build(variables, brief=brief)
        except SpecBuildError as exc:
            return CompileResult(None, "rejected", [str(exc)])
        first = build_prompt(brief, base, notes, resources)
        prompt, current, problems, rnd = first, None, [], None
        started = time.monotonic()
        for round_no in range(self.max_repairs + 1):
            if round_no and time.monotonic() - started > self._budget():
                logger.warning("behaviour spec draft: time budget spent after %d round(s)", round_no)
                break
            try:
                rnd = self._round(base, brief, prompt, current, final=round_no == self.max_repairs)
            except Exception as exc:                      # no key, provider outage, timeout, invalid JSON
                logger.warning("behaviour spec draft: model unavailable (%s: %s)", type(exc).__name__, exc)
                return CompileResult(None, "unavailable", [f"{type(exc).__name__}: {_short(str(exc), 240)}"], round_no)
            if rnd.spec is not None:
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
        must pass the deterministic checks again and is judged again. The candidate with the least left wrong wins.
        """
        if current.spec is None or current.overlay is None:
            return None
        try:
            base = build(variables, brief=brief)
        except SpecBuildError:
            return None
        started = time.monotonic()
        # the reviewer reads the spec that is being improved, on its own simulation
        probe = self._round_for(base, brief, current)
        findings, review = self._all_findings(brief, base, probe, notes, resources)
        healed = [h for h in current.overlay.get("healed") or [] if isinstance(h, dict) and h.get("problem")]
        findings = sorted(healed + findings, key=lambda f: f["severity"] != "error")
        best = (_weight(findings), current.spec, current.overlay, current.report, findings, review)
        first = build_prompt(brief, base, notes, resources)
        overlay, rounds = {k: v for k, v in current.overlay.items() if k != "healed"}, 0
        for _ in range(self.refine_rounds):
            if _weight(findings) <= GOOD_ENOUGH_WEIGHT or time.monotonic() - started > self._refine_budget():
                break
            rounds += 1
            try:
                rnd = self._round(base, brief, repair_prompt(first, overlay, [review_problem(f) for f in findings]), overlay)
            except Exception as exc:
                logger.warning("behaviour spec refinement: model unavailable (%s: %s)", type(exc).__name__, exc)
                break
            if rnd.spec is None:                           # the patch broke a deterministic check: tell the author, keep the old spec
                overlay = rnd.overlay if rnd.overlay else overlay
                findings = [{"id": "check", "severity": "error", "columns": [], "problem": p, "evidence": "", "fix": ""} for p in rnd.problems[:8]] + findings
                logger.info("behaviour spec refinement round %d: patch rejected (%d problem(s))", rounds, len(rnd.problems))
                continue
            findings, review = self._all_findings(brief, base, rnd, notes, resources)
            overlay = rnd.overlay
            weight = _weight(findings)
            logger.info("behaviour spec refinement round %d: %d finding(s), weight %d (best %d)", rounds, len(findings), weight, best[0])
            if weight < best[0]:
                best = (weight, rnd.spec, rnd.overlay, rnd.report, findings, review)
                if on_improve is not None:                  # whoever waits for the spec gets each improvement as soon as it exists
                    report = dict(rnd.report)
                    report.update(rounds=int(current.report.get("rounds", 0)) + rounds, review=review)
                    try:
                        on_improve(CompileResult(rnd.spec, "compiled", [], int(report["rounds"]), rnd.overlay, report, findings))
                    except Exception:
                        logger.exception("could not publish a refined generation spec")
        if best[1] is current.spec:
            current.findings = best[4]
            current.report["review"] = best[5]
            return None
        report = dict(best[3])
        report.update(rounds=int(current.report.get("rounds", 0)) + rounds, review=best[5])
        return CompileResult(best[1], "compiled", [], int(report["rounds"]), best[2], report, best[4])

    def _round_for(self, base: Baseline, brief: dict[str, Any], current: CompileResult) -> _Round:
        """Re-simulate an accepted spec (without a model call) to get the rows and facts a review needs."""
        records: list[dict[str, Any]] = []
        definitions = [base.variables[c] | {"name": current.spec.columns[c].column} for c in current.spec.delivered if c in base.variables]
        aggregational = str(brief.get("type_of_data") or "").lower() == "aggregational"
        _, report = verify(current.spec, definitions, aggregational=aggregational, records_out=records)
        return _Round(current.spec, current.overlay or {}, report, [], records)

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
