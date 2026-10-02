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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from synth.baseline import Baseline, SpecBuildError, build, order_entries, spec_dtype
from synth.clock import RunContext, get_tz
from synth.contract import build_contract
from synth.engines import get_engine
from synth.expr import Expr
from synth.projection import parse_columns, project, style_of, time_resolution
from synth.samplers import sampler_names
from synth.scorer import score_rows
from synth.spec import EVAL_VARS, SPEC_VERSION, Emit, GenerationSpec, Invariant, SpecColumn, Target, safe_identifier

logger = logging.getLogger(__name__)

COMPILER_VERSION = 2
MAX_REPAIR_ROUNDS = 2
SCENARIO_KEYS = ("industry", "domain", "country", "use_case", "scenario_type", "business_scenario", "business_response",
                 "expected_outcome", "type_of_data", "entity_key")
REVIEW_ENTITIES = 5
REVIEW_EVENTS = 6
VERIFY_ENTITIES = 60
VERIFY_ROWS_PER_ENTITY = 6
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
- source_resources (when present): what the industry standard says each resource is, so a column is read in the meaning of the
  resource it belongs to (a "history" resource describes earlier actions on the main resource; it is not an unrelated record).
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

THE LANGUAGE
Expressions (python-like, safe subset). Variables: any column id or latent you declare (order is resolved automatically;
no cycles); event_at (this row's time), event_index (0-based position in the entity's history), first_event_at, as_of;
prev (the previous row of the same entity as a dict: prev['column'], or None on the first row); P['name'] (your "model"
parameters); REF['table'] (your "reference" tables).
Operators: + - * /  == != < <= > >=  and or not  is None / is not None  in / not in  (a if cond else b)  [lists]  x[key].
Functions: add_days(dt,n) add_secs(dt,n) secs(a,b)=seconds(a-b) days(a,b) text(x) lower(x) upper(x) int(x) float(x) abs min max
round(x,n) len sigmoid(x) matches(regex,str)=whole-string regex match is_none(x) hour_of(dt,offset_minutes).
Invariants and targets read the column ids, as_of, REF[...] and P[...] only (not event_at, event_index or prev).
Entity-level invariants only (variables become lists over the entity's rows, oldest first): distinct(list)
non_decreasing(list) strictly_increasing(list) all_after(starts,ends,applies) min_gap_secs(list).
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


def _short(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit - 1] + "…"


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


def build_prompt(brief: dict[str, Any], base: Baseline, notes: dict[str, dict[str, Any]] | None = None,
                 resources: dict[str, str] | None = None) -> str:
    scenario = {k: v for k, v in brief.items() if v not in (None, "") and k in SCENARIO_KEYS}
    spec = base.spec
    context: dict[str, Any] = {
        "scenario": scenario,
        "entity_column": spec.entity_column,
        "default_clock": spec.clock,
        "timestamp_precision": "minute" if time_resolution(spec) == 60 else "second",
    }
    if resources:
        context["source_resources"] = resources
    context["columns"] = column_cards(base, notes)
    return ("Design the behaviour spec for this scenario and these columns.\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":")))


def repair_prompt(first: str, overlay: dict[str, Any], problems: list[str]) -> str:
    return (first + "\n\nYOUR PREVIOUS SPEC:\n" + json.dumps(overlay, ensure_ascii=False, separators=(",", ":"))
            + "\n\nIT WAS SIMULATED AND CHECKED. THESE PROBLEMS MUST BE FIXED:\n- " + "\n- ".join(problems[:14])
            + "\n\nReturn the COMPLETE corrected spec (same format, one JSON object). Fix the root cause in the behaviour; "
              "do not remove an invariant or target merely to make it pass unless it is itself wrong.")


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
           per_entity: int = VERIFY_ROWS_PER_ENTITY, aggregational: bool = False,
           records_out: list[dict[str, Any]] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Simulate the spec and return ``(problems, report)``; no problems means the spec may be used.

    ``records_out`` (optional) receives the simulated rows as a client would see them, for review.
    """
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
    for cid, example in placeholders.items():
        problems.append(f"PLACEHOLDER column '{cid}' still draws neutral tokens such as '{example}'; give it realistic values for this "
                        "scenario (declare them in 'reference')")
    for cid, n in future.items():
        problems.append(f"FUTURE column '{cid}' has {n} value(s) later than as_of, but it records something that has already happened; "
                        "bound its delay, make it absent when it would be in the future, or list it in 'future_ok' if it is a planned/future fact")
    report["problems"] = len(problems)
    return problems, report


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
                        notes: dict[str, dict[str, Any]] | None = None, resources: dict[str, str] | None = None) -> str:
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


class SpecCompiler:
    """Drives the models: author a spec, simulate-and-check it, have it reviewed, repair; accept only a spec that passes.

    Two checks stand between a draft and the dataset. The deterministic one (structure, runtime, the definition
    contract, the author's own invariants and targets, placeholders, impossible future times) must pass for a spec to
    be accepted at all. The review is a second model that did not write the spec and reads a sample of the data it
    produces for what a person would see is wrong; its findings (errors first) go back to the author for the remaining repair
    rounds, the candidate with the fewest and least severe findings is kept, and whatever it still reports is recorded on
    the spec as a warning, because the deterministic checks have already passed.
    """

    def __init__(self, llm: Any | None = None, *, max_repairs: int = MAX_REPAIR_ROUNDS, review: bool | None = None):
        self._llm = llm
        self.max_repairs = max_repairs
        self._review = review

    @staticmethod
    def _budget() -> float:
        from config.runtime import SPEC_COMPILE_BUDGET_SECONDS

        return float(SPEC_COMPILE_BUDGET_SECONDS)

    def _reviewing(self) -> bool:
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
                       notes: dict[str, dict[str, Any]] | None, resources: dict[str, str] | None) -> list[dict[str, Any]] | None:
        """The reviewer's findings, or None when the reviewer could not be consulted (which never blocks a spec)."""
        try:
            raw = self._client().generate_json(system_instruction=REVIEW_PROMPT, temperature=0.0,
                                               user_prompt=build_review_prompt(brief, base, spec, records, notes, resources))
        except Exception as exc:
            logger.warning("behaviour spec review unavailable (%s: %s)", type(exc).__name__, exc)
            return None
        return parse_review(raw)

    def compile(self, variables: list[dict[str, Any]], brief: dict[str, Any], *, notes: dict[str, dict[str, Any]] | None = None,
                resources: dict[str, str] | None = None) -> CompileResult:
        try:
            base = build(variables, brief=brief)
        except SpecBuildError as exc:
            return CompileResult(None, "rejected", [str(exc)])
        aggregational = str(brief.get("type_of_data") or "").lower() == "aggregational"
        first = build_prompt(brief, base, notes, resources)
        prompt, overlay, problems = first, None, []
        best: tuple[int, GenerationSpec, dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None = None
        started = time.monotonic()
        for round_no in range(self.max_repairs + 1):
            if round_no and time.monotonic() - started > self._budget():
                logger.warning("behaviour spec compile: time budget spent after %d round(s)", round_no)
                break
            try:
                raw = self._client().generate_json(system_instruction=SYSTEM_PROMPT, user_prompt=prompt, temperature=0.0)
            except Exception as exc:                      # no key, provider outage, timeout, invalid JSON
                logger.warning("behaviour spec compile: model unavailable (%s: %s)", type(exc).__name__, exc)
                if best is not None:
                    break
                return CompileResult(None, "unavailable", [f"{type(exc).__name__}: {_short(str(exc), 240)}"], round_no)
            overlay = raw if isinstance(raw, dict) else {}
            spec, errors = merge(base, overlay, brief=brief)
            findings: list[dict[str, Any]] = []
            if spec is not None:
                records: list[dict[str, Any]] = []
                problems, report = verify(spec, [base.variables[c] | {"name": spec.columns[c].column} for c in spec.delivered if c in base.variables],
                                          aggregational=aggregational, records_out=records)
                if not problems:
                    report.update(rounds=round_no, problems=0)
                    reviewed = self._review_issues(brief, base, spec, records, notes, resources) if self._reviewing() else None
                    findings = reviewed or []
                    report["review"] = "skipped" if reviewed is None else "clean" if not findings else "findings"
                    weight = 3 * sum(f["severity"] == "error" for f in findings) + sum(f["severity"] != "error" for f in findings)
                    if best is None or weight < best[0]:
                        best = (weight, spec, report, overlay, findings)
                    if not findings:
                        break
                    problems = [review_problem(f) for f in sorted(findings, key=lambda f: f["severity"] != "error")]
            else:
                problems = [f"STRUCTURE {e}" for e in errors]
            logger.info("behaviour spec compile round %d: %d problem(s)", round_no, len(problems))
            prompt = repair_prompt(first, overlay, problems)
        if best is None:
            return CompileResult(None, "rejected", problems, self.max_repairs, overlay)
        _, spec, report, overlay, findings = best
        spec = spec.model_copy(update={"warnings": [f"{f['severity']}: {f['problem']} ({', '.join(f['columns'])})" for f in findings]})
        report["review_findings"] = findings
        return CompileResult(spec, "compiled", [], int(report.get("rounds", 0)), overlay, report)


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
    })
