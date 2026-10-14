"""What must hold for a scenario, stated independently of the spec that is supposed to deliver it.

A spec author who writes the rules *and* the checks that verify them only ever proves its own assumptions. This module is the
second opinion: a separate model call reads the scenario and the column definitions - never the author's spec - and states, in a
small closed vocabulary that means the same in every industry, which relations between columns a real system running that
scenario would always show:

* ``entity_facts``   columns that stay the same across all events of one entity;
* ``constants``      columns that legitimately carry one single value on every row;
* ``rules``          ``present_iff`` (a fact exists exactly when a condition holds), ``order`` (one moment is not before another),
                     ``at_most`` (a quantity does not exceed its bound), ``determined_by`` (a column is a function of others, such
                     as the attributes of one catalogue item), ``holds`` (any other condition every row satisfies) and
                     ``separate`` (two columns are different facts, never a relabelling or an exact complement of each other) and
                     ``slow_state`` (a status of the entity or its account that persists from one event to the next and changes only
                     on a rare, specific trigger, so it does not flip between most consecutive events);
* ``state_shares``   how common a state value is, as a wide band, for this scenario type.

``check`` measures every expectation on the simulated rows. A spec is then judged against something it did not write, and the
same machinery serves every industry because nothing in it names one.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime
from typing import Any

from synth.expr import Expr, ExprError
from synth.spec import GenerationSpec

MAX_RULES = 40
MAX_SHARES = 24
MIN_EVALUATED = 20                       # rows a rule must be evaluated on before a violation counts
MIN_VIOLATIONS = 5
VIOLATION_SHARE = 0.03
FUNCTIONAL_SHARE = 0.97                  # rows on which a determined column takes its group's commonest value
SHARE_TOLERANCE = 0.05
RULE_KINDS = ("present_iff", "order", "at_most", "determined_by", "holds", "separate", "slow_state")
SLOW_STATE_CHANGE_SHARE = 0.12            # a slow state may change on at most this share of an entity's consecutive events
MIN_TRANSITIONS = 60

EXPECTATION_PROMPT = """You are a domain expert writing the acceptance criteria for a synthetic dataset BEFORE it exists.
You receive a business scenario (industry, domain, country, use case, scenario type, scenario text), the number of events per
entity, and the columns of the dataset with their definitions. Say what a real system running exactly this scenario would
ALWAYS show in its records. You never write rows and you never see how the data will be generated.

The scenario type is a hard signal: write the expectations for THIS type (an outcome that is rare for one scenario type can be
common for another; a fact that exists for one type can be absent for another).

Use only the column ids you were given. Expressions are python-like over column ids and as_of (the moment the data is read;
every recorded fact is earlier than as_of): operators + - * /  == != < <= > >=  and or not  is None / is not None  in / not in
(a if c else b)  [lists]; functions abs min max round len lower upper text secs(a,b)=seconds(a-b) days(a,b) matches(regex,s)
hour_of(dt,offset_minutes) weekday_of(dt,offset_minutes) (0 = Monday ... 6 = Sunday; the offset in minutes turns the time into the local clock, e.g. 330 for UTC+5:30).
A column that can be empty must be guarded: "x is None or x > 0". No attribute access, comprehensions or lambda.

OUTPUT: one JSON object, nothing else:
{
 "entity_facts": ["column ids that describe the entity itself and are the same on every one of its events"],
 "constants": ["column ids that really carry one single value on every row: a fixed currency or country, or a kind/type column that the scenario itself fixes (a scenario about one product line has one kind) - never an outcome, a state or an amount"],
 "rules": [
  {"kind":"present_iff","column":"id","when":"<expr>","why":"..."},      the column has a value exactly when the condition is true, and is empty otherwise
  {"kind":"order","later":"id","earlier":"id","why":"..."},             when both have a value, later is not before earlier
  {"kind":"at_most","column":"id","bound":"id","why":"..."},            a quantity never exceeds the limit, total or capacity it is measured against
  {"kind":"determined_by","column":"id","by":["id"],"why":"..."},       the same values of "by" always give the same value of the column (the attributes of one catalogue item: its name determines its price, size, validity and kind)
  {"kind":"holds","expr":"<expr>","why":"..."},                         any other condition that is true on every row (a flag agrees with the state or timestamp that records the same step; a later step is never true while the step before it is false; an action that a flag forbids does not happen)
  {"kind":"separate","columns":["id","id"],"why":"..."},               two columns that are different facts: neither is a relabelling, a copy, or an exact sum/difference complement of the other (two scores about one subject, two dates of one entity, two statuses of different stages)
  {"kind":"slow_state","column":"id","why":"..."}                      a status of the customer or account (lifecycle, standing, tier, restriction) that stays the same from one event to the next and changes only on a rare trigger; NOT the outcome of each event (a transaction's own status), which changes freely
 ],
 "state_shares": [{"column":"id","value":"one value of a status/outcome/stage column","min":0.0,"max":1.0}],
 "trajectories": ["column ids whose value must CHANGE over the events of one entity because the scenario describes a progression"],
 "shared_parties": ["column ids that name ANOTHER party (an owner, controlling party, counterparty, beneficiary, linked account, device) which the scenario says is shared between entities"]
}

GUIDANCE
- Think in the classes of relations that every record system has, not in this scenario's wording: what exists only once something
  succeeded; what follows what in time; what is limited by what; what belongs to one item or party and so repeats with it; what a
  yes/no flag says about the columns that record the same step; what is one fact and what are two.
- A column whose meaning is "when something happened because of another step" is later than that step: write an order rule.
- Facts of the entity (identifier, segment, plan or tier it holds, profile dates) never change across its events; facts of an event
  (status, amount, timestamp of a step, channel) do. A fact that summarises the entity's events (latest, total, average) is not an
  entity fact unless the scenario fixes it.
- "state_shares" gives WIDE bands (at least 0.15 wide) for the outcome values of this scenario type, for example how common a
  failure, an acceptance or an escalation is. Use the exact values the column's allowed list gives (true or false for a yes/no column). Skip a column whose values
  you cannot judge. Never give a band for an identifier.
- Columns that the standard keeps in different resources (the card shows "resource" and "source_field") but that record one fact of
  one event - the amount of a payment block and the amount of the transaction that carries it - are meant to be equal: state it as
  "holds" (equal when both have a value), so the pair is declared related and is not mistaken for a copy.
- Real operating rules are expectations too. For the country and industry of the scenario, state as "holds" rules the limits that the
  rails, channels, instruments and services named by the columns really enforce: a per-transaction cap, a minimum for a high-value
  rail, whole denominations for cash, the fee that belongs to a fee type, a quota, a regulated contact window or opt-out. Condition
  each on the column that names the kind (for example the payment mechanism) so the rule binds only the rows of that kind. Also state
  that a party in a blocking state (frozen, closed, suppressed, opted out, barred) has no new outreach or activity after it began.
- For EVERY value of a column that names a rail, channel, instrument or product (read its "allowed" list), ask what limit the country's
  regulator or operator sets on it - a minimum or maximum amount, a window of hours, a settlement day, an eligibility - and state each one
  you are sure of as a "holds" rule conditioned on that value. An amount range on the column is only the outer envelope: a value with a
  floor (a high-value rail) must never be drawn below it, and a value with a cap never above it.
- A fact that records a step finishing (posting, settlement, clearing, delivery) is empty whenever the outcome column says the step did
  not succeed (failed, declined, rejected, cancelled, pending): state it as a "holds" rule on the outcome column.
- The scenario text (business scenario, business response, expected outcome, scenario type) is the catalogue's own statement of what
  the data must show: read it for what is NORMAL and what is the EXCEPTION. A baseline, benchmark, legitimate, everyday or "normal
  behaviour" scenario is mostly uneventful: every column that records an intervention or control decision (block, deny, reject,
  hold, escalate, review, restrict, warn) gets a "state_shares" band that keeps those values rare (at most 0.05 each) and the
  ordinary value (notify, allow, approve, pass) at 0.8 or more. A scenario about breaches, anomalies, fraud or deterioration
  inverts that for the stage it describes.
- When the scenario text describes a progression (behaviour that evolves, escalates, deteriorates, grows or is lost over time), list in
  "trajectories" the columns that carry that progression (a risk rating or score, an alert or severity level, a stage, a balance).
  A fact the scenario wants to evolve must not be fixed per entity. Do not list identifiers, segments or anything the entity keeps.
- When the scenario is about a network, ring, group, common owner or shared counterparty (mule accounts, shell companies, related
  parties, a household), list in "shared_parties" the columns that name the other party: several entities must point at the same
  party. Leave it empty when the scenario is about independent customers. "country_limits" in the scenario (when present) are the
  stored floors, caps and fees of the country's payment rails: write no rule that contradicts them.
- The date or time of something that is calculated from events (an alert, an insight, an assessment, a rating, a score, a review) is
  not before the events it is calculated from: write an "order" rule between it and the time of the event it concerns.
- Where a payment, deposit or withdrawal block repeats the amount of the transaction that carries it, they are the same money: state
  that they are equal on every row where both exist ("holds"), and when a fee column exists, that the larger equals the smaller plus
  that fee. A recurring credit or debit (salary, pension, rent, instalment, subscription) of one entity keeps nearly the same amount
  from one occurrence to the next.
- Prefer a few rules you are sure of (at most 30) over many guesses: a rule that is wrong rejects correct data. Do not write a rule
  for a column pair that has no real relation. Every rule needs a one-sentence "why" that is true for this scenario.
- Do not restate a column's own definition (allowed values, ranges, formats): those are already checked. DO state what is true
  in the real world but a loose definition does not enforce, as "holds" rules you are sure of: an identifier in the format its
  country really issues (a phone number, tax or account number: use matches(regex, value) and allow an empty value when the
  column can be empty), a score or rating on the scale that country's institutions really use, a quantity that cannot be smaller
  than the smallest real unit of what it counts. Two columns that rate the same subject (a score and a risk rating) agree in
  direction but are not one fact: state it as a band ("holds" that a rating is within its band of the score, or that the
  rating's order follows the score's order), never as an exact formula, because two independent assessments differ by noise.
"""

PROMPT_HASH = hashlib.sha256(EXPECTATION_PROMPT.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------------------
def build_prompt(brief: dict[str, Any], cards: list[dict[str, Any]], events: int, scenario_keys: tuple[str, ...]) -> str:
    """The model-facing request: the scenario and compact columns (no spec, no behaviour)."""
    scenario = {k: v for k, v in brief.items() if v not in (None, "") and k in scenario_keys}
    slim = []
    for card in cards:
        item = {k: card[k] for k in ("id", "dtype", "description", "allowed", "min", "max", "nullable") if k in card}
        if isinstance(item.get("allowed"), list):
            item["allowed"] = item["allowed"][:25]
        slim.append(item)
    context = {"scenario": scenario, "events_per_entity": events, "columns": slim}
    return "Write the acceptance criteria for this scenario.\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)


def _ids(raw: Any, known: set[str]) -> list[str]:
    return [str(x) for x in raw if str(x) in known] if isinstance(raw, list) else []


def _expr(source: Any, known: set[str]) -> str | None:
    if not isinstance(source, str) or not source.strip():
        return None
    try:
        names = Expr(source).names
    except ExprError:
        return None
    return source.strip() if names <= known | {"as_of"} else None


def parse(raw: Any, columns: set[str], *, entity: str | None = None) -> dict[str, Any]:
    """The expectations in ``raw`` that name real columns and carry valid expressions; everything else is dropped.

    ``columns`` are the delivered column ids. The result is plain JSON, stored with the design and reused by every later check.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    facts = [c for c in _ids(raw.get("entity_facts"), columns) if c != entity]
    if facts:
        out["entity_facts"] = sorted(set(facts))
    constants = _ids(raw.get("constants"), columns)
    if constants:
        out["constants"] = sorted(set(constants))
    rules: list[dict[str, Any]] = []
    for item in raw.get("rules") if isinstance(raw.get("rules"), list) else []:
        if not isinstance(item, dict) or item.get("kind") not in RULE_KINDS:
            continue
        kind, why = item["kind"], str(item.get("why") or "").strip()[:240]
        rule: dict[str, Any] | None = None
        if kind == "present_iff":
            when = _expr(item.get("when"), columns)
            if str(item.get("column")) in columns and when:
                rule = {"kind": kind, "column": str(item["column"]), "when": when}
        elif kind == "order":
            later, earlier = str(item.get("later")), str(item.get("earlier"))
            if later in columns and earlier in columns and later != earlier:
                rule = {"kind": kind, "later": later, "earlier": earlier}
        elif kind == "at_most":
            column, bound = str(item.get("column")), str(item.get("bound"))
            if column in columns and bound in columns and column != bound:
                rule = {"kind": kind, "column": column, "bound": bound}
        elif kind == "determined_by":
            column, by = str(item.get("column")), [b for b in _ids(item.get("by"), columns) if b != str(item.get("column"))]
            if column in columns and by and len(by) <= 3:
                rule = {"kind": kind, "column": column, "by": by}
        elif kind == "holds":
            expr = _expr(item.get("expr"), columns)
            if expr:
                rule = {"kind": kind, "expr": expr}
        elif kind == "slow_state":
            if str(item.get("column")) in columns and str(item.get("column")) != entity:
                rule = {"kind": kind, "column": str(item["column"])}
        elif kind == "separate":
            pair = _ids(item.get("columns"), columns)
            if len(pair) == 2 and pair[0] != pair[1]:
                rule = {"kind": kind, "columns": pair}
        if rule is not None and rule not in [{k: v for k, v in r.items() if k != "why"} for r in rules]:
            rules.append({**rule, "why": why})
    if rules:
        out["rules"] = rules[:MAX_RULES]
    shares = []
    for item in raw.get("state_shares") if isinstance(raw.get("state_shares"), list) else []:
        try:
            column, value = str(item.get("column")), str(item.get("value"))
            lo, hi = float(item.get("min")), float(item.get("max"))
        except (AttributeError, TypeError, ValueError):
            continue
        if column in columns and 0.0 <= lo <= hi <= 1.0 and hi - lo >= 0.1:
            shares.append({"column": column, "value": value, "min": round(lo, 3), "max": round(hi, 3)})
    if shares:
        out["state_shares"] = shares[:MAX_SHARES]
    moving = [c for c in _ids(raw.get("trajectories"), columns) if c != entity]
    if moving:
        out["trajectories"] = sorted(set(moving))[:8]
    linked = [c for c in _ids(raw.get("shared_parties"), columns) if c != entity]
    if linked:
        out["shared_parties"] = sorted(set(linked))[:6]
    return out


def declared_pairs(expectations: dict[str, Any] | None) -> tuple[set[frozenset[str]], set[frozenset[str]]]:
    """``(related, separate)``: the column pairs the expectations relate on purpose, and those they declare to be different facts.

    Two columns that appear together in one rule (a flag and the state it agrees with, a reason that is present when a status
    holds, an item and its price) are related by design and are not "mirrored"; a ``separate`` pair is the opposite.
    """
    related: set[frozenset[str]] = set()
    separate: set[frozenset[str]] = set()
    for rule in (expectations or {}).get("rules") or []:
        if rule.get("kind") == "separate":
            separate.add(frozenset(rule.get("columns") or []))
            continue
        used = [rule.get(k) for k in ("column", "later", "earlier", "bound") if rule.get(k)] + list(rule.get("by") or [])
        for src in (rule.get("when"), rule.get("expr")):
            if src:
                try:
                    used += sorted(Expr(src).names - {"as_of"})
                except ExprError:
                    pass
        for i, a in enumerate(used):
            for b in used[i + 1:]:
                if a != b:
                    related.add(frozenset((a, b)))
    return related, separate


def signature(expectations: dict[str, Any] | None) -> str:
    return hashlib.sha256(json.dumps(expectations or {}, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------------------
def _dominated(values: list[Any]) -> bool:
    counts: dict[str, int] = {}
    for v in values:
        counts[repr(v)] = counts.get(repr(v), 0) + 1
    return max(counts.values()) > 0.95 * len(values)


def same_fact(pairs: list[tuple[Any, Any]]) -> str | None:
    """How two columns repeat each other on ``pairs`` of non-empty values, or None when they do not.

    ``identical`` (the same value on nearly every row), ``sum`` / ``difference`` (two numbers that add up to, or differ by, one
    constant - a score and its "complement"), ``one_to_one`` (two categorical columns that relabel each other). Independent
    facts, even correlated ones, do none of these; shares of one whole are meant to add up and are the caller's to exempt.
    """
    n = len(pairs)
    if n < 100 or _dominated([x for x, _ in pairs]) or _dominated([y for _, y in pairs]):
        return None                                        # a column that says one thing on nearly every row repeats (and is repeated) by accident
    if sum(1 for x, y in pairs if x == y) >= 0.97 * n:
        return "identical"
    numeric = all(isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool)
                  for x, y in pairs)
    if numeric:
        if len({round(x, 2) for x, _ in pairs}) < 8 or len({round(y, 2) for _, y in pairs}) < 8:
            return None                                    # a column with few values can add up to a constant by accident
        for what, fn in (("sum", lambda x, y: x + y), ("difference", lambda x, y: x - y)):
            counts: dict[float, int] = {}
            for x, y in pairs:
                key = round(fn(x, y), 2)
                counts[key] = counts.get(key, 0) + 1
            if max(counts.values()) >= 0.97 * n:
                return what
        return None
    if isinstance(pairs[0][0], datetime) or isinstance(pairs[0][1], datetime):
        return None
    ab: dict[str, dict[str, int]] = {}
    ba: dict[str, dict[str, int]] = {}
    for x, y in pairs:
        sx, sy = str(x), str(y)
        ab.setdefault(sx, {})[sy] = ab.setdefault(sx, {}).get(sy, 0) + 1
        ba.setdefault(sy, {})[sx] = ba.setdefault(sy, {}).get(sx, 0) + 1
    if 2 <= len(ab) <= 12 and len(ab) == len(ba) and sum(max(c.values()) for c in ab.values()) >= 0.97 * n \
            and sum(max(c.values()) for c in ba.values()) >= 0.97 * n:
        return "one_to_one"
    return None


def entity_pairs(spec: GenerationSpec, rows: list[dict[str, Any]], a: str, b: str) -> list[tuple[Any, Any]]:
    """One ``(a, b)`` pair per entity, for two entity-level columns (their rows repeat the same value, so rows would overcount)."""
    seen: dict[Any, tuple[Any, Any]] = {}
    for r in rows:
        if r.get(a) is not None and r.get(b) is not None:
            seen.setdefault(r.get(spec.entity_column), (r[a], r[b]))
    return list(seen.values())


def near_linear(pairs: list[tuple[Any, Any]]) -> float | None:
    """The Pearson correlation of two numeric columns when it is so close to +-1 that one is the other with a little noise.

    Meant for facts of the entity (two scores, two ratings), measured on one pair per entity: two such facts about one subject
    that move together this exactly are one measure written twice (a rating that is a constant minus the score).
    """
    nums = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool)]
    if len(nums) < 15 or len({round(x, 2) for x, _ in nums}) < 10 or len({round(y, 2) for _, y in nums}) < 10:
        return None
    try:
        r = statistics.correlation([x for x, _ in nums], [y for _, y in nums])
    except statistics.StatisticsError:
        return None
    return r if abs(r) >= 0.98 else None


def _short(value: Any, limit: int = 60) -> str:
    text = value.isoformat() if isinstance(value, datetime) else repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _example(row: dict[str, Any], names: dict[str, str], ids: list[str]) -> str:
    return "{" + ", ".join(f"{names.get(c, c)}: {_short(row.get(c))}" for c in ids) + "}"


def _flags(bad: int, seen: int) -> bool:
    return seen >= MIN_EVALUATED and bad >= MIN_VIOLATIONS and bad >= VIOLATION_SHARE * seen


def _finding(kind: str, severity: str, names: list[str], problem: str, evidence: str, fix: str) -> dict[str, Any]:
    return {"id": f"expected_{kind}", "severity": severity, "columns": names, "problem": problem, "evidence": evidence, "fix": fix}


def _env(row: dict[str, Any], names: set[str], as_of: datetime) -> dict[str, Any]:
    env = {n: row.get(n) for n in names}
    env["as_of"] = as_of
    return env


def _holds(expr: Expr, rows: list[dict[str, Any]], as_of: datetime) -> tuple[list[dict[str, Any]], int]:
    """Rows on which ``expr`` is false, and how many rows it could be evaluated on (a guard that cannot be evaluated is skipped)."""
    bad, seen = [], 0
    for r in rows:
        try:
            ok = bool(expr(_env(r, expr.names, as_of)))
        except (TypeError, ValueError, ZeroDivisionError, KeyError, OverflowError):
            continue
        seen += 1
        if not ok:
            bad.append(r)
    return bad, seen


def check(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any] | None, as_of: datetime) -> list[dict[str, Any]]:
    """Findings (same shape as a review's) for every expectation the simulated ``rows`` break."""
    if not expectations or not rows:
        return []
    names = spec.delivered
    known = set(names)
    out: list[dict[str, Any]] = []

    for rule in expectations.get("rules") or []:
        kind = rule.get("kind")
        used = [rule.get(k) for k in ("column", "later", "earlier", "bound")] + list(rule.get("by") or []) + list(rule.get("columns") or [])
        if any(u not in known for u in used if u):
            continue                                       # the rule speaks of a column this dataset does not deliver
        why = f" ({rule['why']})" if rule.get("why") else ""
        if kind == "present_iff":
            col, source = rule["column"], rule["when"]
            try:
                cond = Expr(source)
            except ExprError:
                continue
            extra = cond.names - known - {"as_of"}
            if extra:
                continue
            present_wrong: list[dict[str, Any]] = []
            absent_wrong: list[dict[str, Any]] = []
            seen = 0
            for r in rows:
                try:
                    holds = bool(cond(_env(r, cond.names, as_of)))
                except (TypeError, ValueError, ZeroDivisionError, KeyError, OverflowError):
                    continue
                seen += 1
                has = r.get(col) is not None
                if has and not holds:
                    present_wrong.append(r)
                elif holds and not has:
                    absent_wrong.append(r)
            for wrong, what, fix in ((present_wrong, "has a value although", "Leave it empty whenever the condition is false."),
                                     (absent_wrong, "is empty although", "Give it a value on every row where the condition is true.")):
                if _flags(len(wrong), seen):
                    cols = [names[col]] + [names[c] for c in sorted(cond.names & known) if c != col][:4]
                    out.append(_finding(kind, "error", cols,
                                        f"{names[col]} {what} [{source}] {'is false' if wrong is present_wrong else 'holds'} on {len(wrong)} of {seen} rows{why}.",
                                        f"e.g. {_example(wrong[0], names, [col] + sorted(cond.names & known)[:4])}", fix))
        elif kind == "order":
            later, earlier = rule["later"], rule["earlier"]
            bad, seen = [], 0
            for r in rows:
                a, b = r.get(later), r.get(earlier)
                if a is None or b is None:
                    continue
                try:
                    early = a < b
                except TypeError:
                    continue
                seen += 1
                if early:
                    bad.append(r)
            if _flags(len(bad), seen):
                out.append(_finding(kind, "error", [names[later], names[earlier]],
                                    f"{names[later]} is before {names[earlier]} on {len(bad)} of {seen} rows{why}.",
                                    f"e.g. {_example(bad[0], names, [later, earlier])}",
                                    f"Draw {names[later]} from {names[earlier]} with a delay of at least one timestamp unit."))
        elif kind == "at_most":
            col, bound = rule["column"], rule["bound"]
            bad, seen = [], 0
            for r in rows:
                a, b = r.get(col), r.get(bound)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or isinstance(a, bool) or isinstance(b, bool):
                    continue
                seen += 1
                if a > b + 1e-9 * max(1.0, abs(b)):
                    bad.append(r)
            if _flags(len(bad), seen):
                out.append(_finding(kind, "error", [names[col], names[bound]],
                                    f"{names[col]} exceeds {names[bound]} on {len(bad)} of {seen} rows{why}.",
                                    f"e.g. {_example(bad[0], names, [col, bound])}",
                                    f"Derive {names[col]} from {names[bound]} (a share or a remainder of it), never as a separate draw."))
        elif kind == "determined_by":
            col, by = rule["column"], rule["by"]
            groups: dict[tuple, dict[Any, int]] = {}
            for r in rows:
                key = tuple(r.get(b) for b in by)
                if r.get(col) is None or any(k is None for k in key):
                    continue
                value = round(r[col], 6) if isinstance(r[col], float) else r[col]
                groups.setdefault(key, {})[value] = groups.setdefault(key, {}).get(value, 0) + 1
            shared = [g for g in groups.values() if sum(g.values()) >= 2]
            seen = sum(sum(g.values()) for g in shared)
            agree = sum(max(g.values()) for g in shared)
            if seen >= 30 and agree < FUNCTIONAL_SHARE * seen:
                split = max(shared, key=lambda g: sum(g.values()) - max(g.values()))
                out.append(_finding(kind, "error", [names[col]] + [names[b] for b in by],
                                    f"{names[col]} is not a function of {', '.join(names[b] for b in by)}: only {agree / seen:.0%} of {seen} rows carry the "
                                    f"commonest {names[col]} of their {'/'.join(names[b] for b in by)}{why}.",
                                    f"e.g. one {names[by[0]]} appears with {_short(sorted(map(str, split)), 90)}",
                                    f"Draw the item (or party) once into a latent and read {names[col]} from it with a switch or a REF table indexed by it."))
        elif kind == "holds":
            try:
                expr = Expr(rule["expr"])
            except ExprError:
                continue
            if expr.names - known - {"as_of"}:
                continue
            bad, seen = _holds(expr, rows, as_of)
            if _flags(len(bad), seen):
                ids = sorted(expr.names & known)[:5]
                out.append(_finding(kind, "error", [names[c] for c in ids],
                                    f"[{rule['expr']}] is false on {len(bad)} of {seen} rows{why}.",
                                    f"e.g. {_example(bad[0], names, ids)}",
                                    "Make the columns it relates agree, by deriving them from one hidden decision."))
        elif kind == "slow_state":
            col = rule["column"]
            histories: dict[Any, list[dict[str, Any]]] = {}
            for r in rows:
                histories.setdefault(r.get(spec.entity_column), []).append(r)
            moved = seen = 0
            example: list[Any] = []
            for g in histories.values():
                ordered = sorted(g, key=lambda r: r.get(spec.clock) or as_of) if spec.clock else g
                values = [r.get(col) for r in ordered if r.get(col) is not None]
                for x, y in zip(values, values[1:]):
                    seen += 1
                    if x != y:
                        moved += 1
                        if not example:
                            example = values[:5]
            if seen >= MIN_TRANSITIONS and moved > SLOW_STATE_CHANGE_SHARE * seen:
                out.append(_finding(kind, "error", [names[col]],
                                    f"{names[col]} is a state that persists but it changes between {moved} of {seen} consecutive events of the same entity{why}.",
                                    f"e.g. one entity: {_short(example, 100)}",
                                    f"Draw its starting value once per entity (an entity-scope latent) and let each event carry the previous value "
                                    f"(prev['{col}'] when it exists), changing it only on a rare, stated trigger."))
        elif kind == "separate":
            a, b = rule["columns"]
            pairs = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
            how = same_fact(pairs)
            if how is None and names[a] in spec.entity_columns and names[b] in spec.entity_columns:
                r = near_linear(entity_pairs(spec, rows, a, b))
                if r is not None:
                    out.append(_finding(kind, "error", [names[a], names[b]],
                                        f"{names[a]} and {names[b]} move together almost exactly (correlation {r:+.2f} across entities) although they are "
                                        f"different facts{why}.", f"{len(entity_pairs(spec, rows, a, b))} entities",
                                        "Give each its own hidden driver (correlated if related, but with its own variation)."))
            if how:
                said = {"identical": "carry the same value", "sum": "add up to one constant", "difference": "differ by one constant",
                        "one_to_one": "relabel each other one-to-one"}[how]
                out.append(_finding(kind, "error", [names[a], names[b]],
                                    f"{names[a]} and {names[b]} {said} on nearly every row although they are different facts{why}.",
                                    f"{len(pairs)} simulated rows",
                                    "Give each its own hidden driver (correlated if related, with its own variation) or derive them from different facts."))

    out += _entity_facts(spec, rows, expectations)
    out += _state_shares(spec, rows, expectations)
    out += _trajectories(spec, rows, expectations)
    out += _shared_parties(spec, rows, expectations)
    return out


SHARED_MIN_ENTITIES = 10
SHARED_MIN_SHARE = 0.3                    # at least this share of the entities must point at a party that another entity points at too


def _shared_parties(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings for a column naming a party the scenario says is shared (a network) when almost every entity has a party of its own."""
    columns = [c for c in expectations.get("shared_parties") or [] if c in spec.delivered and c != spec.entity_column]
    if not columns or not spec.entity_column:
        return []
    out = []
    for col in columns:
        owners: dict[Any, set[Any]] = {}
        for r in rows:
            value = r.get(col)
            if value is not None:
                owners.setdefault(value, set()).add(r.get(spec.entity_column))
        entities = set().union(*owners.values()) if owners else set()
        if len(entities) < SHARED_MIN_ENTITIES:
            continue
        sharing = {e for who in owners.values() if len(who) > 1 for e in who}
        if len(sharing) < SHARED_MIN_SHARE * len(entities):
            name = spec.delivered[col]
            out.append(_finding("shared_parties", "error", [name],
                                f"{name} names a party that the scenario says is shared between entities, but only {len(sharing)} of {len(entities)} entities "
                                f"point at a party that another entity also has ({len(owners)} distinct parties).",
                                f"{len(owners)} distinct values for {len(entities)} entities",
                                f"Draw {name} from a pool shared by all entities - the pool sampler with a ratio of about 0.3 members per entity (use the same pool "
                                "name for every column that names the same kind of party) - so that several entities share one party."))
    return out[:3]


TRAJECTORY_MIN_EVENTS = 3
TRAJECTORY_MIN_ENTITIES = 10
TRAJECTORY_MOVING_SHARE = 0.3             # at least this share of the entities must show a change for a column meant to evolve


def _trajectories(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings for a column the scenario wants to evolve (a risk rating that worsens, a stage that advances) that stays fixed per entity."""
    columns = [c for c in expectations.get("trajectories") or [] if c in spec.delivered and c != spec.entity_column]
    if not columns or not spec.entity_column:
        return []
    by_entity: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by_entity.setdefault(r.get(spec.entity_column), []).append(r)
    out = []
    for col in columns:
        eligible = moving = 0
        example: list[Any] = []
        for g in by_entity.values():
            ordered = sorted(g, key=lambda r: r.get(spec.clock) or datetime.min) if spec.clock else g
            values = [r.get(col) for r in ordered if r.get(col) is not None]
            if len(values) < TRAJECTORY_MIN_EVENTS:
                continue
            eligible += 1
            if len({repr(v) for v in values}) > 1:
                moving += 1
            elif not example:
                example = values[:4]
        if eligible >= TRAJECTORY_MIN_ENTITIES and moving < TRAJECTORY_MOVING_SHARE * eligible:
            name = spec.delivered[col]
            out.append(_finding("trajectory", "error", [name],
                                f"{name} is meant to evolve over the events of an entity (the scenario describes a progression) but it changes for only "
                                f"{moving} of {eligible} entities; for the others it carries one value on every event.",
                                f"e.g. one entity: {_short(example, 100)}",
                                f"Make {name} event scope and drive it from each entity's own progression (a latent for its start, speed and "
                                f"turning point, read through the event's position in time), so early events differ from late ones."))
    return out[:3]


def _entity_facts(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any]) -> list[dict[str, Any]]:
    facts = [c for c in expectations.get("entity_facts") or [] if c in spec.delivered and c != spec.entity_column]
    if not facts or not spec.entity_column:
        return []
    by_entity: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by_entity.setdefault(r.get(spec.entity_column), []).append(r)
    multi = [g for g in by_entity.values() if len(g) >= 2]
    if len(multi) < MIN_EVALUATED:
        return []
    out = []
    for col in facts:
        changing = [g for g in multi if len({repr(r.get(col)) for r in g}) > 1]
        if len(changing) >= MIN_VIOLATIONS and len(changing) >= VIOLATION_SHARE * len(multi):
            g = changing[0]
            out.append(_finding("entity_fact", "error", [spec.delivered[col]],
                                f"{spec.delivered[col]} describes the entity and must not change between its events, but it differs between events for "
                                f"{len(changing)} of {len(multi)} entities.",
                                f"e.g. {_short([r.get(col) for r in g][:4], 100)}",
                                "Make it an entity-scope fact drawn once per entity (scope \"entity\"), or derive it from an entity-level latent; "
                                "what changes per event belongs in another column."))
    return out


def _state_shares(spec: GenerationSpec, rows: list[dict[str, Any]], expectations: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in expectations.get("state_shares") or []:
        col = item["column"]
        if col not in spec.delivered:
            continue
        present = [r[col] for r in rows if r.get(col) is not None]
        if len(present) < 100:
            continue
        share = sum(1 for v in present if str(v).lower() == item["value"].lower()) / len(present)
        if share < item["min"] - SHARE_TOLERANCE or share > item["max"] + SHARE_TOLERANCE:
            far = share > item["max"] + 0.10 or share < item["min"] - 0.10
            out.append(_finding("state_share", "error" if far else "warn", [spec.delivered[col]],
                                f"{spec.delivered[col]} is '{item['value']}' on {share:.0%} of {len(present)} rows; for this scenario type it is expected "
                                f"between {item['min']:.0%} and {item['max']:.0%}.",
                                f"{len(present)} simulated rows",
                                f"Set the probabilities behind {spec.delivered[col]} so that '{item['value']}' falls in that band "
                                f"(or state in 'assumptions' why this scenario differs)."))
    return out[:6]


def render(expectations: dict[str, Any] | None) -> dict[str, Any] | None:
    """The expectations as the author sees them: the same content, without bookkeeping."""
    return dict(expectations) if expectations else None
