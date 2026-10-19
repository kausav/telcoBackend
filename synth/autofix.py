"""Repairs of measured defects that need no model: the correct value follows from the data, so the spec is edited directly.

A finding such as "the balance does not move with the amount" is certain (measured on the simulation) and so is its cure: a balance
IS the previous balance plus or minus the event's amount. Asking the author model to rewrite it costs a call, and the model often
leaves it unfixed; the same cure written here is exact, universal (it reads only column names, values and the stored country
limits) and cheap. Every fix edits the *overlay* (the model-written layer of the spec), never the baseline, and the compiler keeps
it only when the checks run again on the result and find it better (``SpecCompiler._heal_measured``).

* ``balance_not_running``  - the balance becomes ``previous balance + credit - debit - fee`` with an opening balance drawn once per entity.
* ``debit_exceeds_balance``- a debit is reduced to a fraction of the balance that is there to pay it.
* ``derived_before_event`` - a date that summarises an entity's activity becomes a rollup (the latest event); a date of one event is never earlier than it.
* ``rail_limit_violated``  - amounts are brought inside the cap and floor of the payment rail the row names, free rails carry no fee.

Which events are credits is read from the data (the words of a category column, or the names of the amount columns filled for
deposits); when neither says, nothing is changed and the author is asked as before.
"""
from __future__ import annotations

import copy
import logging
import statistics
from collections import Counter
from typing import Any, Callable

from synth.spec import GenerationSpec, safe_identifier

logger = logging.getLogger(__name__)

CREDIT_WORDS = frozenset({"credit", "credited", "deposit", "deposits", "salary", "payroll", "refund", "inward", "incoming", "received", "receipt",
                          "cashback", "dividend", "reversal", "remittance", "inflow", "income", "topup", "recharge"})
DEBIT_WORDS = frozenset({"debit", "debited", "withdrawal", "withdrawals", "withdraw", "payment", "payments", "purchase", "outward", "outgoing", "bill",
                         "emi", "rent", "fee", "charge", "atm", "dispense", "payout", "repayment", "outflow", "expense", "spend"})
DIRECTION_ROW_SHARE = 0.8        # a category column names the direction when its values say it for at least this share of all rows
OVERDRAW_FRACTION = 0.9          # a debit that cannot be paid is reduced to this share of the balance that is there
OPENING_REACH = 12.0             # an opening balance is at least this many times the usual (upper quartile) amount the account moves
OPENING_FLOOR = 10.0             # the lowest opening balance, in upper-quartile amounts
OPENING_SIGMA = (0.25, 1.2)      # spread of the opening balances across entities, kept inside a realistic range


def _c() -> Any:
    from synth import compiler

    return compiler


# ------------------------------------------------------------------------------------------------------ overlay helpers
def _list(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if isinstance(raw, dict):
        return [{"name": k, **(v if isinstance(v, dict) else {"expr": v})} for k, v in raw.items()]
    return [dict(i) for i in raw if isinstance(i, dict)]


def _emit(spec: GenerationSpec, cid: str) -> Any:
    return next((e for e in spec.emit if e.column == cid), None)


def _fresh(spec: GenerationSpec, overlay: dict[str, Any], stem: str) -> str:
    taken = set(spec.columns) | {str(i.get("name")) for i in _list(overlay.get("latents"))}
    return safe_identifier(stem, taken)


def _put_column(overlay: dict[str, Any], cid: str, scope: str, *, expr: str | None = None, sample: dict[str, Any] | None = None) -> None:
    items = [i for i in _list(overlay.get("columns")) if str(i.get("name") or i.get("column")) != cid]
    items.append({"name": cid, "scope": scope, "when": None, "expr": expr, "sample": sample})
    overlay["columns"] = items


def _put_latent(overlay: dict[str, Any], name: str, dtype: str, scope: str, *, when: str | None = None, expr: str | None = None,
                sample: dict[str, Any] | None = None) -> None:
    items = _list(overlay.get("latents"))
    items.append({"name": name, "dtype": dtype, "scope": scope, "when": when, "expr": expr, "sample": sample,
                  "description": "added by the compiler's automatic repair"})
    overlay["latents"] = items


def _wrap(spec: GenerationSpec, overlay: dict[str, Any], cid: str, build: Callable[[str], str]) -> bool:
    """Move the column's own definition into a hidden latent and make the column a function of that latent (``build`` gets its name)."""
    from synth.samplers import is_rollup

    entry = _emit(spec, cid)
    if entry is None or is_rollup(entry.sample):
        return False
    raw = _fresh(spec, overlay, f"{cid}_raw")
    _put_latent(overlay, raw, spec.columns[cid].dtype, entry.scope, when=entry.when, expr=entry.expr, sample=copy.deepcopy(entry.sample))
    _put_column(overlay, cid, entry.scope, expr=build(raw))
    return True


def _num(r: dict[str, Any], cid: str) -> float | None:
    v = r.get(cid)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _name_to_cid(spec: GenerationSpec, name: str) -> str | None:
    return next((cid for cid, n in spec.delivered.items() if n == name), None)


# ------------------------------------------------------------------------------------------------------ direction of money
def direction(spec: GenerationSpec, rows: list[dict[str, Any]]) -> str | None:
    """An expression that is true for an event that puts money INTO the account, read from the data; None when the data does not say."""
    C = _c()
    total = len(rows)
    if not total:
        return None
    best: tuple[float, str, set[str]] | None = None
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("string", "categorical"):
            continue
        counts = Counter(v for r in rows for v in [r.get(cid)] if isinstance(v, str) and len(v) <= 60)
        if not counts or len(counts) > 40:
            continue
        credit: set[str] = set()
        debit: set[str] = set()
        explicit = 0
        for value, n in counts.items():
            words = C._value_words(value)
            c, d = bool(words & CREDIT_WORDS), bool(words & DEBIT_WORDS)
            if c and not d:
                credit.add(value)
                explicit += n
            elif d and not c:
                debit.add(value)
                explicit += n
        share = explicit / total
        if (credit or debit) and (best is None or share > best[0]):
            best = (share, cid, credit)
    if best is not None and best[0] >= DIRECTION_ROW_SHARE:
        return _is_in(best[1], best[2])
    amounts = C._movement_amounts(spec, rows)
    credit_cols = [c for c, n in amounts.items() if C._tokens(n) & CREDIT_WORDS and not C._tokens(n) & DEBIT_WORDS]
    debit_cols = [c for c, n in amounts.items() if C._tokens(n) & DEBIT_WORDS and not C._tokens(n) & CREDIT_WORDS]
    if credit_cols:
        def filled(r: dict[str, Any], cols: list[str]) -> bool:
            return any(_num(r, c) is not None for c in cols)

        both = sum(1 for r in rows if filled(r, credit_cols) and filled(r, debit_cols))
        some = sum(1 for r in rows if filled(r, credit_cols))
        if some and both <= 0.05 * total and some < 0.95 * total:
            # the guard that makes a deposit amount exist is the direction itself (and reads nothing that depends on the amount)
            guards = [f"({_emit(spec, c).when})" if _emit(spec, c) is not None and _emit(spec, c).when else f"{c} is not None" for c in credit_cols]
            return "(" + " or ".join(guards) + ")"
    if best is not None and best[0] >= 0.5:
        return _is_in(best[1], best[2])
    return None


def _is_in(cid: str, values: set[str]) -> str:
    if not values:
        return "(False)"
    return f"({cid} in ({', '.join(repr(v) for v in sorted(values))},))"


# ------------------------------------------------------------------------------------------------------ balance
def _chain(cols: list[str]) -> str:
    """The first amount that has a value on the row (the amount the check measures the balance's change against)."""
    expr = "0"
    for c in reversed(cols):
        expr = f"({c} if {c} is not None else {expr})"
    return expr


def _fee_columns(spec: GenerationSpec, rows: list[dict[str, Any]]) -> list[str]:
    C = _c()
    cols = [cid for cid, name in spec.delivered.items() if spec.columns[cid].dtype in ("float", "integer")
            and C._tokens(name) & C._FEE_WORDS and not C._tokens(name) & C._NOT_RAIL_FEE]
    return [c for c in cols if any((_num(r, c) or 0) > 0.005 for r in rows)][:2]


def _opening(spec: GenerationSpec, rows: list[dict[str, Any]], balance: str, moved: str, limit: dict[str, float]) -> dict[str, Any]:
    """The distribution of the balance an entity starts with: what the design already had, else a multiple of its usual amount."""
    import math

    C = _c()
    means = []
    for g in C._by_entity(spec, rows):
        values = [v for v in (_num(r, balance) for r in g) if v is not None and v > 0]
        if values:
            means.append(sum(values) / len(values))
    if len(means) >= 8:
        logs = [math.log(m) for m in means]
        median, sigma = statistics.median(means), statistics.pstdev(logs)
    else:
        sizes = [v for v in (_num(r, moved) for r in rows) if v is not None and v > 0]
        median, sigma = (20.0 * statistics.median(sizes) if sizes else 50000.0), 0.6
    sizes = sorted(v for v in (_num(r, moved) for r in rows) if v is not None and v > 0)
    if sizes:                                           # an account that moves this much money started with more than a few payments' worth
        median = max(median, OPENING_REACH * sizes[int(0.75 * (len(sizes) - 1))])
    sigma = min(OPENING_SIGMA[1], max(OPENING_SIGMA[0], sigma))
    least = max(0.0, float(limit.get("min", 0.0)))
    if sizes:
        least = max(least, OPENING_FLOOR * sizes[int(0.75 * (len(sizes) - 1))])      # no account starts too poor to make its own payments
    out = {"type": "lognormal", "median": round(float(median), 2), "sigma": round(sigma, 3), "min": round(least, 2)}
    if limit.get("max") is not None:
        out["max"] = float(limit["max"]) / 2.0                      # room for the credits that follow
        out["median"] = min(out["median"], out["max"] / 2.0)
    return out


def fix_balance(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], finding: dict[str, Any],
                bounds: dict[str, dict[str, float]]) -> str | None:
    C = _c()
    cid = _name_to_cid(spec, finding["columns"][0])
    credit = direction(spec, rows)
    amounts = C._movement_amounts(spec, rows)
    if cid is None or credit is None or not amounts or _emit(spec, cid) is None:
        return None
    order = sorted(amounts, key=lambda c: (-sum(1 for r in rows if _num(r, c) is not None), not C._tokens(amounts[c]) & {"transaction"}))[:3]
    fees = _fee_columns(spec, rows)
    fee = " + ".join(f"({f} if {f} is not None else 0)" for f in fees) or "0"
    move = _chain(order)
    opening = _fresh(spec, overlay, f"{cid}_opening")
    _put_latent(overlay, opening, "float", "entity", sample=_opening(spec, rows, cid, order[0], bounds.get(cid) or {}))
    step = f"(({move}) - ({fee}) if {credit} else 0 - (({move}) + ({fee})))"
    base = f"(prev['{cid}'] if prev is not None and prev['{cid}'] is not None else {opening})"
    digits = "" if spec.columns[cid].dtype == "integer" else ", 2"
    _put_column(overlay, cid, "event", expr=f"round({base} + {step}{digits})")
    limit = bounds.get(cid) or {}
    capped = _cap_debits(spec, rows, overlay, cid, credit, max(0.0, float(limit.get("min", 0.0))), opening, limit.get("max"))
    return (f"{finding['columns'][0]} now follows the money (debits are limited to what the balance holds: {capped} amount column(s)): previous balance plus the amount of a credit or minus the amount of a debit "
            f"(and minus {len(fees)} fee column(s)), starting from an opening balance drawn once per entity.")


def _cap_debits(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], balance: str, credit: str, low: float,
                opening: str | None, high: float | None = None) -> int:
    """Keep the balance inside its limits: a debit is reduced to a share of the balance above ``low``, a credit to a share of the room
    below ``high`` (the opening balance is what the first event may use)."""
    import re

    done = 0
    for amount in _c()._movement_amounts(spec, rows):
        if amount == balance or re.search(rf"\b{re.escape(amount)}\b", credit):
            continue                                       # the direction reads this column: it cannot also depend on the direction

        def cap(r: str) -> str:
            held = f"(prev['{balance}'] if prev is not None else {opening})" if opening else f"prev['{balance}']"
            ready = f"{held} is not None" if opening else f"prev is not None and {held} is not None"
            out = f"{r}"
            out = f"round(min({r}, {OVERDRAW_FRACTION} * ({held} - {low:g})), 2) if ({ready} and not {credit} and {held} - {low:g} >= 1) else {r}"
            if high is not None:
                out = (f"round(min({r}, {OVERDRAW_FRACTION} * ({high:g} - {held})), 2) if ({ready} and {credit} and {high:g} - {held} >= 1) else "
                       + out)
            return f"None if {r} is None else {out}"

        done += _wrap(spec, overlay, amount, cap)
    return done


def fix_overdraw(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], finding: dict[str, Any],
                 bounds: dict[str, dict[str, float]], opening: str | None = None) -> str | None:
    cid = _name_to_cid(spec, finding["columns"][0])
    credit = direction(spec, rows)
    if cid is None or credit is None:
        return None
    limit = bounds.get(cid) or {}
    done = _cap_debits(spec, rows, overlay, cid, credit, max(0.0, float(limit.get("min", 0.0))), opening, limit.get("max"))
    return f"A debit larger than the balance that is there is reduced to {OVERDRAW_FRACTION:.0%} of it ({done} amount column(s))." if done else None


# ------------------------------------------------------------------------------------------------------ dates
def fix_date(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], finding: dict[str, Any],
             bounds: dict[str, dict[str, float]]) -> str | None:
    from synth.samplers import is_rollup

    C = _c()
    cid = _name_to_cid(spec, finding["columns"][0])
    entry = _emit(spec, cid) if cid else None
    if cid is None or entry is None or not spec.clock or is_rollup(entry.sample):
        return None
    varies = any(len({r.get(cid) for r in g if r.get(cid) is not None}) > 1 for g in C._by_entity(spec, rows))
    if not varies:
        _put_column(overlay, cid, "entity", sample={"type": "rollup", "fn": "max", "of": f"min(add_days({spec.clock}, 1), as_of)"})
        return f"{finding['columns'][0]} is now dated the day after the entity's latest event, so it can no longer precede the activity it summarises."
    if not _wrap(spec, overlay, cid, lambda r: f"None if {r} is None else max({r}, event_at)"):
        return None
    return f"{finding['columns'][0]} is never earlier than the event it belongs to."


# ------------------------------------------------------------------------------------------------------ payment rails
def _rail_values(spec: GenerationSpec, rows: list[dict[str, Any]], rails: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """rail id -> text column -> the values of that column which name the rail (by the same words the check reads)."""
    C = _c()
    out: dict[str, dict[str, set[str]]] = {}
    for cid, name in spec.delivered.items():
        if spec.columns[cid].dtype not in ("string", "categorical"):
            continue
        for value in {r.get(cid) for r in rows if isinstance(r.get(cid), str) and len(r[cid]) <= 60}:
            words = C._value_words(value)
            hit = [rail for rail in rails if any(set(a.split()) <= words for a in rail["aliases"])]
            if len(hit) == 1:
                out.setdefault(str(hit[0].get("id") or hit[0]["aliases"][0]), {}).setdefault(cid, set()).add(value)
    return out


def _when_rail(values: dict[str, set[str]]) -> str:
    return " or ".join(f"{cid} in ({', '.join(repr(v) for v in sorted(vals))},)" for cid, vals in sorted(values.items()))


def fix_rail(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], finding: dict[str, Any],
             bounds: dict[str, dict[str, float]]) -> str | None:
    C = _c()
    limits = (getattr(spec, "scenario", None) or {}).get("country_limits") or {}
    rails = [r for r in limits.get("rails") or [] if isinstance(r, dict) and r.get("aliases")]
    named = _rail_values(spec, rows, rails)
    if not rails or not named:
        return None
    caps, floors, free = [], [], []
    for rail in rails:
        rid = str(rail.get("id") or rail["aliases"][0])
        if rid not in named:
            continue
        when = _when_rail(named[rid])
        if rail.get("max_amount") is not None:
            caps.append((when, float(rail["max_amount"])))
        if rail.get("min_amount") is not None:
            floors.append((when, float(rail["min_amount"])))
        if rail.get("fee") is not None and float(rail["fee"]) == 0.0:
            free.append(when)

    def pick(table: list[tuple[str, float]], otherwise: str) -> str:
        expr = otherwise
        for when, number in reversed(table):
            expr = f"({number:g} if ({when}) else {expr})"
        return expr

    changed = 0
    if caps or floors:
        # the cap and the floor of the rail a row names are two hidden facts of the row; every amount is brought inside them
        cap, floor = _fresh(spec, overlay, "rail_cap"), None
        _put_latent(overlay, cap, "float", "event", expr=pick(caps, "None"))
        floor = _fresh(spec, overlay, "rail_floor")
        _put_latent(overlay, floor, "float", "event", expr=pick(floors, "0"))

        def inside(r: str) -> str:
            # under the floor: lifted above it by its own size; over the cap: folded back below it (not piled up on it)
            lifted = f"({r} if {r} >= {floor} else {r} + {floor})"
            return (f"None if {r} is None else round({lifted} if ({cap} is None or {lifted} <= {cap}) "
                    f"else {cap} * (0.3 + 0.65 * {cap} / {lifted}), 2)")

        for amount in C._movement_amounts(spec, rows):
            changed += _wrap(spec, overlay, amount, inside)
    if free:
        zero = " or ".join(f"({w})" for w in free)
        for fee in _fee_columns(spec, rows):
            changed += _wrap(spec, overlay, fee, lambda r: f"None if {r} is None else 0 if ({zero}) else {r}")
    return f"Amounts and fees now respect the limits of the payment rail a row names ({changed} column(s))." if changed else None


FIXERS: dict[str, Callable[..., str | None]] = {          # the order is the order they are tried in
    "rail_limit_violated": fix_rail,
    "derived_before_event": fix_date,
    "balance_not_running": fix_balance,
    "debit_exceeds_balance": fix_overdraw,
}


def apply(spec: GenerationSpec, rows: list[dict[str, Any]], overlay: dict[str, Any], finding: dict[str, Any],
          bounds: dict[str, dict[str, float]] | None = None) -> tuple[dict[str, Any], str] | None:
    """The overlay with the fix for ``finding`` applied, and what was done; None when this finding has no automatic fix here."""
    fixer = FIXERS.get(str(finding.get("id")))
    if fixer is None:
        return None
    trial = copy.deepcopy(overlay)
    try:
        note = fixer(spec, rows, trial, finding, bounds or {})
    except Exception:
        logger.exception("automatic repair of %s failed", finding.get("id"))
        return None
    return (trial, note) if note else None
