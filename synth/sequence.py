"""The order of a dataset's columns: the story a reader follows from left to right.

A variable carries ``sequence`` (a whole number; smaller comes first). The intent agent (``agents/intent_agent.py``) decides it from what
the columns mean when it gives the variables; this module is what stands behind that answer: it enforces the hard rules on it, numbers
the columns, places a column the model left out, and orders everything itself when the model cannot be reached. Variables without a
number are placed by the story order below, so a dataset read as a CSV tells its story: who (the party), what they hold (account, contract), the facts about them, when it
started and ended, then each event - its identifier, its moments in time, its kind, its amounts, and how it ended.

The story order uses only the shape of the definitions (dtype, scope, name words, ``depends_on``) and a small domain-neutral
vocabulary of how things relate (a party holds an account, an account has transactions; a start comes before an end):

1. entity-level identifiers, the party before what it holds before what happens to it (customer, account, ...);
2. entity-level facts (name, type, segment, ...);
3. entity-level dates, a start before its end;
4. event identifiers (the transaction's own id first);
5. event dates, a start before the moment an event happens, that before its processing, that before its end;
6. what kind of event it is (type, category, channel, purpose, ...);
7. event numbers: the amount, then its fees, then the balance it leads to;
8. how it ended (status, outcome, flags).

A column that reads another column (``depends_on``) always comes after it, and a ``start`` / ``end`` pair that differs only by that
word keeps its order whatever else is around. Explicit ``sequence`` numbers always win over the story order.
"""
from __future__ import annotations

import heapq
import re
from typing import Any, Iterable

SEQUENCE_STEP = 10               # distance between the sequence numbers given automatically, so a person can insert a column between two

_ID_LAST = frozenset({"id", "identifier", "uuid", "key", "reference", "ref", "number", "no", "num"})
_NOT_ID = frozenset({"phone", "mobile", "telephone", "fax", "zip", "postal", "pin", "amount", "count"})
# how the things of a business relate: who -> what they hold -> what happens to it (matched on the noun in front of "id")
_LEVEL_PARTY = frozenset({"customer", "client", "user", "member", "subscriber", "party", "person", "patient", "borrower", "applicant", "holder",
                          "owner", "employee", "student", "passenger", "guest", "individual", "household", "company", "organisation",
                          "organization", "merchant", "vendor", "supplier"})
_LEVEL_HOLDING = frozenset({"account", "contract", "policy", "subscription", "agreement", "card", "loan", "plan", "product", "device", "line",
                            "mandate", "portfolio", "asset", "wallet", "facility", "connection", "meter", "vehicle", "property", "branch",
                            "relationship", "profile"})
_LEVEL_EVENT = frozenset({"transaction", "payment", "order", "claim", "case", "invoice", "ticket", "event", "request", "transfer", "withdrawal",
                          "deposit", "session", "call", "message", "alert", "notification", "statement", "incident", "visit", "booking",
                          "shipment", "application", "task", "assessment", "evaluation", "enquiry", "inquiry", "usage", "record"})

_START = frozenset({"start", "started", "begin", "began", "open", "opened", "opening", "create", "created", "creation", "effective", "from",
                    "issue", "issued", "onboard", "onboarded", "registration", "registered", "enrol", "enrolled", "join", "joined", "request",
                    "requested", "birth", "incorporation", "activation", "activated", "first", "booked"})
_END = frozenset({"end", "ended", "close", "closed", "closing", "closure", "expiry", "expire", "expires", "expired", "expiration", "maturity",
                  "matured", "terminate", "terminated", "termination", "until", "to", "complete", "completed", "completion", "resolved",
                  "resolution", "settled", "settlement", "last", "final", "due", "cancelled", "canceled", "cancellation"})
_PROCESSING = frozenset({"posting", "posted", "processed", "processing", "value", "calculation", "calculated", "assessed", "review", "reviewed",
                         "evaluated", "notification", "notified", "updated", "modified", "recorded", "logged", "reported"})
_DATE_WORDS = frozenset({"date", "time", "timestamp", "datetime", "at", "on", "day"})

_STATUS = frozenset({"status", "state", "stage", "outcome", "result", "flag", "indicator", "response", "action", "decision", "verdict", "resolved"})
_KIND = frozenset({"type", "category", "kind", "class", "subcategory", "sub", "group"})
_MEANS = frozenset({"channel", "mechanism", "mode", "method", "purpose", "reason", "description", "source", "location", "merchant",
                    "counterparty", "network", "rail", "medium", "device"})
_ENTITY_HEAD = frozenset({"name", "type", "segment", "category", "tier", "class", "title", "gender", "language", "grade"})

_BALANCE = frozenset({"balance", "total", "closing", "opening", "outstanding", "remaining", "available", "cumulative"})
_FEE = frozenset({"fee", "fees", "charge", "charges", "tax", "gst", "vat", "commission", "surcharge", "levy", "mdr", "interest", "discount"})
_AMOUNT = frozenset({"amount", "value", "price", "quantity", "volume", "cost", "sum", "payment", "principal"})


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _dtype(variable: dict[str, Any]) -> str:
    t = str(variable.get("dtype") or "string").strip().lower()
    if t in {"datetime", "date", "timestamp"}:
        return "date"
    if t in {"bool", "boolean"}:
        return "boolean"
    if t in {"int", "integer", "float", "decimal", "number", "numeric", "double"}:
        return "number"
    return "string"


def _is_id(tokens: list[str], dtype: str) -> bool:
    return bool(tokens) and dtype in ("string", "number") and tokens[-1] in _ID_LAST and not set(tokens) & _NOT_ID


def id_level(tokens: list[str]) -> float:
    """0 for the party, 1 for what it holds, 2 for what happens; the noun nearest to ``id`` decides; unknown ids sit between 1 and 2."""
    for word in reversed([t for t in tokens if t not in _ID_LAST]):
        if word in _LEVEL_EVENT:
            return 2.0
        if word in _LEVEL_HOLDING:
            return 1.0
        if word in _LEVEL_PARTY:
            return 0.0
    return 1.5


def _date_rank(tokens: list[str], is_clock: bool) -> float:
    words = set(tokens)
    if words & _END and not words & _START:
        return 3.0
    if words & _START and not words & _END:
        return 0.0
    if words & _PROCESSING:
        return 2.0
    return 0.5 if is_clock else 1.0


def _stem(tokens: list[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if t not in _START and t not in _END and t not in _DATE_WORDS)


def _classify(variable: dict[str, Any], *, kind: str, entity_key: str | None, clock: str | None) -> tuple[int, float]:
    """(tier, rank inside the tier) of one variable; of two alike, the one with fewer qualifying words (the generic one) comes first."""
    tier, rank = _tier(variable, kind=kind, entity_key=entity_key, clock=clock)
    return tier, rank + 0.001 * len(_tokens(str(variable.get("name"))))


def _tier(variable: dict[str, Any], *, kind: str, entity_key: str | None, clock: str | None) -> tuple[int, float]:
    name = str(variable.get("name"))
    tokens = _tokens(name)
    dtype = _dtype(variable)
    words = set(tokens)
    entity = kind == "entity"
    is_clock = bool(clock) and name == clock
    if _is_id(tokens, dtype):
        level = id_level(tokens)
        if entity:
            return 1, level * 2 + (0 if name == entity_key else 1)
        own = 0 if (clock and set(_tokens(clock)) & set(tokens) - _ID_LAST - _DATE_WORDS) else 1
        return 4, level * 2 + own
    if dtype == "date":
        return (3 if entity else 5), _date_rank(tokens, is_clock)
    if dtype == "boolean" or tokens[:1] in (["is"], ["has"], ["can"], ["should"]) or words & _STATUS:
        return (2, 3.0) if entity else (8, 1.0 if dtype != "boolean" else 2.0)
    if dtype == "number":
        if entity:
            return 2, 2.0
        if words & _BALANCE:
            return 7, 2.0
        if words & _FEE:
            return 7, 1.0
        return 7, 0.0 if words & _AMOUNT else 0.5
    # a text fact
    if entity:
        return 2, 0.0 if words & _ENTITY_HEAD else 1.0
    return 6, 0.0 if words & _KIND else 1.0 if words & _MEANS else 2.0


def _edges(variables: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(before, after) pairs that must hold: a column after the columns it reads, a start before its end, a parent date before its child."""
    names = {str(v["name"]) for v in variables}
    edges: set[tuple[str, str]] = set()
    for v in variables:
        for dep in v.get("depends_on") or []:
            if str(dep) in names and str(dep) != str(v["name"]):
                edges.add((str(dep), str(v["name"])))
    dates = [(str(v["name"]), _tokens(str(v["name"]))) for v in variables if _dtype(v) == "date"]
    for a, ta in dates:
        for b, tb in dates:
            if a != b and _stem(ta) == _stem(tb) and set(ta) & _START and not set(ta) & _END and set(tb) & _END and not set(tb) & _START:
                edges.add((a, b))
    try:
        from agents.data_generation_agent import _infer_temporal_relationships

        for parent, child, *_ in _infer_temporal_relationships([v for v in variables if _dtype(v) == "date"]):
            edges.add((str(parent), str(child)))
    except Exception:                                   # a relation that cannot be inferred is simply not used
        pass
    return sorted(edges)


def _constrained(key: dict[str, tuple], edges: list[tuple[str, str]]) -> list[str]:
    """The names sorted by ``key``, except that an edge ``(a, b)`` always puts ``a`` before ``b`` (a stable topological pass)."""
    after: dict[str, list[str]] = {n: [] for n in key}
    waiting = {n: 0 for n in key}
    for a, b in edges:
        after[a].append(b)
        waiting[b] += 1
    heap = [(key[n], n) for n in key if waiting[n] == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, n = heapq.heappop(heap)
        order.append(n)
        for m in after[n]:
            waiting[m] -= 1
            if waiting[m] == 0:
                heapq.heappush(heap, (key[m], m))
    placed = set(order)
    order += sorted((n for n in key if n not in placed), key=lambda n: key[n])        # a cycle in the edges: keep the given order for the rest
    return order


def merge_preferred(preferred: Iterable[str], fallback: list[str]) -> list[str]:
    """``preferred`` (a model's order; unknown and repeated names dropped) with every name it left out put back after the name that
    came before it in ``fallback``, so a partial answer still yields a complete order."""
    known = set(fallback)
    merged: list[str] = []
    for name in preferred:
        if name in known and name not in merged:
            merged.append(name)
    for position, name in enumerate(fallback):
        if name in merged:
            continue
        before = next((p for p in reversed(fallback[:position]) if p in merged), None)
        merged.insert(merged.index(before) + 1 if before is not None else 0, name)
    return merged


def story_order(variables: Iterable[dict[str, Any]], *, entity_key: str | None = None, kinds: dict[str, str] | None = None,
                clock: str | None = None, preferred: Iterable[str] | None = None, fit: bool = False) -> list[str]:
    """The names in the order a reader should meet them (explicit ``sequence`` numbers win over the story order).

    ``kinds`` maps a name to ``entity`` (constant over the entity's history) or ``event``; without it a variable is an entity fact
    when its ``scope`` says so or when it is the entity key.

    ``preferred`` is an order someone else decided (the intent agent's): it replaces the story order, and only the hard rules still
    apply - a column after the columns it reads, a start before its end. Names it leaves out are placed by the story order.
    ``fit`` places a variable that has no ``sequence`` right after the numbered variable that precedes it in the story (see ``assign``).
    """
    return _plan(variables, entity_key=entity_key, kinds=kinds, clock=clock, preferred=preferred, fit=fit)[0]


def _plan(variables: Iterable[dict[str, Any]], *, entity_key: str | None, kinds: dict[str, str] | None, clock: str | None,
          preferred: Iterable[str] | None, fit: bool) -> tuple[list[str], dict[str, int]]:
    items = [v for v in variables if isinstance(v, dict) and v.get("name")]
    seen: set[str] = set()
    items = [v for v in items if not (str(v["name"]) in seen or seen.add(str(v["name"])))]
    kinds = kinds or {}
    key: dict[str, tuple[int, float, int]] = {}
    for i, v in enumerate(items):
        name = str(v["name"])
        kind = kinds.get(name) or ("entity" if str(v.get("scope") or "").lower() == "entity" or name == entity_key else "event")
        tier, rank = _classify(v, kind=kind, entity_key=entity_key, clock=clock)
        key[name] = (tier, rank, i)

    # the story order, then a stable pass so that a column never precedes what it reads (and a start never follows its end)
    edges = [(a, b) for a, b in _edges(items) if a in key and b in key]
    story = _constrained(key, edges)
    if preferred is not None:
        wanted = merge_preferred(preferred, story)
        story = _constrained({n: (p, 0.0, 0) for p, n in enumerate(wanted)}, edges)

    explicit: dict[str, int] = {}
    for v in items:
        s = v.get("sequence")
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            explicit[str(v["name"])] = int(s)
    if not explicit:
        return story, {n: (p + 1) * SEQUENCE_STEP for p, n in enumerate(story)}
    effective: dict[str, int] = {}
    if fit:
        # a column without a number sits right after the numbered column before it (or ahead of the first numbered one)
        last: int | None = None
        for n in story:
            if n in explicit:
                effective[n] = last = explicit[n]
            else:
                effective[n] = last + 1 if last is not None else max(0, min(explicit.values()) - 1)
                last = effective[n]
    else:
        effective = {n: explicit.get(n, (p + 1) * SEQUENCE_STEP) for p, n in enumerate(story)}
    order = sorted(story, key=lambda n: (effective[n], story.index(n)))
    return order, effective


def assign(variables: list[dict[str, Any]], *, entity_key: str | None = None, kinds: dict[str, str] | None = None,
           clock: str | None = None, preferred: Iterable[str] | None = None, fit: bool = False) -> list[dict[str, Any]]:
    """The variables in order, each carrying a ``sequence`` (one given by a person is kept; the others are numbered 10, 20, ...).

    The order is the story order, or ``preferred`` (the intent agent's) with the hard rules enforced; with ``fit`` the variables that
    have no number are slotted next to the numbered ones instead of renumbering everything.
    """
    order, effective = _plan(variables, entity_key=entity_key, kinds=kinds, clock=clock, preferred=preferred, fit=fit)
    by_name = {str(v["name"]): v for v in variables if isinstance(v, dict) and v.get("name")}
    out: list[dict[str, Any]] = []
    for position, name in enumerate(order):
        item = dict(by_name[name])
        given = item.get("sequence")
        if isinstance(given, bool) or not isinstance(given, (int, float)):
            item["sequence"] = effective[name] if fit else (position + 1) * SEQUENCE_STEP
        out.append(item)
    out += [dict(v) for v in variables if not (isinstance(v, dict) and v.get("name"))]
    return out
