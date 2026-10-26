"""An independent integrity checker for generated tables.

It never looks at how the tables were made: it takes the model and the delivered tables (field names and records, exactly as
a caller receives them) and re-checks every promise from the data alone. Keys, foreign keys, cardinality, groups, business
rules and declared value ranges are all verified, and every check is reported with what was expected and what was found.

Status of a check: ``pass``, ``fail``, ``warn`` (worth a look, not a broken promise) or ``info`` (a fact, such as a histogram).
The report passes when no check failed.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any

from er.model import Model


def _d(value: Any) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


class _Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, id_: str, kind: str, scope: str, description: str, status: str, expected: Any = None, observed: Any = None) -> None:
        self.checks.append({"id": id_, "kind": kind, "scope": scope, "description": description, "status": status,
                            "expected": expected, "observed": observed})

    def zero(self, id_: str, kind: str, scope: str, description: str, count: int, detail: Any = None) -> None:
        self.add(id_, kind, scope, description, "pass" if count == 0 else "fail", 0, count if detail is None or count == 0 else {"count": count, **detail})


def check(model: Model, tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rep = _Report()
    rows = {name: tables.get(name, {}).get("records", []) for name in model.entities}
    keys = {name: {r.get(e.key) for r in rows[name]} for name, e in model.entities.items()}
    touched = {(model.relationships[c.relationship].child if c.kind != "aggregate" else model.relationships[c.relationship].parent,
                c.params["column"] if c.kind != "aggregate" else c.params["parent_column"]) for c in model.constraints}

    # ---- tables ----
    for name in model.order:
        e, recs = model.entities[name], rows[name]
        fields = tables.get(name, {}).get("fields")
        rep.add(f"table.{name}.fields", "table", name, f"'{name}' delivers key, foreign keys, then declared columns in the planned order",
                "pass" if fields == e.fields else "fail", e.fields, fields)
        rep.add(f"table.{name}.rows", "table", name, f"rows in '{name}'", "info", None, len(recs))
        values = [r.get(e.key) for r in recs]
        rep.zero(f"pk.{name}.unique", "table", name, f"every {name}.{e.key} is unique", len(values) - len(set(values)))
        rep.zero(f"pk.{name}.not_null", "table", name, f"no {name}.{e.key} is empty", sum(v is None for v in values))
        _domain(rep, model, name, recs, touched)

    # ---- relationships ----
    for rel in model.relationships.values():
        if rel.kind == "one_to_many":
            child, fk = rows[rel.child], rel.fk
            _orphans(rep, rel.name, rel.child, fk, rel.parent, child, keys[rel.parent], required=True)
            per_parent = Counter(r[fk] for r in child if r.get(fk) is not None)
            hist = Counter(per_parent.get(k, 0) for k in keys[rel.parent])
            allowed = set(rel.card.weights) if rel.card.weights else set(range(rel.card.min, rel.card.max + 1))
            rep.add(f"card.{rel.name}", "relationship", rel.name,
                    f"each {rel.parent} has {rel.card.min} to {rel.card.max} {rel.child} rows" + (" (as weighted)" if rel.card.weights else ""),
                    "pass" if set(hist) <= allowed else "fail", sorted(allowed), {str(k): v for k, v in sorted(hist.items())})
        elif rel.kind == "many_to_one":
            child, fk = rows[rel.child], rel.fk
            _orphans(rep, rel.name, rel.child, fk, rel.parent, child, keys[rel.parent], required=rel.null_share == 0)
            used = Counter(r[fk] for r in child if r.get(fk) is not None)
            if used:
                top = used.most_common(1)[0]
                rep.add(f"spread.{rel.name}", "relationship", rel.name, f"how the {rel.child} rows spread over {rel.parent}", "info", None,
                        {"parents_used": len(used), "of": len(keys[rel.parent]), "busiest_parent_share": round(top[1] / sum(used.values()), 3)})
        elif rel.kind == "many_to_many":
            through = rows[rel.through]
            _orphans(rep, rel.name, rel.through, rel.left_fk, rel.left, through, keys[rel.left], required=True, suffix="left")
            _orphans(rep, rel.name, rel.through, rel.right_fk, rel.right, through, keys[rel.right], required=True, suffix="right")
            pairs = [(r.get(rel.left_fk), r.get(rel.right_fk)) for r in through]
            rep.zero(f"pairs.{rel.name}", "relationship", rel.name, f"no {rel.left}/{rel.right} pair appears twice in {rel.through}", len(pairs) - len(set(pairs)))
            per_right = Counter(p[1] for p in pairs)
            hist = Counter(per_right.get(k, 0) for k in keys[rel.right])
            allowed = set(rel.card.weights) if rel.card.weights else set(range(rel.card.min, rel.card.max + 1))
            allowed = {min(k, len(keys[rel.left])) for k in allowed}
            rep.add(f"card.{rel.name}", "relationship", rel.name, f"each {rel.right} is linked to {rel.card.min} to {rel.card.max} {rel.left} rows",
                    "pass" if set(hist) <= allowed else "fail", sorted(allowed), {str(k): v for k, v in sorted(hist.items())})
            uncovered = len(keys[rel.left] - {p[0] for p in pairs})
            rep.add(f"coverage.{rel.name}", "relationship", rel.name, f"every {rel.left} is linked to at least one {rel.right}",
                    "pass" if uncovered == 0 else ("warn" if rel.cover_left else "info"), 0, uncovered)
        else:
            _group(rep, model, rel, rows[rel.parent], keys[rel.parent])

    # ---- business rules ----
    for c in model.constraints:
        _constraint(rep, model, c, rows)

    # ---- dates measured from another column ----
    for name in model.order:
        for col in model.entities[name].columns:
            if col.gen["type"] != "date_after":
                continue
            base, lo, hi = col.gen["column"], col.gen["min_days"], col.gen["max_days"]
            bad = 0
            for r in rows[name]:
                a, b = _d(r.get(base)), _d(r.get(col.name))
                if (a is None) != (b is None) and col.gen.get("null_share", 0) == 0:
                    bad += 1
                elif a and b and not lo <= (b - a).days <= hi:
                    bad += 1
            rep.zero(f"after.{name}.{col.name}", "column", name, f"{name}.{col.name} is {lo} to {hi} days after {name}.{base}", bad)

    failed = sum(c["status"] == "fail" for c in rep.checks)
    return {
        "passed": failed == 0,
        "summary": {
            "checks": len(rep.checks),
            "passed": sum(c["status"] == "pass" for c in rep.checks),
            "failed": failed,
            "warnings": sum(c["status"] == "warn" for c in rep.checks),
            "info": sum(c["status"] == "info" for c in rep.checks),
            "tables": len(model.entities),
            "total_rows": sum(len(r) for r in rows.values()),
        },
        "checks": rep.checks,
    }


def _orphans(rep: _Report, rel: str, child: str, fk: str, parent: str, child_rows, parent_keys, required: bool, suffix: str = "") -> None:
    tag = f".{suffix}" if suffix else ""
    values = [r.get(fk) for r in child_rows]
    rep.zero(f"fk.{rel}.orphans{tag}", "relationship", rel, f"every {child}.{fk} points at an existing {parent}",
             sum(v is not None and v not in parent_keys for v in values))
    nulls = sum(v is None for v in values)
    if required:
        rep.zero(f"fk.{rel}.required{tag}", "relationship", rel, f"no {child}.{fk} is empty", nulls)
    else:
        rep.add(f"fk.{rel}.required{tag}", "relationship", rel, f"{child}.{fk} may be empty", "info", None, nulls)


def _group(rep: _Report, model: Model, rel, rows, key_set) -> None:
    key, fk, entity = model.entities[rel.parent].key, rel.fk, rel.parent
    position = {r.get(key): i for i, r in enumerate(rows)}
    _orphans(rep, rel.name, entity, fk, entity, rows, key_set, required=False)
    parent_of = {r.get(key): r.get(fk) for r in rows}
    rep.zero(f"group.{rel.name}.self", "relationship", rel.name, f"no {entity} is its own main {entity}", sum(k == v for k, v in parent_of.items()))
    deepest, loops = 0, 0
    for start in parent_of:
        seen, cur, depth = set(), start, 0
        while parent_of.get(cur) is not None:
            if cur in seen:
                loops += 1
                break
            seen.add(cur)
            cur, depth = parent_of[cur], depth + 1
            if depth > len(parent_of):
                break
        deepest = max(deepest, depth)
    rep.zero(f"group.{rel.name}.loops", "relationship", rel.name, f"the {entity} links never lead back to themselves", loops)
    rep.add(f"group.{rel.name}.depth", "relationship", rel.name, f"a group is one main {entity} and its members (no member is itself a main)",
            "pass" if deepest <= 1 else "fail", "at most 1", deepest)
    members = Counter(v for v in parent_of.values() if v is not None)
    allowed = set(rel.card.weights) if rel.card.weights else set(range(rel.card.min, rel.card.max + 1))
    hist = Counter(members.values())
    rep.add(f"group.{rel.name}.size", "relationship", rel.name, f"each main {entity} has {rel.card.min} to {rel.card.max} members",
            "pass" if set(hist) <= allowed else "fail", sorted(allowed), {str(k): v for k, v in sorted(hist.items())})
    late = sum(1 for k, v in parent_of.items() if v is not None and v in position and position[v] > position[k])
    rep.zero(f"group.{rel.name}.main_first", "relationship", rel.name, f"each main {entity} is listed before its members", late)
    rep.add(f"group.{rel.name}.groups", "relationship", rel.name, f"main {entity} rows and members", "info", None,
            {"groups": len(members), "members": sum(members.values()), "standalone": len(rows) - len(members) - sum(members.values())})


def _constraint(rep: _Report, model: Model, c, rows) -> None:
    rel, p = model.relationships[c.relationship], c.params
    child_rows, parent_rows = rows[rel.child], rows[rel.parent]
    pkey = model.entities[rel.parent].key
    parents = {r.get(pkey): r for r in parent_rows}
    scope = f"{c.kind}:{c.relationship}"
    cid = f"rule.{c.name}"
    if c.kind == "aggregate":
        collected: dict[Any, list[Any]] = {}
        for r in child_rows:
            if r.get(rel.fk) is not None:
                collected.setdefault(r[rel.fk], []).append(r.get(p["child_column"]) if p["child_column"] else None)
        bad = 0
        for r in parent_rows:
            vals = collected.get(r.get(pkey), [])
            nums = [v for v in vals if v is not None]
            want = {"count": len(vals), "sum": sum(nums) if nums else 0,
                    "min": min(nums) if nums else None, "max": max(nums) if nums else None,
                    "avg": sum(nums) / len(nums) if nums else None}[p["fn"]]
            got = r.get(p["parent_column"])
            if want is None or got is None:
                bad += want != got
            elif abs(float(want) - float(got)) > 0.011:
                bad += 1
        what = f"{rel.parent}.{p['parent_column']} equals the {p['fn']} of {rel.child}" + (f".{p['child_column']}" if p["child_column"] else " rows")
        rep.zero(cid, "constraint", scope, what, bad, {"parents_checked": len(parent_rows)})
        return
    bad = checked = 0
    for r in child_rows:
        parent = parents.get(r.get(rel.fk)) if r.get(rel.fk) is not None else None
        if parent is None:
            continue
        if c.kind == "date_not_before_parent":
            raw = r.get(p["column"])
            a, b = _d(raw), _d(parent.get(p["parent_column"]))
            if raw is not None and a is None:
                bad += 1                                   # not a readable date
                continue
            if a is None or b is None:
                continue                                   # nothing to compare (an empty date is allowed)
            checked += 1
            bad += a < b
        elif c.kind == "same_as_parent":
            checked += 1
            bad += r.get(p["column"]) != parent.get(p["parent_column"])
        else:
            if parent.get(p["parent_column"]) in p["when"]:
                checked += 1
                bad += r.get(p["column"]) != p["set"]
    what = {
        "date_not_before_parent": f"{rel.child}.{p.get('column')} is not before {rel.parent}.{p.get('parent_column')}",
        "same_as_parent": f"{rel.child}.{p.get('column')} matches {rel.parent}.{p.get('parent_column')}",
        "status_follows_parent": f"{rel.child}.{p.get('column')} is '{p.get('set')}' whenever {rel.parent}.{p.get('parent_column')} is {p.get('when')}",
    }[c.kind]
    rep.zero(cid, "constraint", scope, what, bad)
    rep.checks[-1]["rows_checked"] = checked


def _domain(rep: _Report, model: Model, name: str, recs, touched) -> None:
    """Declared value ranges hold. Columns a business rule is allowed to rewrite are left to that rule."""
    bad: dict[str, int] = {}
    for col in model.entities[name].columns:
        g = col.gen
        if (name, col.name) in touched or g["type"] in ("const", "date_after", "digits"):
            continue
        nullable = bool(g.get("null_share"))
        count = 0
        for r in recs:
            v = r.get(col.name)
            if v is None:
                count += not nullable
            elif g["type"] == "int":
                count += not (isinstance(v, int) and not isinstance(v, bool) and g["min"] <= v <= g["max"])
            elif g["type"] == "float":
                count += not (isinstance(v, (int, float)) and g["min"] - 1e-9 <= v <= g["max"] + 1e-9)
            elif g["type"] == "choice":
                count += v not in (g["weights"] if g.get("weights") else g["values"])
            elif g["type"] == "bool":
                count += not isinstance(v, bool)
            elif g["type"] == "date":
                d = _d(v)
                count += not (d and date.fromisoformat(g["start"]) <= d <= date.fromisoformat(g["end"]))
        if count:
            bad[col.name] = count
    rep.zero(f"domain.{name}", "table", name, f"every {name} value is inside the range or list declared for its column", sum(bad.values()), {"columns": bad})
