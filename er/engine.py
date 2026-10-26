"""Generate the tables of an ER model: parents first, foreign keys from real parents, then the business rules.

Order of work for each entity (entities are taken parent-first):
  1. how many rows, and which parent each belongs to (one_to_many), or which pairs exist (many_to_many);
  2. which rows start a group and which rows belong to it (group);
  3. the rows: key, foreign keys, then the declared columns in order;
  4. the constraints that read the parent (dates, copied values, statuses);
and, once every table exists, the aggregates (a parent's total from its children, children first).

Every random choice comes from a stream named by seed, entity and stage, so the same model and seed always give the same data.
"""
from __future__ import annotations

import itertools
import time
from datetime import date, timedelta
from typing import Any

from er.generators import Gen, GenerationError, stream
from er.model import MAX_TOTAL_ROWS, Model, check_budget, check_counts


def generate(model: Model, seed: int | None = None, counts: dict[str, int] | None = None) -> dict[str, Any]:
    seed = model.seed if seed is None else seed
    problems = check_counts(model, counts) + check_budget(model, counts)
    if problems:
        raise GenerationError("; ".join(problems))
    started = time.perf_counter()
    rows_by_entity: dict[str, list[dict[str, Any]]] = {}
    by_key: dict[str, dict[Any, dict[str, Any]]] = {}
    total = 0
    for name in model.order:
        rows = _build(model, name, seed, counts or {}, rows_by_entity)
        total += len(rows)
        if total > MAX_TOTAL_ROWS:
            raise GenerationError(f"the run passed {MAX_TOTAL_ROWS:,} rows; lower a count or a children range")
        rows_by_entity[name] = rows
        by_key[name] = {r[model.entities[name].key]: r for r in rows}
        _apply_constraints(model, name, seed, rows_by_entity, by_key)
    _apply_aggregates(model, rows_by_entity)
    return {
        "seed": seed,
        "order": list(model.order),
        "tables": {name: {"fields": model.entities[name].fields, "records": [_plain(r) for r in rows_by_entity[name]]} for name in model.order},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in row.items()}


# ---- one entity ----------------------------------------------------------------------------------------------------
def _build(model: Model, name: str, seed: int, counts: dict[str, int], done: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    entity = model.entities[name]
    links, rows_rng, group_rng = stream(seed, name, "links"), stream(seed, name, "rows"), stream(seed, name, "group")
    driver = model.relationships.get(entity.driver) if entity.driver else None

    # 1. how many rows, and who their parents are
    driver_values: list[Any] = []
    if driver is None:
        n = counts.get(name, entity.count)
    elif driver.kind == "one_to_many":
        parent_key = model.entities[driver.parent].key
        for parent in done[driver.parent]:
            driver_values += [parent[parent_key]] * driver.card.draw(links)
        n = len(driver_values)
    else:  # many_to_many: the rows are the (left, right) pairs
        driver_values = _pairs(model, driver, done, links)
        n = len(driver_values)

    # 2. groups: indexes only, so a main row always comes before its members
    main_of = _groups(model, name, n, group_rng)

    # how each foreign key of this entity gets its value
    pickers: dict[str, Any] = {}
    for fk in entity.fks:
        rel = model.relationships[fk.relationship]
        if rel is driver and rel.kind == "one_to_many":
            pickers[fk.column] = lambda i, row: driver_values[i]
        elif rel is driver:
            side = 0 if fk.column == rel.left_fk else 1
            pickers[fk.column] = lambda i, row, side=side: driver_values[i][side]
        elif rel.kind == "many_to_one":
            pickers[fk.column] = _lookup_picker(model, rel, done, links, name)
        else:  # group
            pickers[fk.column] = lambda i, row, fkcol=fk.column: None

    keygen = Gen(entity.key_gen, f"{name}.{entity.key}")
    gens = [(c.name, Gen(c.gen, f"{name}.{c.name}")) for c in entity.columns]
    group_fk = next((f.column for f in entity.fks if model.relationships[f.relationship].kind == "group"), None)
    rows: list[dict[str, Any]] = []
    key = entity.key
    for i in range(n):
        row: dict[str, Any] = {key: keygen.value(rows_rng, {})}
        for fk in entity.fks:
            row[fk.column] = pickers[fk.column](i, row)
        if group_fk and i in main_of:
            row[group_fk] = rows[main_of[i]][key]
        for cname, gen in gens:
            row[cname] = gen.value(rows_rng, row)
        rows.append(row)
    return rows


def _pairs(model: Model, rel, done, rng) -> list[tuple[Any, Any]]:
    """The (left key, right key) pairs of a many_to_many relationship, never repeating a pair."""
    left_keys = [r[model.entities[rel.left].key] for r in done[rel.left]]
    right_keys = [r[model.entities[rel.right].key] for r in done[rel.right]]
    if not left_keys or not right_keys:
        return []
    uncovered = set(left_keys)
    pending = left_keys[:]
    rng.shuffle(pending)
    pairs: list[tuple[Any, Any]] = []
    for right in right_keys:
        k = min(rel.card.draw(rng), len(left_keys))
        chosen: list[Any] = []
        if rel.cover_left:
            while pending and len(chosen) < k:
                candidate = pending.pop()
                if candidate in uncovered:
                    chosen.append(candidate)
        if len(chosen) < k:
            for candidate in rng.sample(left_keys, min(len(left_keys), k + len(chosen))):
                if candidate not in chosen:
                    chosen.append(candidate)
                if len(chosen) == k:
                    break
        uncovered.difference_update(chosen)
        pairs += [(left, right) for left in chosen]
    return pairs


def _groups(model: Model, name: str, n: int, rng) -> dict[int, int]:
    """member row index -> main row index. The main of a group is its lowest row index, so it is listed first."""
    rel = next((r for r in model.relationships.values() if r.kind == "group" and r.parent == name), None)
    if rel is None or n < 2:
        return {}
    card = rel.card
    mains = min(round(rel.share * n), n // (1 + card.min))
    if mains < 1:
        return {}
    sizes = [card.draw(rng) for _ in range(mains)]
    while sum(sizes) > n - mains:                      # more members than there are rows: trim, never below the minimum
        i = rng.randrange(mains)
        if sizes[i] > card.min:
            sizes[i] -= 1
    order = list(range(n))
    rng.shuffle(order)
    main_of: dict[int, int] = {}
    pos = 0
    for size in sizes:
        members = order[pos:pos + 1 + size]
        pos += 1 + size
        main = min(members)
        main_of.update({m: main for m in members if m != main})
    return main_of


def _lookup_picker(model: Model, rel, done, rng, child: str):
    parent_rows = done[rel.parent]
    if not parent_rows:
        raise GenerationError(f"'{rel.parent}' has no rows, so '{child}.{rel.fk}' has nothing to point at; raise its count or children minimum")
    keys = [r[model.entities[rel.parent].key] for r in parent_rows]
    rng.shuffle(keys)                                                        # which parents become the hubs is random, not the first rows
    cum = list(itertools.accumulate((1.0 / (j + 1)) ** rel.skew for j in range(len(keys)))) if rel.skew else None
    null_share = rel.null_share

    def pick(i: int, row: dict[str, Any]) -> Any:
        if null_share and rng.random() < null_share:
            return None
        return rng.choices(keys, cum_weights=cum)[0] if cum else rng.choice(keys)
    return pick


# ---- business rules ------------------------------------------------------------------------------------------------
def _apply_constraints(model: Model, name: str, seed: int, rows_by_entity, by_key) -> None:
    cons = [c for c in model.constraints if c.kind != "aggregate" and model.relationships[c.relationship].child == name]
    if not cons:
        return
    entity = model.entities[name]
    rng = stream(seed, name, "fix")
    gens = {c.name: Gen(c.gen, f"{name}.{c.name}") for c in entity.columns}
    dependents: dict[str, list[str]] = {}
    for c in entity.columns:
        if c.gen["type"] == "date_after":
            dependents.setdefault(c.gen["column"], []).append(c.name)

    def changed(row: dict[str, Any], column: str) -> None:
        for dep in dependents.get(column, []):             # a date measured from this column is drawn again
            row[dep] = gens[dep].value(rng, row)
            changed(row, dep)

    for row in rows_by_entity[name]:
        for c in cons:
            rel = model.relationships[c.relationship]
            ref = row.get(rel.fk)
            parent = by_key[rel.parent].get(ref) if ref is not None else None
            if parent is None:
                continue
            p = c.params
            if c.kind == "date_not_before_parent":
                mine, theirs = row[p["column"]], parent[p["parent_column"]]
                if mine is not None and theirs is not None and mine < theirs:
                    row[p["column"]] = theirs + timedelta(days=rng.randint(0, p["max_days_after"]))
                    changed(row, p["column"])
            elif c.kind == "same_as_parent":
                row[p["column"]] = parent[p["parent_column"]]
                changed(row, p["column"])
            elif c.kind == "status_follows_parent":
                if parent[p["parent_column"]] in p["when"]:
                    row[p["column"]] = p["set"]
                    changed(row, p["column"])


def _apply_aggregates(model: Model, rows_by_entity) -> None:
    for name in reversed(model.order):                      # children first, so a total can use a child's own total
        for c in model.constraints:
            rel = model.relationships[c.relationship]
            if c.kind != "aggregate" or rel.parent != name:
                continue
            p = c.params
            collected: dict[Any, list[Any]] = {}
            for child in rows_by_entity[rel.child]:
                ref = child.get(rel.fk)
                if ref is not None:
                    collected.setdefault(ref, []).append(child[p["child_column"]] if p["child_column"] else None)
            key = model.entities[name].key
            results = {row[key]: _aggregate(p["fn"], collected.get(row[key], [])) for row in rows_by_entity[name]}
            for row in rows_by_entity[name]:
                row[p["parent_column"]] = results[row[key]]


def _aggregate(fn: str, values: list[Any]) -> Any:
    if fn == "count":
        return len(values)
    nums = [v for v in values if v is not None]
    if fn == "sum":
        total = sum(nums)
        return total if all(isinstance(v, int) for v in nums) else round(total, 2)
    if not nums:
        return None
    if fn == "min":
        return min(nums)
    if fn == "max":
        return max(nums)
    return round(sum(nums) / len(nums), 2)
