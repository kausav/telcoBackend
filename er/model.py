"""Parse and validate an ER model description (plain JSON) into a ``Model``.

The description names entities (tables), how they relate, and the business rules between them::

    {"name": "...", "seed": 42,
     "entities":      [{"name": "customer", "key": "customer_id", "count": 40, "columns": [{"name": "segment", "gen": {...}}]}],
     "relationships": [{"type": "one_to_many", "parent": "customer", "child": "account", "children": {"min": 1, "max": 4}}],
     "constraints":   [{"type": "date_not_before_parent", "relationship": "...", "column": "...", "parent_column": "..."}]}

Relationship types: ``one_to_many``, ``many_to_one`` (a lookup), ``many_to_many`` (through a junction entity) and ``group``
(rows of one entity that belong to a main row of the same entity). Everything is checked here and every problem is reported
at once, with the path of the offending value, so a person can fix a whole model in one pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

MAX_ENTITIES = 30
MAX_COLUMNS = 80
MAX_ROOT_COUNT = 5000                 # entities per root table, as the main generator
MAX_TOTAL_ROWS = 200_000              # rows over all tables of one run
MAX_CARD = 500                        # children per parent
SEED_MAX = 2_147_483_647
DEFAULT_SEED = 42

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REL_TYPES = ("one_to_many", "many_to_one", "many_to_many", "group")
CONSTRAINT_TYPES = ("date_not_before_parent", "same_as_parent", "status_follows_parent", "aggregate")
AGG_FNS = ("sum", "count", "min", "max", "avg")

# generator type -> (required keys, optional keys)
GEN_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "id": (set(), {"prefix", "digits", "start"}),
    "choice": (set(), {"weights", "values"}),
    "int": ({"min", "max"}, set()),
    "float": ({"min", "max"}, {"round"}),
    "date": ({"start", "end"}, set()),
    "bool": (set(), {"p"}),
    "const": ({"value"}, set()),
    "date_after": ({"column"}, {"min_days", "max_days"}),
    "digits": ({"length"}, {"prefix", "unique"}),
}
_DATE_TYPES = ("date", "date_after")


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


# ---- structures ---------------------------------------------------------------------------------------------------
@dataclass
class Card:
    """How many children one parent gets: a range, optionally shaped by weights per count."""
    min: int
    max: int
    weights: dict[int, float] | None = None

    def expected(self) -> float:
        if self.weights:
            total = sum(self.weights.values())
            return sum(k * w for k, w in self.weights.items()) / total
        return (self.min + self.max) / 2

    def draw(self, rng) -> int:
        if self.weights:
            counts = list(self.weights)
            return rng.choices(counts, weights=[self.weights[k] for k in counts])[0]
        return rng.randint(self.min, self.max)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"min": self.min, "max": self.max}
        if self.weights:
            out["weights"] = {str(k): v for k, v in sorted(self.weights.items())}
        return out


@dataclass
class Column:
    name: str
    gen: dict[str, Any]


@dataclass
class Fk:
    column: str
    relationship: str
    parent: str
    parent_key: str


@dataclass
class Entity:
    name: str
    key: str
    key_gen: dict[str, Any]
    count: int | None
    columns: list[Column] = field(default_factory=list)
    fks: list[Fk] = field(default_factory=list)
    driver: str | None = None              # the relationship that decides how many rows this entity has

    @property
    def fields(self) -> list[str]:
        """Delivered column order: the key, the foreign keys, then the declared columns."""
        return [self.key] + [f.column for f in self.fks] + [c.name for c in self.columns]


@dataclass
class Relationship:
    name: str
    kind: str
    parent: str | None = None
    child: str | None = None
    fk: str | None = None
    card: Card | None = None
    null_share: float = 0.0
    skew: float = 0.0
    left: str | None = None
    right: str | None = None
    through: str | None = None
    left_fk: str | None = None
    right_fk: str | None = None
    cover_left: bool = True
    share: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.kind}
        if self.kind in ("one_to_many", "many_to_one"):
            out.update(parent=self.parent, child=self.child, fk=self.fk)
        if self.kind == "one_to_many" and self.card:
            out["children"] = self.card.to_dict()
        if self.kind == "many_to_one":
            out.update(null_share=self.null_share, skew=self.skew)
        if self.kind == "many_to_many":
            out.update(left=self.left, right=self.right, through=self.through, left_fk=self.left_fk, right_fk=self.right_fk,
                       cover_left=self.cover_left)
            if self.card:
                out["per_right"] = self.card.to_dict()
        if self.kind == "group":
            out.update(entity=self.parent, fk=self.fk, share=self.share)
            if self.card:
                out["members"] = self.card.to_dict()
        return out


@dataclass
class Constraint:
    name: str
    kind: str
    relationship: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.kind, "relationship": self.relationship, **self.params}


@dataclass
class Model:
    name: str
    seed: int
    entities: dict[str, Entity]
    relationships: dict[str, Relationship]
    constraints: list[Constraint]
    order: list[str]
    raw: dict[str, Any]

    def relationship_of(self, constraint: Constraint) -> Relationship:
        return self.relationships[constraint.relationship]



# ---- parsing ------------------------------------------------------------------------------------------------------
def _card(raw: Any, default: tuple[int, int], path: str, errors: list[str], min_floor: int = 0) -> Card:
    if raw is None:
        return Card(default[0], default[1])
    if not isinstance(raw, dict):
        errors.append(f"{path}: must be an object like {{\"min\": 1, \"max\": 3}}")
        return Card(*default)
    extra = set(raw) - {"min", "max", "weights"}
    if extra:
        errors.append(f"{path}: unknown key(s) {sorted(extra)}; allowed: min, max, weights")
    lo, hi = raw.get("min", default[0]), raw.get("max", default[1])
    if not (_int(lo) and _int(hi)):
        errors.append(f"{path}: min and max must be whole numbers")
        return Card(*default)
    if lo < min_floor:
        errors.append(f"{path}.min: must be at least {min_floor}")
    if hi < lo:
        errors.append(f"{path}: max ({hi}) is below min ({lo})")
    if hi > MAX_CARD:
        errors.append(f"{path}.max: must not exceed {MAX_CARD}")
    weights = None
    if raw.get("weights") is not None:
        w = raw["weights"]
        parsed: dict[int, float] = {}
        if not isinstance(w, dict) or not w:
            errors.append(f"{path}.weights: must be an object such as {{\"1\": 0.7, \"2\": 0.3}}")
        else:
            for k, v in w.items():
                try:
                    count = int(k)
                except (TypeError, ValueError):
                    errors.append(f"{path}.weights: key '{k}' is not a whole number")
                    continue
                if not (_num(v) and v > 0):
                    errors.append(f"{path}.weights['{k}']: must be a positive number")
                elif not lo <= count <= hi:
                    errors.append(f"{path}.weights: count {count} is outside min..max ({lo}..{hi})")
                else:
                    parsed[count] = float(v)
            weights = parsed or None
    return Card(lo if _int(lo) else default[0], hi if _int(hi) else default[1], weights)


def _check_gen(gen: Any, path: str, errors: list[str], earlier: dict[str, str]) -> dict[str, Any] | None:
    """Validate one column generator; return it with its defaults filled in. ``earlier``: declared column -> generator type."""
    if not isinstance(gen, dict) or not isinstance(gen.get("type"), str):
        errors.append(f"{path}: needs a generator object with a 'type'; one of {sorted(GEN_SPECS)}")
        return None
    kind = gen["type"]
    if kind not in GEN_SPECS:
        errors.append(f"{path}.type: unknown generator '{kind}'; one of {sorted(GEN_SPECS)}")
        return None
    before = len(errors)
    required, optional = GEN_SPECS[kind]
    unknown = set(gen) - required - optional - {"type", "null_share"}
    if unknown:
        errors.append(f"{path}: unknown key(s) {sorted(unknown)} for '{kind}'; allowed: {sorted(required | optional | {'null_share'})}")
    for key in sorted(required - set(gen)):
        errors.append(f"{path}: '{kind}' needs '{key}'")
    out = dict(gen)

    if "null_share" in gen and not (_num(gen["null_share"]) and 0 <= gen["null_share"] <= 1):
        errors.append(f"{path}.null_share: must be a number from 0 to 1")

    if kind == "id":
        out.setdefault("prefix", "")
        out.setdefault("digits", 6)
        out.setdefault("start", 1)
        if not isinstance(out["prefix"], str):
            errors.append(f"{path}.prefix: must be text")
        if not (_int(out["digits"]) and 1 <= out["digits"] <= 12):
            errors.append(f"{path}.digits: must be a whole number from 1 to 12")
        if not (_int(out["start"]) and out["start"] >= 0):
            errors.append(f"{path}.start: must be a whole number, 0 or more")
    elif kind == "choice":
        has_w, has_v = gen.get("weights") is not None, gen.get("values") is not None
        if has_w == has_v:
            errors.append(f"{path}: 'choice' needs exactly one of 'weights' or 'values'")
        elif has_w:
            w = gen["weights"]
            if not isinstance(w, dict) or not w or not all(_num(v) and v > 0 for v in w.values()):
                errors.append(f"{path}.weights: must be an object of label -> positive number")
        elif not isinstance(gen["values"], list) or not gen["values"]:
            errors.append(f"{path}.values: must be a non-empty list")
    elif kind in ("int", "float"):
        lo, hi = gen.get("min"), gen.get("max")
        ok = (_int(lo) and _int(hi)) if kind == "int" else (_num(lo) and _num(hi))
        if "min" in gen and "max" in gen:
            if not ok:
                errors.append(f"{path}: min and max must be {'whole numbers' if kind == 'int' else 'numbers'}")
            elif hi < lo:
                errors.append(f"{path}: max ({hi}) is below min ({lo})")
        if kind == "float":
            out.setdefault("round", 2)
            if not (_int(out["round"]) and 0 <= out["round"] <= 8):
                errors.append(f"{path}.round: must be a whole number from 0 to 8")
    elif kind == "date":
        lo, hi = _date(gen.get("start")), _date(gen.get("end"))
        if "start" in gen and lo is None:
            errors.append(f"{path}.start: must be an ISO date like 2024-01-31")
        if "end" in gen and hi is None:
            errors.append(f"{path}.end: must be an ISO date like 2024-01-31")
        if lo and hi and hi < lo:
            errors.append(f"{path}: end is before start")
    elif kind == "bool":
        out.setdefault("p", 0.5)
        if not (_num(out["p"]) and 0 <= out["p"] <= 1):
            errors.append(f"{path}.p: must be a number from 0 to 1")
    elif kind == "const":
        if isinstance(gen.get("value"), (dict, list)):
            errors.append(f"{path}.value: must be a single value, not a list or object")
    elif kind == "date_after":
        out.setdefault("min_days", 0)
        out.setdefault("max_days", 365)
        base = gen.get("column")
        if "column" in gen:
            if base not in earlier:
                errors.append(f"{path}.column: '{base}' must be a date column declared earlier in the same entity")
            elif earlier[base] not in _DATE_TYPES:
                errors.append(f"{path}.column: '{base}' is not a date column")
        if not (_int(out["min_days"]) and out["min_days"] >= 0):
            errors.append(f"{path}.min_days: must be a whole number, 0 or more")
        if not (_int(out["max_days"]) and _int(out["min_days"]) and out["max_days"] >= out["min_days"]):
            errors.append(f"{path}.max_days: must be a whole number, not below min_days")
    elif kind == "digits":
        out.setdefault("prefix", "")
        out.setdefault("unique", False)
        n = gen.get("length")
        if "length" in gen and not (_int(n) and 1 <= n <= 18):
            errors.append(f"{path}.length: must be a whole number from 1 to 18")
        if not (isinstance(out["prefix"], str) and (out["prefix"] == "" or out["prefix"].isdigit())):
            errors.append(f"{path}.prefix: must be digits only")
        elif _int(n) and len(out["prefix"]) >= n:
            errors.append(f"{path}.prefix: must be shorter than length")
        if not isinstance(out["unique"], bool):
            errors.append(f"{path}.unique: must be true or false")
    return out if len(errors) == before else None


def parse(payload: Any, counts: dict[str, int] | None = None) -> tuple[Model | None, list[str], list[str]]:
    """Return (model, errors, warnings). The model is None whenever there is an error."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return None, ["the model must be a JSON object"], warnings
    extra = set(payload) - {"name", "seed", "entities", "relationships", "constraints", "description"}
    if extra:
        errors.append(f"unknown top-level key(s) {sorted(extra)}; allowed: name, description, seed, entities, relationships, constraints")
    name = payload.get("name") or "er_model"
    if not isinstance(name, str):
        errors.append("name: must be text")
        name = "er_model"
    seed = payload.get("seed", DEFAULT_SEED)
    if not (_int(seed) and 0 <= seed <= SEED_MAX):
        errors.append(f"seed: must be a whole number from 0 to {SEED_MAX}")
        seed = DEFAULT_SEED

    # ---- entities (names, keys, counts) ----
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list) or not raw_entities:
        return None, errors + ["entities: needs a non-empty list"], warnings
    if len(raw_entities) > MAX_ENTITIES:
        errors.append(f"entities: at most {MAX_ENTITIES} entities")
    entities: dict[str, Entity] = {}
    raw_columns: dict[str, list[Any]] = {}
    for i, raw in enumerate(raw_entities[:MAX_ENTITIES]):
        path = f"entities[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{path}: must be an object")
            continue
        unknown = set(raw) - {"name", "key", "count", "columns", "description"}
        if unknown:
            errors.append(f"{path}: unknown key(s) {sorted(unknown)}; allowed: name, key, count, columns, description")
        ename = raw.get("name")
        if not (isinstance(ename, str) and _IDENT.match(ename)):
            errors.append(f"{path}.name: needs a name of letters, digits and underscores, not starting with a digit")
            continue
        if ename in entities:
            errors.append(f"{path}.name: entity '{ename}' is declared twice")
            continue
        key = raw.get("key", f"{ename}_id")
        if not (isinstance(key, str) and _IDENT.match(key)):
            errors.append(f"{path}.key: needs a column name of letters, digits and underscores")
            key = f"{ename}_id"
        count = raw.get("count")
        if count is not None and not (_int(count) and 1 <= count <= MAX_ROOT_COUNT):
            errors.append(f"{path}.count: must be a whole number from 1 to {MAX_ROOT_COUNT}")
            count = None
        cols = raw.get("columns", [])
        if not isinstance(cols, list):
            errors.append(f"{path}.columns: must be a list")
            cols = []
        if len(cols) > MAX_COLUMNS:
            errors.append(f"{path}.columns: at most {MAX_COLUMNS} columns")
            cols = cols[:MAX_COLUMNS]
        entities[ename] = Entity(ename, key, {}, count)
        raw_columns[ename] = cols

    # ---- relationships ----
    relationships: dict[str, Relationship] = {}
    raw_rels = payload.get("relationships", [])
    if not isinstance(raw_rels, list):
        errors.append("relationships: must be a list")
        raw_rels = []
    drivers: dict[str, list[str]] = {}
    fk_owner: dict[tuple[str, str], str] = {}

    def need_entity(ref: Any, path: str) -> bool:
        if ref in entities:
            return True
        errors.append(f"{path}: unknown entity {ref!r}; declared entities: {sorted(entities)}")
        return False

    def add_fk(child: str, column: str, rel: str, parent: str, path: str) -> None:
        if column == entities[child].key:
            errors.append(f"{path}: foreign key '{column}' clashes with the key of '{child}'")
            return
        if (child, column) in fk_owner:
            errors.append(f"{path}: '{child}' already has a foreign key '{column}' (from relationship '{fk_owner[(child, column)]}')")
            return
        fk_owner[(child, column)] = rel
        entities[child].fks.append(Fk(column, rel, parent, entities[parent].key))

    for i, raw in enumerate(raw_rels):
        path = f"relationships[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{path}: must be an object")
            continue
        kind = raw.get("type")
        if kind not in REL_TYPES:
            errors.append(f"{path}.type: must be one of {list(REL_TYPES)}")
            continue
        rname = raw.get("name") or f"rel_{i + 1}"
        if not (isinstance(rname, str) and _IDENT.match(rname)):
            errors.append(f"{path}.name: needs a name of letters, digits and underscores")
            continue
        if rname in relationships:
            errors.append(f"{path}.name: relationship '{rname}' is declared twice")
            continue
        allowed = {"type", "name", "description"}
        if kind == "one_to_many":
            allowed |= {"parent", "child", "fk", "children"}
        elif kind == "many_to_one":
            allowed |= {"parent", "child", "fk", "null_share", "skew"}
        elif kind == "many_to_many":
            allowed |= {"left", "right", "through", "left_fk", "right_fk", "per_right", "cover_left"}
        else:
            allowed |= {"entity", "fk", "share", "members"}
        unknown = set(raw) - allowed
        if unknown:
            errors.append(f"{path}: unknown key(s) {sorted(unknown)} for '{kind}'; allowed: {sorted(allowed)}")
        rel = Relationship(rname, kind)

        if kind in ("one_to_many", "many_to_one"):
            parent, child = raw.get("parent"), raw.get("child")
            ok = need_entity(parent, f"{path}.parent") & need_entity(child, f"{path}.child")
            if not ok:
                continue
            if parent == child:
                errors.append(f"{path}: parent and child are both '{parent}'; use a 'group' relationship for rows of one entity that belong together")
                continue
            rel.parent, rel.child = parent, child
            rel.fk = raw.get("fk", entities[parent].key)
            if not (isinstance(rel.fk, str) and _IDENT.match(rel.fk)):
                errors.append(f"{path}.fk: needs a column name of letters, digits and underscores")
                continue
            if kind == "one_to_many":
                rel.card = _card(raw.get("children"), (1, 3), f"{path}.children", errors)
                drivers.setdefault(child, []).append(rname)
            else:
                for k in ("null_share", "skew"):
                    v = raw.get(k, 0)
                    if not (_num(v) and 0 <= v <= (1 if k == "null_share" else 2)):
                        errors.append(f"{path}.{k}: must be a number from 0 to {1 if k == 'null_share' else 2}")
                    else:
                        setattr(rel, k, float(v))
            add_fk(child, rel.fk, rname, parent, path)

        elif kind == "many_to_many":
            left, right, through = raw.get("left"), raw.get("right"), raw.get("through")
            ok = need_entity(left, f"{path}.left") & need_entity(right, f"{path}.right") & need_entity(through, f"{path}.through")
            if not ok:
                continue
            if len({left, right, through}) < 3:
                errors.append(f"{path}: left, right and through must be three different entities")
                continue
            rel.left, rel.right, rel.through = left, right, through
            rel.left_fk = raw.get("left_fk", entities[left].key)
            rel.right_fk = raw.get("right_fk", entities[right].key)
            if rel.left_fk == rel.right_fk:
                errors.append(f"{path}: left_fk and right_fk are both '{rel.left_fk}'; name them differently")
                continue
            rel.card = _card(raw.get("per_right"), (1, 2), f"{path}.per_right", errors, min_floor=1)
            cover = raw.get("cover_left", True)
            if not isinstance(cover, bool):
                errors.append(f"{path}.cover_left: must be true or false")
            else:
                rel.cover_left = cover
            drivers.setdefault(through, []).append(rname)
            add_fk(through, rel.left_fk, rname, left, f"{path}.left_fk")
            add_fk(through, rel.right_fk, rname, right, f"{path}.right_fk")

        else:  # group
            entity = raw.get("entity")
            if not need_entity(entity, f"{path}.entity"):
                continue
            rel.parent = rel.child = entity
            rel.fk = raw.get("fk", f"parent_{entities[entity].key}")
            if not (isinstance(rel.fk, str) and _IDENT.match(rel.fk)):
                errors.append(f"{path}.fk: needs a column name of letters, digits and underscores")
                continue
            share = raw.get("share", 0.2)
            if not (_num(share) and 0 < share <= 0.9):
                errors.append(f"{path}.share: must be a number above 0 and up to 0.9 (the share of rows that start a group)")
            else:
                rel.share = float(share)
            rel.card = _card(raw.get("members"), (1, 5), f"{path}.members", errors, min_floor=1)
            if any(r.kind == "group" and r.parent == entity for r in relationships.values()):
                errors.append(f"{path}: '{entity}' already has a group relationship")
                continue
            add_fk(entity, rel.fk, rname, entity, path)
        relationships[rname] = rel

    # ---- columns (now that the foreign-key names are known) ----
    for ename, entity in entities.items():
        fk_names = {f.column for f in entity.fks}
        earlier: dict[str, str] = {}
        seen: set[str] = set()
        key_gen = None
        for j, raw in enumerate(raw_columns[ename]):
            path = f"entities[{list(entities).index(ename)}].columns[{j}]"
            if not isinstance(raw, dict):
                errors.append(f"{path}: must be an object")
                continue
            unknown = set(raw) - {"name", "gen", "description"}
            if unknown:
                errors.append(f"{path}: unknown key(s) {sorted(unknown)}; allowed: name, gen, description")
            cname = raw.get("name")
            if not (isinstance(cname, str) and _IDENT.match(cname)):
                errors.append(f"{path}.name: needs a name of letters, digits and underscores")
                continue
            if cname in seen:
                errors.append(f"{path}.name: column '{cname}' is declared twice in '{ename}'")
                continue
            seen.add(cname)
            if cname in fk_names:
                warnings.append(f"{ename}.{cname} is filled from its relationship, so the generator written for it is ignored")
                continue
            if cname == entity.key:
                gen = _check_gen(raw.get("gen"), f"{path}.gen", errors, earlier)
                if gen is not None and gen["type"] != "id":
                    errors.append(f"{path}.gen.type: the key '{cname}' must be generated with type 'id' so it is unique")
                elif gen is not None:
                    key_gen = gen
                continue
            gen = _check_gen(raw.get("gen"), f"{path}.gen", errors, earlier)
            if gen is not None:
                entity.columns.append(Column(cname, gen))
                earlier[cname] = gen["type"]
        if key_gen is None and entity.key not in seen:
            key_gen = {"type": "id", "prefix": ename[:3].upper(), "digits": 6, "start": 1}
            warnings.append(f"{ename}.{entity.key}: key not declared; generated as '{key_gen['prefix']}000001', '{key_gen['prefix']}000002', ...")
        entity.key_gen = key_gen or {"type": "id", "prefix": "", "digits": 6, "start": 1}
        if entity.key in fk_names:
            errors.append(f"entity '{ename}': its key '{entity.key}' is also a foreign key")

    # ---- who decides each entity's row count ----
    for ename, entity in entities.items():
        rels = drivers.get(ename, [])
        if len(rels) > 1:
            errors.append(f"entity '{ename}' would get its row count from more than one relationship ({', '.join(rels)}); only one one_to_many or many_to_many may create its rows")
        elif rels:
            entity.driver = rels[0]
            if entity.count is not None:
                warnings.append(f"{ename}.count is ignored: its rows are created by relationship '{rels[0]}'")
        elif entity.count is None:
            errors.append(f"entity '{ename}' has no parent relationship creating its rows, so it needs a 'count'")

    # ---- constraints ----
    constraints: list[Constraint] = []
    raw_cons = payload.get("constraints", [])
    if not isinstance(raw_cons, list):
        errors.append("constraints: must be a list")
        raw_cons = []
    for i, raw in enumerate(raw_cons):
        path = f"constraints[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{path}: must be an object")
            continue
        kind = raw.get("type")
        if kind not in CONSTRAINT_TYPES:
            errors.append(f"{path}.type: must be one of {list(CONSTRAINT_TYPES)}")
            continue
        rel = relationships.get(raw.get("relationship"))
        if rel is None:
            errors.append(f"{path}.relationship: unknown relationship {raw.get('relationship')!r}; declared: {sorted(relationships)}")
            continue
        if rel.kind == "many_to_many":
            errors.append(f"{path}.relationship: constraints on a many_to_many relationship are not supported yet; use its one_to_many sides")
            continue
        child_name = rel.child if rel.kind != "group" else rel.parent
        parent_name = rel.parent
        child, parent = entities[child_name], entities[parent_name]
        allowed = {"type", "relationship", "description"}
        params: dict[str, Any] = {}
        if kind == "date_not_before_parent":
            allowed |= {"column", "parent_column", "max_days_after"}
        elif kind == "same_as_parent":
            allowed |= {"column", "parent_column"}
        elif kind == "status_follows_parent":
            allowed |= {"column", "parent_column", "when", "set"}
        else:
            allowed |= {"parent_column", "child_column", "fn"}
        unknown = set(raw) - allowed
        if unknown:
            errors.append(f"{path}: unknown key(s) {sorted(unknown)} for '{kind}'; allowed: {sorted(allowed)}")

        def column(entity: Entity, key: str, label: str, want_dates: bool = False, optional: bool = False):
            value = raw.get(key)
            if value is None:
                if not optional:
                    errors.append(f"{path}.{key}: required")
                return None
            col = next((c for c in entity.columns if c.name == value), None)
            if col is None:
                errors.append(f"{path}.{key}: '{value}' is not a declared column of '{entity.name}' (the {label}); key and foreign-key columns cannot be used")
                return None
            if want_dates and col.gen["type"] not in _DATE_TYPES:
                errors.append(f"{path}.{key}: '{value}' is not a date column")
                return None
            return value

        if kind == "date_not_before_parent":
            c, p = column(child, "column", "child", True), column(parent, "parent_column", "parent", True)
            days = raw.get("max_days_after", 365)
            if not (_int(days) and days >= 0):
                errors.append(f"{path}.max_days_after: must be a whole number, 0 or more")
            params = {"column": c, "parent_column": p, "max_days_after": days}
        elif kind == "same_as_parent":
            params = {"column": column(child, "column", "child"), "parent_column": column(parent, "parent_column", "parent")}
        elif kind == "status_follows_parent":
            c, p = column(child, "column", "child"), column(parent, "parent_column", "parent")
            when = raw.get("when")
            when = when if isinstance(when, list) else [when]
            if "when" not in raw or not when or any(isinstance(v, (dict, list)) for v in when):
                errors.append(f"{path}.when: needs a value (or a list of values) the parent column can take")
            if "set" not in raw or isinstance(raw.get("set"), (dict, list)):
                errors.append(f"{path}.set: needs the value the child column takes")
            params = {"column": c, "parent_column": p, "when": when, "set": raw.get("set")}
        else:  # aggregate
            fn = raw.get("fn", "sum")
            if fn not in AGG_FNS:
                errors.append(f"{path}.fn: must be one of {list(AGG_FNS)}")
            p = column(parent, "parent_column", "parent")
            c = column(child, "child_column", "child", optional=(fn == "count"))
            params = {"parent_column": p, "child_column": c, "fn": fn}
        constraints.append(Constraint(raw.get("name") or f"{kind}_{i + 1}", kind, rel.name, params))

    # ---- parent-first order (and loops) ----
    order, cycle = _order(entities, relationships)
    if cycle:
        errors.append(f"the relationships form a loop between {', '.join(cycle)}; a parent must exist before its child, so make one link a many_to_one lookup that does not create rows or remove it")

    model = Model(str(name), seed, entities, relationships, constraints, order, payload) if not errors else None
    if model is not None:
        problems = check_budget(model, counts)
        if problems:
            return None, problems, warnings
    return model, errors, warnings


def _order(entities: dict[str, Entity], relationships: dict[str, Relationship]) -> tuple[list[str], list[str]]:
    """Parents before children (declaration order breaks ties). Returns (order, entities left in a loop)."""
    edges: set[tuple[str, str]] = set()
    for rel in relationships.values():
        if rel.kind in ("one_to_many", "many_to_one"):
            edges.add((rel.parent, rel.child))
        elif rel.kind == "many_to_many":
            edges.add((rel.left, rel.through))
            edges.add((rel.right, rel.through))
    remaining = list(entities)
    order: list[str] = []
    while remaining:
        ready = [e for e in remaining if not any(b == e and a in remaining for a, b in edges)]
        if not ready:
            return order, remaining
        order.append(ready[0])
        remaining.remove(ready[0])
    return order, []


# ---- sizes --------------------------------------------------------------------------------------------------------
def expected_rows(model: Model, counts: dict[str, int] | None = None) -> dict[str, float]:
    """Expected rows per entity, in generation order (a root entity's count, or parents x average children)."""
    counts = counts or {}
    rows: dict[str, float] = {}
    for name in model.order:
        entity = model.entities[name]
        rel = model.relationships.get(entity.driver) if entity.driver else None
        if rel is None:
            rows[name] = float(counts.get(name, entity.count or 0))
        elif rel.kind == "one_to_many":
            rows[name] = rows[rel.parent] * rel.card.expected()
        else:  # many_to_many: pairs
            rows[name] = rows[rel.right] * rel.card.expected()
    return rows


def check_counts(model: Model, counts: dict[str, Any] | None) -> list[str]:
    """Problems with a per-entity count override: only root entities have a count."""
    problems = []
    for name, value in (counts or {}).items():
        entity = model.entities.get(name)
        if entity is None:
            problems.append(f"counts.{name}: unknown entity")
        elif entity.driver:
            problems.append(f"counts.{name}: its rows come from relationship '{entity.driver}'; override the count of a root entity instead")
        elif not (_int(value) and 1 <= value <= MAX_ROOT_COUNT):
            problems.append(f"counts.{name}: must be a whole number from 1 to {MAX_ROOT_COUNT}")
    return problems


def check_budget(model: Model, counts: dict[str, int] | None = None) -> list[str]:
    total = sum(expected_rows(model, counts).values())
    if total > MAX_TOTAL_ROWS:
        return [f"this model would create about {int(total):,} rows in total; the limit for one run is {MAX_TOTAL_ROWS:,}. Lower a count or a children range."]
    return []


# ---- the plan the review screen shows ----------------------------------------------------------------------------
def describe(model: Model, counts: dict[str, int] | None = None) -> dict[str, Any]:
    rows = expected_rows(model, counts)
    entities = []
    for name in model.order:
        e = model.entities[name]
        rel = model.relationships.get(e.driver) if e.driver else None
        role = "root" if rel is None else ("junction" if rel.kind == "many_to_many" else "child")
        columns = [{"name": e.key, "kind": "key", "generator": "id"}]
        columns += [{"name": f.column, "kind": "fk", "references": f"{f.parent}.{f.parent_key}", "relationship": f.relationship} for f in e.fks]
        columns += [{"name": c.name, "kind": "column", "generator": c.gen["type"]} for c in e.columns]
        entities.append({"name": name, "key": e.key, "role": role, "created_by": e.driver, "rows_expected": round(rows[name]), "columns": columns})
    return {
        "order": list(model.order),
        "entities": entities,
        "relationships": [r.to_dict() for r in model.relationships.values()],
        "constraints": [c.to_dict() for c in model.constraints],
        "projected_rows": round(sum(rows.values())),
        "limits": {"max_root_count": MAX_ROOT_COUNT, "max_total_rows": MAX_TOTAL_ROWS, "max_entities": MAX_ENTITIES},
    }
