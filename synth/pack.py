"""Behaviour packs: the data-only description of a domain's behaviour.

A pack is what the source JSON standards (TMF, FHIR, ACORD, ...) *cannot* express: which fields
are the same business fact, which resources a scenario actually needs, realistic vocabularies and
distributions, causal rules, and what "correct" means for a dataset (invariants + targets).

Packs are data and live only in MongoDB (collection ``behavior_packs``), next to the other
collections that describe a domain (``industry_source_documents``, ``scenario_variables``); no domain
knowledge is shipped inside the code. A pack is selected by the same keys those collections use
(``industry_key`` / ``domain_key``). The LLM never writes executable behaviour: it may *draft* a
pack, but a pack only becomes active after schema validation here and a human approval
(``status == "active"``).
"""
from __future__ import annotations

import itertools
import logging
import os
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth import samplers
from synth.expr import Expr

logger = logging.getLogger(__name__)


class Bind(BaseModel):
    """How a source/DB column is recognised as a concept. All given criteria must match."""
    model_config = ConfigDict(extra="forbid")
    name: str | None = None          # regex over the normalised (snake_case) column name
    model: str | None = None         # source JSON model (e.g. Order) - exact, case-insensitive
    path: str | None = None          # regex over the source JSON path
    description: str | None = None   # regex over the description (weak signal; rank-penalised)


class Concept(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["entity", "event", "derived"]
    dtype: Literal["string", "integer", "float", "boolean", "categorical", "datetime", "date"] = "string"
    description: str = ""
    required: bool = False           # the engine cannot produce a meaningful dataset without it
    bind: list[Bind] = Field(default_factory=list)
    default_on: bool = False         # kind == "derived": add the column when the scenario needs it
    column_name: str | None = None   # kind == "derived": name of the column to create
    null_token: str | None = None    # value to emit instead of null when the bound column is not nullable
    precision: int | None = None     # float concepts: decimals in the delivered value (None = as generated)


class Exclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str                        # regex over the normalised column name
    reason: str


class Invariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    level: Literal["record", "entity", "dataset"] = "record"
    severity: Literal["error", "warn"] = "error"
    expr: str
    message: str = ""
    requires: list[str] = Field(default_factory=list)   # concept ids; skipped when a column is absent

    @model_validator(mode="after")
    def _compile(self) -> "Invariant":
        Expr(self.expr)  # fail fast on invalid / unsafe expressions
        return self


class Target(BaseModel):
    """A distribution expectation: the observed statistic must fall inside [min, max]."""
    model_config = ConfigDict(extra="forbid")
    id: str
    stat: Literal["share", "mean", "median"]
    concept: str | None = None       # mean/median: concept the statistic is computed on
    condition: str | None = None     # share: expression that must hold for a row to count
    where: str | None = None         # expression restricting the population
    min: float
    max: float
    min_n: int = 100                 # skip (not fail) when fewer rows qualify
    description: str = ""

    @model_validator(mode="after")
    def _compile(self) -> "Target":
        if self.stat == "share" and not self.condition:
            raise ValueError(f"target '{self.id}': share needs a condition")
        if self.stat != "share" and not self.concept:
            raise ValueError(f"target '{self.id}': {self.stat} needs a concept")
        for src in (self.where, self.condition):
            if src:
                Expr(src)
        return self


class Emit(BaseModel):
    """How one concept's value is produced for every row (``expr`` or a ``sample``; see ``synth.samplers``).

    ``scope`` entity: drawn once per entity (after its timeline, so it may use ``first_event_at``);
    ``scope`` event: drawn per row, in declaration order, and may read every earlier concept of the row.
    ``when`` is an optional guard; when it is false the value is null.
    """
    model_config = ConfigDict(extra="forbid")
    concept: str
    scope: Literal["entity", "event"] = "event"
    when: str | None = None
    expr: str | None = None
    sample: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Emit":
        if (self.expr is None) == (self.sample is None):
            raise ValueError(f"emit '{self.concept}': give exactly one of expr / sample")
        for src in (self.when, self.expr):
            if src:
                Expr(src)
        if self.sample is not None:
            samplers.validate(self.sample, f"emit '{self.concept}'")
        return self


def normalize_key(value: Any) -> str:
    """Same normalisation as the ``*_key`` fields of the source/variable collections (``<domain>_and_<sub_domain>``)."""
    text = str(value or "").strip().casefold().replace("&", " and ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text).strip("_"))


class Match(BaseModel):
    """Which requests a pack applies to. Keys are the collections' own ``industry_key`` / ``domain_key``."""
    model_config = ConfigDict(extra="forbid")
    industry_key: str
    domain_key: str
    scenario_ids: list[str] = Field(default_factory=list)   # optional: restrict to e.g. ["LB-01"]
    use_case: list[str] = Field(default_factory=list)
    country: list[str] = Field(default_factory=list)
    type_of_data: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _keys(self) -> "Match":
        self.industry_key, self.domain_key = normalize_key(self.industry_key), normalize_key(self.domain_key)
        if not self.industry_key or not self.domain_key:
            raise ValueError("match.industry_key and match.domain_key are required")
        return self


class BehaviorPack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pack_id: str
    version: int = 1
    status: Literal["draft", "active", "retired"] = "active"
    title: str = ""
    engine: str
    match: Match
    currency: str = "USD"
    timezone: str = "UTC"
    unmapped_policy: Literal["drop"] = "drop"
    entity_concept: str
    assumptions: list[str] = Field(default_factory=list)
    concepts: dict[str, Concept]
    exclusions: list[Exclusion] = Field(default_factory=list)
    reference: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    modes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    invariants: list[Invariant] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)
    emit: list[Emit] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "BehaviorPack":
        emitted = [e.concept for e in self.emit]
        if len(emitted) != len(set(emitted)):
            raise ValueError("emit declares a concept twice")
        if self.emit and (set(emitted) != set(self.concepts)):
            diff = sorted(set(self.concepts) ^ set(emitted))
            raise ValueError(f"every concept needs exactly one emit entry; mismatch: {diff}")
        if self.entity_concept not in self.concepts:
            raise ValueError(f"entity_concept '{self.entity_concept}' is not a declared concept")
        known = set(self.concepts)
        for inv in self.invariants:
            unknown = set(inv.requires) - known
            if unknown:
                raise ValueError(f"invariant '{inv.id}' requires unknown concepts {sorted(unknown)}")
            names = Expr(inv.expr).names - known
            if names:
                raise ValueError(f"invariant '{inv.id}' references unknown concepts {sorted(names)}")
        for t in self.targets:
            if t.concept and t.concept not in known:
                raise ValueError(f"target '{t.id}' references unknown concept '{t.concept}'")
            for src in (t.where, t.condition):
                names = (Expr(src).names - known - {"REF"}) if src else set()
                if names:
                    raise ValueError(f"target '{t.id}' references unknown concepts {sorted(names)}")
        order_by = self.output.get("order_by")
        if order_by is not None and order_by not in known:
            raise ValueError(f"output.order_by '{order_by}' is not a declared concept")
        for name in (re.compile(e.name) for e in self.exclusions):  # validates regexes
            pass
        from synth.engines import get_engine

        try:
            engine = get_engine(self.engine)
        except KeyError as exc:
            raise ValueError(str(exc.args[0])) from exc
        check = getattr(engine, "validate", None)
        if check is not None:
            check(self)
        return self

    # ---- matching ---------------------------------------------------------------------------
    def match_score(self, *, industry_key: str, domain_key: str, scenario_id: str = "", use_case: str = "",
                    country: str = "", type_of_data: str = "") -> int | None:
        """Specificity score, or None when the pack does not apply."""
        m = self.match
        if normalize_key(industry_key) != m.industry_key or normalize_key(domain_key) != m.domain_key:
            return None
        score = 10
        for given, allowed, weight, fold in (
            (scenario_id, m.scenario_ids, 8, str.upper),
            (use_case, m.use_case, 3, normalize_key),
            (country, m.country, 2, str.upper),
            (type_of_data, m.type_of_data, 1, normalize_key),
        ):
            if allowed:
                if fold(str(given or "")) not in {fold(x) for x in allowed}:
                    return None
                score += weight
        return score

    def mode_params(self, mode: str) -> dict[str, Any]:
        """Base model parameters with the scenario-mode overrides deep-merged on top."""
        return deep_merge(self.model, (self.modes.get(mode) or {}).get("model", {}))

    def mode_targets(self, mode: str) -> list[Target]:
        override = {t["id"]: t for t in ((self.modes.get(mode) or {}).get("targets") or [])}
        result: list[Target] = []
        for t in self.targets:
            if t.id in override:
                merged = {**t.model_dump(), **override[t.id]}
                result.append(Target.model_validate(merged))
            else:
                result.append(t)
        return result


def deep_merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base.get(k), v) if k in base else v
        return out
    return over if over is not None else base


# ---- loading -------------------------------------------------------------------------------------
_MAX_PACKS = 500
_CACHE: dict[str, tuple[float, list[BehaviorPack]]] = {}


def _query(filter_: dict[str, Any]) -> list[BehaviorPack]:
    """Packs from ``behavior_packs`` matching ``filter_``; invalid documents are skipped and logged.

    Lazy (no import-time DB access) and cached for ``BEHAVIOR_PACK_CACHE_SECONDS`` (default 60), so
    ``/scenario/generate`` does not issue a query per request. A database failure yields no packs, which
    means "no pack applies" for a proposal (legacy flow) and a clear error for a draft pinned to a pack.
    """
    ttl = float(os.getenv("BEHAVIOR_PACK_CACHE_SECONDS", "60"))
    key = repr(sorted(filter_.items()))
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    out: list[BehaviorPack] = []
    held: set[tuple[str, int]] | None = None
    try:
        from core.seed import ensure_seeded
        from models._helpers import collection_name
        from models.database import get_database

        ensure_seeded()
        collection = get_database()[collection_name("MONGODB_BEHAVIOR_PACKS_COLLECTION", "behavior_packs")]
        rows = collection.find(filter_, {"_id": 0})
        for row in itertools.islice(rows, _MAX_PACKS):
            if not isinstance(row, dict):
                continue
            try:
                out.append(BehaviorPack.model_validate(row))
            except Exception:
                logger.exception("Ignoring invalid behaviour pack %s in MongoDB", row.get("pack_id"))
        held = {(str(r.get("pack_id")), int(r.get("version") or 0))
                for r in itertools.islice(collection.find({}, {"_id": 0, "pack_id": 1, "version": 1}), _MAX_PACKS)
                if isinstance(r, dict)}
    except Exception as exc:
        logger.warning("behaviour packs not loaded from MongoDB: %s", exc)
    out.extend(_bundled_packs(filter_, held))
    if held is not None:
        _CACHE[key] = (now, out)          # a database failure is not cached: it is retried on the next call
    return out


def _bundled_packs(filter_: dict[str, Any], held: set[tuple[str, int]] | None) -> list[BehaviorPack]:
    """The packs the application ships, for (pack id, version) pairs MongoDB holds no document of.

    MongoDB stays authoritative for what it holds: a pack version it has (in any status) is never taken from the
    bundle, so editing or retiring it there is always honoured. The bundle covers what MongoDB lacks - a pack
    version shipped after the database was seeded, or everything when the database is unreachable or rejects the
    insert - so a newer bundled version is used instead of an older stored one.
    """
    from core.seed import bundled

    def wanted(row: dict[str, Any]) -> bool:
        for field, expected in filter_.items():
            value = row.get(field)
            if isinstance(expected, dict) and "$in" in expected:
                if value not in expected["$in"]:
                    return False
            elif value != expected:
                return False
        return True

    found: list[BehaviorPack] = []
    for row in bundled("behavior_packs"):
        if (held is not None and (str(row.get("pack_id")), int(row.get("version") or 0)) in held) or not wanted(row):
            continue
        try:
            found.append(BehaviorPack.model_validate(row))
        except Exception:
            logger.exception("Ignoring invalid bundled behaviour pack %s", row.get("pack_id"))
    return found


def clear_cache() -> None:
    _CACHE.clear()


def find_pack(*, industry_key: str, domain_key: str, scenario_id: str = "", use_case: str = "",
              country: str = "", type_of_data: str = "",
              packs: list[BehaviorPack] | None = None) -> BehaviorPack | None:
    """Best-matching active pack: most specific match, then highest version."""
    if packs is None:
        packs = _query({"status": "active"})
    best: tuple[int, int, BehaviorPack] | None = None
    for p in packs:
        if p.status != "active":
            continue
        score = p.match_score(industry_key=industry_key, domain_key=domain_key, scenario_id=scenario_id,
                              use_case=use_case, country=country, type_of_data=type_of_data)
        if score is not None and (best is None or (score, p.version) > (best[0], best[1])):
            best = (score, p.version, p)
    return best[2] if best else None


def get_pack(pack_id: str, version: int | None = None) -> BehaviorPack | None:
    """A specific pack. Retired packs stay loadable so drafts approved earlier keep generating the same data."""
    filter_: dict[str, Any] = {"pack_id": pack_id, "status": {"$in": ["active", "retired"]}}
    if version is not None:
        filter_["version"] = int(version)
    found = _query(filter_)
    return max(found, key=lambda p: p.version) if found else None
