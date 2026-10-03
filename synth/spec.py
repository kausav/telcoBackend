"""Generation spec: the executable description of how ONE scenario's data is produced.

A source standard (TMF, FHIR, BIAN, ACORD, ...) and a scenario's variable list say *what* the columns are:
names, types, allowed values, patterns. They cannot say how the columns behave together: which ones belong to
the customer and which to each event, what must have happened before a timestamp may exist, which fact is
absent unless another holds, how the requested scenario type changes the outcomes. That behaviour is the spec.

A spec is data, never code. It is produced for the exact scenario (industry, domain, scenario type, business
scenario and the confirmed columns) by ``synth.compiler`` - a language model that reads the column definitions
and the source documents' descriptions - and is accepted only after this module validates its structure and the
simulator has run it and checked its own rules plus the source-derived value contract. It is then stored
(collection ``generation_specs``) and pinned to the confirmed scenario, so a seed and an ``asOf`` reproduce a
dataset exactly. No industry knowledge lives in the application.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth import samplers
from synth.expr import Expr

SPEC_VERSION = 1
ENGINE = "event_rows/v1"

DTYPES = ("string", "integer", "float", "boolean", "categorical", "datetime", "date")


class SpecColumn(BaseModel):
    """One value the simulator produces: a delivered column (entity / event) or a hidden driver (latent)."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["entity", "event", "latent"]
    dtype: Literal["string", "integer", "float", "boolean", "categorical", "datetime", "date"] = "string"
    column: str | None = None        # delivered column name; None for a latent
    description: str = ""
    precision: int | None = None     # float columns: decimals in the delivered value
    origin: Literal["definition", "llm"] = "definition"
    placeholder: bool = False        # nothing but a neutral token can be drawn from the column's own definition
    timestamp_format: str | None = None   # datetime/date columns: strftime pattern (None = RFC 3339)


class Invariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    level: Literal["record", "entity", "dataset"] = "record"
    severity: Literal["error", "warn"] = "error"
    expr: str
    message: str = ""
    requires: list[str] = Field(default_factory=list)   # column ids; skipped when a column is absent

    @model_validator(mode="after")
    def _compile(self) -> "Invariant":
        Expr(self.expr)  # fail fast on invalid / unsafe expressions
        return self


class Target(BaseModel):
    """A distribution expectation: the observed statistic must fall inside [min, max]."""
    model_config = ConfigDict(extra="forbid")
    id: str
    stat: Literal["share", "mean", "median"] = "share"
    column: str | None = None        # mean/median: column the statistic is computed on
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
        if self.stat != "share" and not self.column:
            raise ValueError(f"target '{self.id}': {self.stat} needs a column")
        for src in (self.where, self.condition):
            if src:
                Expr(src)
        return self


class Emit(BaseModel):
    """How one column's value is produced for every row (``expr`` or a ``sample``; see ``synth.samplers``).

    ``scope`` entity: drawn once per entity (after its timeline, so it may use ``first_event_at``);
    ``scope`` event: drawn per row, and may read every column drawn before it (the order is derived from what each
    entry reads, so an author never has to sort them).
    ``when`` is an optional guard; when it is false the value is null.
    """
    model_config = ConfigDict(extra="forbid")
    column: str
    scope: Literal["entity", "event"] = "event"
    when: str | None = None
    expr: str | None = None
    sample: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Emit":
        if (self.expr is None) == (self.sample is None):
            raise ValueError(f"emit '{self.column}': give exactly one of expr / sample")
        for src in (self.when, self.expr):
            if src:
                Expr(src)
        if self.sample is not None:
            samplers.validate(self.sample, f"emit '{self.column}'")
        return self


class GenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_version: int = SPEC_VERSION
    engine: str = ENGINE
    source: Literal["llm", "definitions"] = "llm"      # who wrote the behaviour: the compiler's model, or nobody (definitions only)
    scenario: dict[str, Any] = Field(default_factory=dict)
    currency: str = "USD"
    timezone: str = "UTC"
    entity_column: str | None = None
    clock: str | None = None                            # the column carrying each event's own time
    assumptions: list[str] = Field(default_factory=list)
    columns: dict[str, SpecColumn]
    emit: list[Emit] = Field(default_factory=list)
    timeline: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)       # numeric/text parameters read as P['name']
    reference: dict[str, Any] = Field(default_factory=dict)   # tables read as REF['name']
    invariants: list[Invariant] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    revision: int = 0                                          # 0 = the first verified design; each accepted refinement adds one
    reviewed: bool = False                                     # True once the reviewer's findings have been worked through (or none were found)
    design: dict[str, Any] = Field(default_factory=dict)       # the designer's own spec, kept so a refinement can patch it

    @model_validator(mode="after")
    def _check(self) -> "GenerationSpec":
        emitted = [e.column for e in self.emit]
        if len(emitted) != len(set(emitted)):
            raise ValueError("emit declares a column twice")
        if self.emit and set(emitted) != set(self.columns):
            raise ValueError(f"every column needs exactly one emit entry; mismatch: {sorted(set(self.columns) ^ set(emitted))}")
        known = set(self.columns)
        if self.entity_column is not None and self.entity_column not in known:
            raise ValueError(f"entity_column '{self.entity_column}' is not a declared column")
        if self.clock is not None and (self.clock not in known or self.columns[self.clock].dtype != "datetime"):
            raise ValueError(f"clock '{self.clock}' must be a declared datetime column")
        for emit in self.emit:
            kind = self.columns[emit.column].kind
            if kind == "entity" and emit.scope != "entity":
                raise ValueError(f"column '{emit.column}' is an entity column but its emit is event-scoped")
            if kind == "event" and emit.scope != "event":
                raise ValueError(f"column '{emit.column}' is an event column but its emit is entity-scoped")
        for inv in self.invariants:
            unknown = set(inv.requires) - known
            if unknown:
                raise ValueError(f"invariant '{inv.id}' requires unknown columns {sorted(unknown)}")
            names = Expr(inv.expr).names - known - EVAL_VARS
            if names:
                raise ValueError(f"invariant '{inv.id}' references unknown columns {sorted(names)}")
        for t in self.targets:
            if t.column and t.column not in known:
                raise ValueError(f"target '{t.id}' references unknown column '{t.column}'")
            for src in (t.where, t.condition):
                names = (Expr(src).names - known - EVAL_VARS) if src else set()
                if names:
                    raise ValueError(f"target '{t.id}' references unknown columns {sorted(names)}")
        order_by = self.output.get("order_by")
        if order_by is not None and order_by not in known:
            raise ValueError(f"output.order_by '{order_by}' is not a declared column")
        from synth.engines import get_engine

        try:
            engine = get_engine(self.engine)
        except KeyError as exc:
            raise ValueError(str(exc.args[0])) from exc
        check = getattr(engine, "validate", None)
        if check is not None:
            check(self)
        return self

    # ---- views ----------------------------------------------------------------------------------
    @property
    def delivered(self) -> dict[str, str]:
        """column id -> delivered column name, in declaration order (latents are not delivered)."""
        return {cid: c.column for cid, c in self.columns.items() if c.kind != "latent" and c.column}

    @property
    def entity_columns(self) -> list[str]:
        """Delivered column names that describe the entity (constant over its history)."""
        return [c.column for c in self.columns.values() if c.kind == "entity" and c.column]


# names an invariant or target may read besides the columns (REF / P are namespaces, not names)
EVAL_VARS = frozenset({"as_of"})
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED = frozenset({"REF", "P", "prev", "as_of", "event_at", "event_index", "first_event_at", "True", "False", "None", "_choice"})


def safe_identifier(name: str, taken: set[str]) -> str:
    """The id a column is known by inside expressions: its own name when that is a legal, unreserved identifier."""
    from synth.expr import FUNCTIONS

    text = str(name or "").strip()
    if _IDENT.match(text) and text not in RESERVED and text not in FUNCTIONS and text not in taken:
        return text
    base = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "col"
    if base[0].isdigit():
        base = f"c_{base}"
    candidate, n = base, 2
    while candidate in RESERVED or candidate in FUNCTIONS or candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate
