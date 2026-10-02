"""Orchestration: confirmed variables -> a verified generation spec -> dataset + honest validation report.

This is the single entry point the API layer calls. It knows nothing about FastAPI or MongoDB (the spec store hides
that) and nothing about any industry: what a column means comes from its definition, and how the columns of one
scenario behave together comes from a spec the compiler's language model designed and the simulator verified.

Designing a spec takes model calls, so it never sits on a request that has to answer quickly: ``warm_spec`` starts it in
the background (at proposal, import and confirmation), ``pin_spec`` names the spec a confirmed scenario will use without
waiting for it, and only ``resolve_spec`` - on the path that generates data and is allowed to wait - waits for a design
in flight or runs one. A spec is identified by a hash of everything it is designed from, so all of these agree on which
spec is meant before any of it exists.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from synth import store
from synth.clock import RunContext
from synth.compiler import SpecCompiler, restrict, spec_key, verify
from synth.contract import build_contract
from synth.engines import get_engine
from synth.projection import parse_columns, project
from synth.scorer import score_rows
from synth.sources import source_notes
from synth.spec import GenerationSpec

logger = logging.getLogger(__name__)

BRIEF_KEYS = ("industry", "domain", "country", "use_case", "scenario_type", "business_scenario", "business_response",
              "expected_outcome", "type_of_data", "entity_key")


@dataclass
class SpecResolution:
    spec: GenerationSpec | None
    key: str | None
    status: str                      # pinned | cached | restricted | compiled | rejected | unavailable | unusable
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.spec is not None


@dataclass
class SpecGeneration:
    records: list[dict[str, Any]]            # flat rows, column name -> formatted value
    fields: list[str]
    entity_columns: list[str]                # delivered columns that describe the entity (constant over its history)
    clock_column: str | None                 # delivered column carrying each event's own time
    validation_report: dict[str, Any]


def make_brief(source: dict[str, Any]) -> dict[str, Any]:
    """The scenario facts a spec is designed from, from a confirmed-scenario context or a proposal request."""
    brief = {k: source.get(k) for k in BRIEF_KEYS}
    brief["industry"] = source.get("industry") or source.get("industry_type")
    return {k: (str(v).strip() if isinstance(v, str) else v) for k, v in brief.items() if v not in (None, "")}


def _names(variables: list[dict[str, Any]]) -> set[str]:
    return {str(v["name"]) for v in variables if isinstance(v, dict) and v.get("name")}


def _covers(spec: GenerationSpec, names: set[str]) -> bool:
    """True when ``spec`` was designed for exactly these columns."""
    return set(spec.delivered.values()) == names


def _describe(spec: GenerationSpec | None, key: str | None, status: str, problems: list[str] | None = None,
              **extra: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"status": status, "spec_key": key, **extra}
    if spec is not None:
        report.update(source=spec.source, assumptions=spec.assumptions, currency=spec.currency, timezone=spec.timezone,
                      invariants=len(spec.invariants), targets=len(spec.targets), columns=len(spec.delivered))
        if spec.warnings:
            report["review_warnings"] = spec.warnings
    if problems:
        report["problems"] = problems[:12]
    return report


_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}


def _claim(key: str) -> bool:
    """True when the caller is the one that should design ``key`` (nobody else is designing it right now)."""
    with _FLIGHT_LOCK:
        if key in _IN_FLIGHT:
            return False
        _IN_FLIGHT[key] = threading.Event()
        return True


def _release(key: str) -> None:
    with _FLIGHT_LOCK:
        event = _IN_FLIGHT.pop(key, None)
    if event is not None:
        event.set()


def _wait(key: str) -> None:
    """Block until the design of ``key`` that is in flight ends (bounded by the compile budget)."""
    from synth.compiler import SpecCompiler as _C

    with _FLIGHT_LOCK:
        event = _IN_FLIGHT.get(key)
    if event is not None:
        event.wait(timeout=_C._budget() + 30.0)


def in_flight(key: str | None) -> bool:
    with _FLIGHT_LOCK:
        return key in _IN_FLIGHT


def design_key(variables: list[dict[str, Any]], brief: dict[str, Any]) -> str | None:
    """The key of the spec for exactly these variables and this scenario (None when there is nothing to design for)."""
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    return spec_key(brief, usable) if usable else None


def ensure_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None,
                compiler: SpecCompiler | None = None) -> SpecResolution:
    """The verified spec for exactly these variables and this scenario.

    In order: a stored spec for the same inputs; the spec of ``hint_key`` narrowed to these variables (the proposal's
    spec after the user deleted columns); a fresh compile by the language model. A result that is not a verified model
    spec (``rejected`` / ``unavailable``) is never stored, so a later request tries again.
    """
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    if not usable:
        return SpecResolution(None, None, "unusable", _describe(None, None, "unusable", ["no variables"]))
    key = spec_key(brief, usable)
    names = _names(usable)

    spec = store.load(key)
    if spec is not None and _covers(spec, names):
        return SpecResolution(spec, key, "cached", _describe(spec, key, "cached"))

    hint = store.load(hint_key) if hint_key and hint_key != key else None
    if hint is not None and names <= set(hint.delivered.values()):
        narrowed = restrict(hint, names)
        problems, _ = verify(narrowed, usable, aggregational=str(brief.get("type_of_data") or "").lower() == "aggregational")
        if not problems and names == set(narrowed.delivered.values()):
            stored = store.save(key, narrowed, {"brief": brief, "derived_from": hint_key})
            return SpecResolution(narrowed, key, "restricted", _describe(narrowed, key, "restricted", persisted=stored))

    if store.recently_failed(key):
        return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", ["a recent attempt to design this spec failed"]))
    if not _claim(key):
        _wait(key)                              # the same design is already running: use its result instead of repeating it
        spec = store.load(key)
        if spec is not None and _covers(spec, names):
            return SpecResolution(spec, key, "cached", _describe(spec, key, "cached"))
        return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", ["the design that was in progress did not produce a spec"]))
    try:
        notes, resources = source_notes(brief, usable)
        result = (compiler or SpecCompiler()).compile(usable, brief, notes=notes, resources=resources)
        if result.spec is None:
            store.remember_failure(key)
            logger.warning("generation spec for %s not available (%s): %s", brief.get("scenario_type"), result.status, result.problems[:3])
            return SpecResolution(None, key, result.status, _describe(None, key, result.status, result.problems, rounds=result.rounds))
        stored = store.save(key, result.spec, {"brief": brief, "rounds": result.rounds})
        return SpecResolution(result.spec, key, "compiled",
                              _describe(result.spec, key, "compiled", rounds=result.rounds, persisted=stored,
                                        score=result.report.get("overall"), review=result.report.get("review")))
    finally:
        _release(key)


def _warm(variables: list[dict[str, Any]], brief: dict[str, Any], hint_key: str | None) -> None:
    try:
        ensure_spec(variables, brief, hint_key=hint_key)
    except Exception:                                     # a background design must never take anything down
        logger.exception("background generation-spec design failed")


def warm_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None) -> str | None:
    """Start designing the spec for these variables in the background, unless it exists or is already being designed.

    Returns the spec's key at once (the key is known before the design is). Never blocks and never raises.
    """
    try:
        key = design_key(variables, brief)
        if key is None or in_flight(key) or store.recently_failed(key) or store.load(key) is not None:
            return key
        threading.Thread(target=_warm, args=(list(variables), dict(brief), hint_key), name=f"spec-{key[:8]}", daemon=True).start()
        return key
    except Exception:
        logger.exception("could not start the background generation-spec design")
        return None


def pin_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None) -> dict[str, Any]:
    """Name the spec a confirmed scenario uses and make sure its design is under way; never waits for a model.

    Returns a report whose ``spec_key`` is what the scenario pins and whose ``status`` is ``ready`` (a verified spec exists
    for exactly these variables) or ``designing`` (it is being designed and generation will wait for it).
    """
    key = design_key(variables, brief)
    if key is None:
        return {"status": "unusable", "spec_key": None, "problems": ["no variables"]}
    spec = store.load(key)
    if spec is not None and _covers(spec, _names(variables)):
        return _describe(spec, key, "ready")
    hint = store.load(hint_key) if hint_key and hint_key != key else None
    if hint is not None and _names(variables) <= set(hint.delivered.values()):
        resolution = ensure_spec(variables, brief, hint_key=hint_key)        # narrowing a stored spec is a simulation, not a model call
        if resolution.spec is not None:
            return {**resolution.report, "status": "ready"}
    warm_spec(variables, brief, hint_key=hint_key)
    return {"status": "designing", "spec_key": key}


def resolve_spec(context: dict[str, Any], variables: list[dict[str, Any]], *, compiler: SpecCompiler | None = None) -> SpecResolution:
    """The spec a confirmed scenario generates from: the one pinned at confirmation, else the stored/compiled one."""
    key = str(context.get("generation_spec_key") or "").strip() or None
    names = _names(variables)
    pinned = store.load(key)
    if pinned is not None and names == set(pinned.delivered.values()):
        return SpecResolution(pinned, key, "pinned", _describe(pinned, key, "pinned"))
    return ensure_spec(variables, make_brief(context), hint_key=key, compiler=compiler)


# ------------------------------------------------------------------------------------------------------------------
def generate(spec: GenerationSpec, variables: list[dict[str, Any]], *, count: int, per_entity: int,
             seed: int | None = None, as_of: Any = None) -> SpecGeneration:
    delivered = spec.delivered
    missing = _names(variables) - set(delivered.values())
    if missing:
        raise ValueError("The generation spec does not cover these confirmed variables: " + ", ".join(sorted(missing)[:20]))
    engine = get_engine(spec.engine)
    aggregational = str(spec.scenario.get("type_of_data") or "").lower() == "aggregational"
    per_entity = 1 if aggregational else per_entity
    ctx, rows = _simulate_with_coverage(engine, spec, set(delivered), seed=seed, as_of=as_of, count=count, per_entity=per_entity)
    records = project(rows, spec, delivered)
    parsed, parse_errors = parse_columns(records, spec, delivered)
    report = score_rows(spec, parsed, engine.reference_view(spec, ctx), columns=set(delivered), parse_errors=parse_errors,
                        contract=build_contract(variables, delivered))
    report.update(seed=seed, as_of=ctx.as_of.isoformat(), engine=spec.engine, generation_engine=spec.engine,
                  spec_assumptions=spec.assumptions, **_summary_keys(report, count, per_entity, len(records), len(spec.warnings)))
    clock = spec.columns[spec.clock].column if spec.clock else None
    return SpecGeneration(records=records, fields=list(records[0]) if records else [], entity_columns=spec.entity_columns,
                          clock_column=clock, validation_report=report)


# Rows a coverage search may simulate in total: small requests (where chance can leave a conditional fact
# unrepresented) get many attempts, large ones (where it cannot) get one.
_COVERAGE_ROW_BUDGET = 20_000
_COVERAGE_MAX_ATTEMPTS = 40


def _attempt_seed(seed: int | None, attempt: int) -> int | None:
    if attempt == 0 or seed is None:
        return seed
    return (int(seed) * 1_000_003 + attempt) % 2_147_483_647


def _simulate_with_coverage(engine, spec: GenerationSpec, columns: set[str], *, seed: int | None, as_of: Any, count: int,
                            per_entity: int):
    """Simulate, preferring a dataset in which every delivered column carries at least one value.

    A conditional fact (a recurring-top-up period, a suspension reason) is empty on the rows it does not apply
    to, but in a small dataset chance alone can leave it empty everywhere, which reads as a broken column. The
    attempts are a deterministic sequence derived from the seed (the first is the seed itself), so the same
    request always yields the same dataset; the first attempt that represents every column wins, otherwise the
    one with the fewest empty columns.
    """
    base_as_of = RunContext(seed=0, as_of=as_of, tz_name=spec.timezone).as_of
    attempts = max(1, min(_COVERAGE_MAX_ATTEMPTS, _COVERAGE_ROW_BUDGET // max(1, count * per_entity)))
    best = None
    for attempt in range(attempts):
        ctx = RunContext(seed=_attempt_seed(seed, attempt), as_of=base_as_of, tz_name=spec.timezone)
        rows = engine.simulate(spec, ctx, entities=count, per_entity=per_entity, hints={})
        empty = sum(1 for c in columns if all(row.get(c) is None for row in rows))
        if best is None or empty < best[0]:
            best = (empty, ctx, rows)
        if empty == 0:
            break
    return best[1], best[2]


def _summary_keys(report: dict[str, Any], count: int, per_entity: int, produced: int, warnings: int) -> dict[str, Any]:
    """The keys clients already read, filled with values *measured* on the delivered rows (never constants)."""
    from config.runtime import MIN_LOGIC_QUALITY_PERCENT

    comps = report.get("components", {})
    expected = count * per_entity
    failed = int(report.get("rows_failed", 0))

    def pct(part: float, whole: float) -> float:
        return round(100.0 * part / whole, 3) if whole else 100.0

    conformance = pct(max(0, produced - int(report.get("structure", {}).get("contract_rows_failed", 0))), produced)
    clean = round(100.0 * comps.get("clean_rows", 0.0), 3)
    return {
        "requested_records": expected,
        "total_input": produced,
        "total_valid": produced - failed,
        "total_dropped": 0,
        "record_errors": failed,
        "valid_record_rate": pct(produced - failed, expected),
        "clean_record_rate": clean,
        "contract_conformance_rate": conformance,
        "contract_conformance_metric": "scored_on_delivered_rows_against_definitions_and_spec_invariants",
        "repaired_record_rate": 0.0,
        "target_valid_record_rate": 100.0,
        "target_contract_conformance_percent": MIN_LOGIC_QUALITY_PERCENT,
        "quality_target_met": report.get("verdict") == "pass" and clean >= MIN_LOGIC_QUALITY_PERCENT and conformance >= MIN_LOGIC_QUALITY_PERCENT
                              and produced == expected,
        "contract_pass_rate": round(100.0 * comps.get("consistency", 0.0), 3),
        "recovered": 0,
        "algo_fixes": 0,
        "llm_fixes": 0,
        "llm_issues": warnings,
        "deterministic_checks": ["spec_invariants", "distribution_targets", "definition_contract", "structure"],
    }
