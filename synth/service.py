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
import time
from dataclasses import dataclass, field
from typing import Any

from synth import store
from synth.clock import RunContext
from synth.compiler import CompileResult, SpecCompiler, finalize, finding_text, restrict, spec_key, verify
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
    brief["events_per_entity"] = source.get("records_per_user")       # what the history is designed to hold (not part of the spec's identity)
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
                      invariants=len(spec.invariants), targets=len(spec.targets), columns=len(spec.delivered),
                      revision=spec.revision, reviewed=spec.reviewed, refining=refining(key))
        if spec.warnings:
            report["review_warnings"] = spec.warnings
    if problems:
        report["problems"] = problems[:12]
    return report


def _why(key: str | None, fallback: str) -> list[str]:
    """The reason a design failed, for the report: what the checks or the model said, else ``fallback``."""
    reason = store.failure_reason(key)
    return [fallback, reason] if reason else [fallback]


class SpecNotReady(RuntimeError):
    """The scenario's generation spec is still being designed; generating from it would have to wait longer than allowed."""

    def __init__(self, key: str | None, retry_after: int):
        super().__init__("The generation spec for this scenario is still being designed. Retry in about "
                         f"{retry_after} seconds; the design continues in the background.")
        self.key, self.retry_after = key, retry_after


_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}
_REFINING: set[str] = set()
_REFINE_SCHEDULED: set[str] = set()                         # refinements that were started and have not begun yet
_REFINE_REVISION: dict[str, int] = {}                        # the newest revision a running refinement has stored
_REFINE_CHANGED = threading.Condition(_FLIGHT_LOCK)          # signalled when a refinement stores a revision or ends
_REFINE_BACKOFF: dict[str, float] = {}
REFINE_BACKOFF_SECONDS = 300.0


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


def _wait(key: str, timeout: float | None = None) -> bool:
    """Block until the design of ``key`` that is in flight ends; False when ``timeout`` passed first."""
    with _FLIGHT_LOCK:
        event = _IN_FLIGHT.get(key)
    if event is None:
        return True
    return event.wait(timeout=timeout if timeout is not None else SpecCompiler._budget() + 30.0)


def in_flight(key: str | None) -> bool:
    with _FLIGHT_LOCK:
        return key in _IN_FLIGHT


def refining(key: str | None) -> bool:
    with _FLIGHT_LOCK:
        return key in _REFINING or key in _REFINE_SCHEDULED


def _after_refinement(key: str, spec: GenerationSpec, names: set[str]) -> GenerationSpec:
    """The spec to generate from: ``spec``, or the refinement's first improvement of it when that arrives within the allowed wait.

    A spec nobody has reviewed is the raw first design; waiting a little for its first reviewed revision costs less than data
    that carries the first design's flaws. A refinement that finishes without improving it, or one that is not running, ends the wait.
    """
    from config.runtime import SPEC_GENERATE_REFINE_WAIT_SECONDS

    if spec.reviewed or SPEC_GENERATE_REFINE_WAIT_SECONDS <= 0:
        return spec
    deadline = time.monotonic() + SPEC_GENERATE_REFINE_WAIT_SECONDS
    with _REFINE_CHANGED:
        while (key in _REFINING or key in _REFINE_SCHEDULED) and _REFINE_REVISION.get(key, spec.revision) <= spec.revision:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _REFINE_CHANGED.wait(remaining)
    latest = store.load(key)
    return latest if latest is not None and names == set(latest.delivered.values()) else spec


def design_key(variables: list[dict[str, Any]], brief: dict[str, Any]) -> str | None:
    """The key of the spec for exactly these variables and this scenario (None when there is nothing to design for)."""
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    return spec_key(brief, usable) if usable else None


def _refine_mode() -> str:
    from config.runtime import SPEC_REFINE_MODE

    return SPEC_REFINE_MODE


def _stored(variables: list[dict[str, Any]], brief: dict[str, Any], key: str, hint_key: str | None = None) -> SpecResolution | None:
    """A spec that exists already for exactly these variables: stored under ``key``, or ``hint_key``'s spec narrowed to them.

    Neither needs a model: finding a stored spec is a lookup and narrowing one is a simulation.
    """
    names = _names(variables)
    spec = store.load(key)
    if spec is not None and _covers(spec, names):
        return SpecResolution(spec, key, "cached", _describe(spec, key, "cached"))
    hint = store.load(hint_key) if hint_key and hint_key != key else None
    if hint is not None and names <= set(hint.delivered.values()):
        narrowed = restrict(hint, names)
        problems, _ = verify(narrowed, variables, aggregational=str(brief.get("type_of_data") or "").lower() == "aggregational")
        if not problems and names == set(narrowed.delivered.values()):
            stored = store.save(key, narrowed, {"brief": brief, "derived_from": hint_key})
            return SpecResolution(narrowed, key, "restricted", _describe(narrowed, key, "restricted", persisted=stored))
    return None


def _refine_job(variables: list[dict[str, Any]], brief: dict[str, Any], key: str, spec: GenerationSpec,
                compiler: SpecCompiler | None) -> None:
    """Improve the stored spec of ``key`` with the reviewer's findings; every better version is stored as it is found. Never raises."""
    with _FLIGHT_LOCK:
        _REFINE_SCHEDULED.discard(key)
        if key in _REFINING:
            return
        _REFINING.add(key)
        _REFINE_REVISION.setdefault(key, spec.revision)
    published: dict[str, Any] = {"spec": spec}
    try:
        comp = compiler or SpecCompiler()
        notes, resources = source_notes(brief, variables)
        current = CompileResult(spec, "compiled", [], spec.revision, spec.design, {}, [])

        def publish(better: CompileResult) -> None:
            better = finalize(better, published["spec"].revision + 1)
            better.spec = better.spec.model_copy(update={"reviewed": False, "design": better.overlay or {}})
            store.save(key, better.spec, {"brief": brief, "rounds": better.rounds})
            published["spec"] = better.spec
            with _REFINE_CHANGED:
                _REFINE_REVISION[key] = better.spec.revision
                _REFINE_CHANGED.notify_all()
            logger.info("generation spec %s improved to revision %d (%d finding(s) left)", key[:8], better.spec.revision, len(better.findings))

        better = comp.refine(variables, brief, current, notes=notes, resources=resources, on_improve=publish)
        if better is not None and better.spec is not None and published["spec"] is spec:
            publish(better)                                  # a refinement that did not publish as it went
        review = str((better.report if better is not None else current.report).get("review") or "")
        findings = better.findings if better is not None else current.findings
        if review in {"skipped", "deferred"}:
            with _FLIGHT_LOCK:
                _REFINE_BACKOFF[key] = time.monotonic()
        latest = published["spec"]
        store.annotate(key, latest, [finding_text(f) for f in findings] or latest.warnings, reviewed=review not in {"skipped", "deferred"})
    except Exception:
        logger.exception("background refinement of generation spec %s failed", key[:8])
    finally:
        with _REFINE_CHANGED:
            _REFINING.discard(key)
            _REFINE_REVISION.pop(key, None)
            _REFINE_CHANGED.notify_all()


def _schedule_refinement(variables: list[dict[str, Any]], brief: dict[str, Any], key: str, spec: GenerationSpec,
                         compiler: SpecCompiler | None, *, inline: bool = False) -> None:
    """Start improving ``spec`` unless that is off, already done or already running."""
    mode = _refine_mode()
    if mode == "off" or spec.reviewed or not spec.design or refining(key):
        return
    with _FLIGHT_LOCK:
        if time.monotonic() - _REFINE_BACKOFF.get(key, -REFINE_BACKOFF_SECONDS) < REFINE_BACKOFF_SECONDS:
            return                                           # the reviewer could not be reached a moment ago
        if not (inline or mode == "inline"):
            _REFINE_SCHEDULED.add(key)
    if inline or mode == "inline":
        _refine_job(variables, brief, key, spec, compiler)
        return
    threading.Thread(target=_refine_job, args=(list(variables), dict(brief), key, spec, compiler), name=f"refine-{key[:8]}", daemon=True).start()


def _design_claimed(variables: list[dict[str, Any]], brief: dict[str, Any], key: str, hint_key: str | None,
                    compiler: SpecCompiler | None) -> SpecResolution:
    """Design the spec of ``key``; the caller holds its claim. Returns once the first verified spec exists (a refinement may follow)."""
    found = _stored(variables, brief, key, hint_key)
    if found is not None:
        return found
    notes, resources = source_notes(brief, variables)
    result = (compiler or SpecCompiler()).draft(variables, brief, notes=notes, resources=resources)
    if result.spec is None:
        store.remember_failure(key, "; ".join(result.problems[:3])[:600] or result.status, transient=result.status == "unavailable")
        logger.warning("generation spec for %s not available (%s): %s", brief.get("scenario_type"), result.status, result.problems[:3])
        return SpecResolution(None, key, result.status, _describe(None, key, result.status, result.problems, rounds=result.rounds))
    result = finalize(result, 0)
    spec = result.spec.model_copy(update={"design": result.overlay or {}, "reviewed": False})
    stored = store.save(key, spec, {"brief": brief, "rounds": result.rounds})
    resolution = SpecResolution(spec, key, "compiled", _describe(spec, key, "compiled", rounds=result.rounds, persisted=stored,
                                                                 score=result.report.get("overall")))
    _schedule_refinement(variables, brief, key, spec, compiler)
    if _refine_mode() == "inline":
        refined = store.load(key)
        if refined is not None:
            resolution = SpecResolution(refined, key, "compiled", _describe(refined, key, "compiled", rounds=result.rounds, persisted=stored))
    return resolution


def ensure_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None,
                compiler: SpecCompiler | None = None) -> SpecResolution:
    """The verified spec for exactly these variables and this scenario, designing it now if it does not exist (blocking).

    In order: a stored spec for the same inputs; the spec of ``hint_key`` narrowed to these variables (the proposal's
    spec after the user deleted columns); a fresh design by the language model. A result that is not a verified model
    spec (``rejected`` / ``unavailable``) is never stored, so a later request tries again. Request handlers use
    ``pin_spec`` / ``resolve_spec`` instead, which never wait on a model for longer than allowed.
    """
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    if not usable:
        return SpecResolution(None, None, "unusable", _describe(None, None, "unusable", ["no variables"]))
    key = spec_key(brief, usable)
    found = _stored(usable, brief, key, hint_key)
    if found is not None:
        return found
    if store.recently_failed(key):
        return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", _why(key, "a recent attempt to design this spec failed")))
    if not _claim(key):
        _wait(key)                              # the same design is already running: use its result instead of repeating it
        spec = store.load(key)
        if spec is not None and _covers(spec, _names(usable)):
            return SpecResolution(spec, key, "cached", _describe(spec, key, "cached"))
        return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", _why(key, "the design that was in progress did not produce a spec")))
    try:
        return _design_claimed(usable, brief, key, hint_key, compiler)
    finally:
        _release(key)


def _background(variables: list[dict[str, Any]], brief: dict[str, Any], key: str, hint_key: str | None) -> None:
    try:
        _design_claimed(variables, brief, key, hint_key, None)
    except Exception as exc:                              # a background design must never take anything down
        logger.exception("background generation-spec design failed")
        store.remember_failure(key, f"{type(exc).__name__}: {exc}"[:600], transient=True)
    finally:
        _release(key)


def warm_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None) -> str | None:
    """Start designing the spec for these variables in the background, unless it exists or is already being designed.

    Returns the spec's key at once (the key is known before the design is). Never blocks and never raises.
    """
    try:
        key = design_key(variables, brief)
        if key is None:
            return None
        spec = store.load(key)
        if spec is not None:
            _schedule_refinement(variables, brief, key, spec, None)       # resumes a refinement a restart interrupted
            return key
        if store.recently_failed(key) or not _claim(key):
            return key
        try:
            threading.Thread(target=_background, args=(list(variables), dict(brief), key, hint_key), name=f"spec-{key[:8]}", daemon=True).start()
        except Exception:
            _release(key)
            raise
        return key
    except Exception:
        logger.exception("could not start the background generation-spec design")
        return None


def pin_spec(variables: list[dict[str, Any]], brief: dict[str, Any], *, hint_key: str | None = None) -> dict[str, Any]:
    """Name the spec a confirmed scenario uses and make sure its design is under way; never waits for a model.

    Returns a report whose ``spec_key`` is what the scenario pins and whose ``status`` is ``ready`` (a verified spec exists
    for exactly these variables) or ``designing`` (it is being designed and generation will wait for it, up to a limit).
    """
    key = design_key(variables, brief)
    if key is None:
        return {"status": "unusable", "spec_key": None, "problems": ["no variables"]}
    spec = store.load(key)
    if spec is not None and _covers(spec, _names(variables)):
        _schedule_refinement(variables, brief, key, spec, None)
        return _describe(spec, key, "ready")
    hint = store.load(hint_key) if hint_key and hint_key != key else None
    if hint is not None and _names(variables) <= set(hint.delivered.values()):
        found = _stored(variables, brief, key, hint_key)      # narrowing a stored spec is a simulation, not a model call
        if found is not None:
            return {**found.report, "status": "ready"}
    warm_spec(variables, brief, hint_key=hint_key)
    return {"status": "designing", "spec_key": key}


def resolve_spec(context: dict[str, Any], variables: list[dict[str, Any]], *, wait_seconds: float | None = None,
                 compiler: SpecCompiler | None = None) -> SpecResolution:
    """The spec a confirmed scenario generates from: the one pinned at confirmation, else the stored or designed one.

    A spec that is still being designed is waited for, but never longer than ``wait_seconds``
    (``SPEC_GENERATE_WAIT_SECONDS``): after that ``SpecNotReady`` says so, while the design carries on in the background.
    Refinement of a spec that already exists never delays generation.
    """
    from config.runtime import SPEC_GENERATE_WAIT_SECONDS

    wait = float(SPEC_GENERATE_WAIT_SECONDS if wait_seconds is None else wait_seconds)
    pinned_key = str(context.get("generation_spec_key") or "").strip() or None
    brief = make_brief(context)
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    names = _names(usable)
    pinned = store.load(pinned_key)
    if pinned is not None and names == set(pinned.delivered.values()):
        _schedule_refinement(usable, brief, pinned_key, pinned, compiler)
        pinned = _after_refinement(pinned_key, pinned, names)
        return SpecResolution(pinned, pinned_key, "pinned", _describe(pinned, pinned_key, "pinned"))
    key = design_key(usable, brief)
    if key is None:
        return SpecResolution(None, None, "unusable", _describe(None, None, "unusable", ["no variables"]))
    found = _stored(usable, brief, key, pinned_key)
    if found is not None:
        if found.spec is not None:
            _schedule_refinement(usable, brief, key, found.spec, compiler)
            latest = _after_refinement(key, found.spec, names)
            if latest is not found.spec:
                return SpecResolution(latest, key, found.status, _describe(latest, key, found.status))
        return found
    if store.recently_failed(key):
        return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", _why(key, "a recent attempt to design this spec failed")))
    if compiler is not None:                                   # a caller that brings its own compiler designs here, now
        return ensure_spec(usable, brief, hint_key=pinned_key, compiler=compiler)
    waiting_on = pinned_key if pinned_key and in_flight(pinned_key) else key
    if not in_flight(waiting_on):
        warm_spec(usable, brief, hint_key=pinned_key)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _wait(waiting_on, timeout=max(0.0, deadline - time.monotonic())):
            break
    spec = store.load(waiting_on)
    if spec is not None and names == set(spec.delivered.values()):
        status = "pinned" if waiting_on == pinned_key else "cached"
        return SpecResolution(spec, waiting_on, status, _describe(spec, waiting_on, status))
    if in_flight(waiting_on):
        raise SpecNotReady(waiting_on, retry_after=max(10, int(wait)))
    return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", _why(key, "the design of this spec did not produce a usable spec")))


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
                  spec_revision=spec.revision, spec_assumptions=spec.assumptions, **_summary_keys(report, count, per_entity, len(records), len(spec.warnings)))
    clock = spec.columns[spec.clock].column if spec.clock else None
    return SpecGeneration(records=records, fields=list(records[0]) if records else [], entity_columns=spec.entity_columns,
                          clock_column=clock, validation_report=report)


# Rows a coverage search may simulate in total: small requests (where chance can leave a conditional fact
# unrepresented) get many attempts, large ones (where it cannot) get one.
_COVERAGE_ROW_BUDGET = 6_000
_COVERAGE_MAX_ATTEMPTS = 24


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
