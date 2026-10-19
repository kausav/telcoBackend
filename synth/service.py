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
    try:
        from core import country_limits

        found = country_limits.for_country(brief.get("country"))
        if found:
            brief["country_limits"] = found                                    # part of the spec's identity: new limits, new design
    except Exception as exc:                                           # the limits are an aid; a design without them is still valid
        logger.warning("country limits not added to the brief: %s: %s", type(exc).__name__, exc)
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
        found = spec.design.get("expectations") if isinstance(spec.design, dict) else None
        if isinstance(found, dict):
            report["expectations"] = {"rules": len(found.get("rules") or []), "entity_facts": len(found.get("entity_facts") or []),
                                      "constants": len(found.get("constants") or []), "state_shares": len(found.get("state_shares") or [])}
        timing = spec.design.get("timing") if isinstance(spec.design, dict) else None
        if isinstance(timing, dict):
            report["design_seconds"] = timing
        refined = spec.design.get("refinement") if isinstance(spec.design, dict) else None
        if isinstance(refined, dict):
            report["refinement"] = refined
        healed = spec.design.get("healed") if isinstance(spec.design, dict) else None
        if healed:
            report["confined_defects"] = [str(h.get("problem")) for h in healed if isinstance(h, dict)][:12]
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
    """The scenario's generation spec is still being designed or refined; generating from it now would give data that the next request replaces."""

    def __init__(self, key: str | None, retry_after: int, phase: str = "designing"):
        verb = "refined" if phase == "refining" else "designed"
        super().__init__(f"The generation spec for this scenario is still being {verb}. Retry in about "
                         f"{retry_after} seconds; the work continues in the background.")
        self.key, self.retry_after, self.phase = key, retry_after, phase


RETRY_AFTER_SECONDS = 30
_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, threading.Event] = {}
_REFINING: set[str] = set()
_REFINE_SCHEDULED: set[str] = set()                         # refinements that were started and have not begun yet
_REFINE_REVISION: dict[str, int] = {}                        # the newest revision a running refinement has stored
_REFINE_CHANGED = threading.Condition(_FLIGHT_LOCK)          # signalled when a refinement stores a revision or ends
_REFINE_BACKOFF: dict[str, float] = {}
_REFINE_STARTED: dict[str, float] = {}                       # key -> monotonic time its refinement was scheduled in this process
REFINE_BACKOFF_SECONDS = 300.0
_REFINE_PREFIX = "refine:"                                 # claim key of a refinement (see synth.store.claim_remote)


def _claim_seconds() -> float:
    """How long a design claim stays valid if its process dies: the longest a design may run, plus slack."""
    from config.runtime import SPEC_COMPILE_BUDGET_SECONDS

    return float(SPEC_COMPILE_BUDGET_SECONDS) + 120.0


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


def _refine_age(key: str) -> float:
    """Seconds the refinement of ``key`` has been under way (in this process or, by its claim, in another worker)."""
    with _FLIGHT_LOCK:
        started = _REFINE_STARTED.get(key)
    local = time.monotonic() - started if started is not None else None
    remote = store.claim_age(_REFINE_PREFIX + key)
    ages = [a for a in (local, remote) if a is not None]
    return max(ages) if ages else 0.0


def _open_errors(spec: GenerationSpec) -> bool:
    """True when the findings recorded on the spec include an error (an impossible record, not a matter of taste)."""
    return any(str(w).startswith("error:") for w in getattr(spec, "warnings", None) or [])


def _after_refinement(key: str, spec: GenerationSpec, names: set[str], *, limit: float | None = None) -> GenerationSpec:
    """The spec to generate from: the settled one, once the refinement that is improving ``spec`` has ended.

    A spec that is still being refined is not final, and data generated from it would differ from the data of the next request
    (and would still carry the errors the review found). So this waits for the end of the refinement for at most ``limit`` seconds
    (what is left of the request's own allowance) and, if it is still running, raises ``SpecNotReady("refining")`` - the caller
    asks again - unless the refinement has already run ``SPEC_SETTLE_SECONDS``, when the best revision so far is served.
    """
    if spec.reviewed:
        return spec
    from config.runtime import SPEC_SERVE_EARLY, SPEC_SETTLE_SECONDS

    if SPEC_SERVE_EARLY and not _open_errors(spec):
        # nothing impossible is known about this spec: serve it now; the refinement carries on and the next request gets its result
        latest = store.load(key)
        return latest if latest is not None and names == set(latest.delivered.values()) and not _open_errors(latest) else spec

    deadline = time.monotonic() + (min(SpecCompiler._budget(), 30.0) if limit is None else limit)
    with _REFINE_CHANGED:
        while key in _REFINING or key in _REFINE_SCHEDULED:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _REFINE_CHANGED.wait(remaining)
    while time.monotonic() < deadline and key not in _REFINING and store.designing_elsewhere(_REFINE_PREFIX + key):
        time.sleep(min(3.0, max(0.0, deadline - time.monotonic())))      # another worker is refining it: wait for its last revision
    latest = store.load(key, fresh=True)
    latest = latest if latest is not None and names == set(latest.delivered.values()) else spec
    if not latest.reviewed and (refining(key) or store.designing_elsewhere(_REFINE_PREFIX + key)) and _refine_age(key) < SPEC_SETTLE_SECONDS:
        raise SpecNotReady(key, retry_after=RETRY_AFTER_SECONDS, phase="refining")
    return latest


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
        report = better.report if better is not None else current.report
        review = str(report.get("review") or "")
        findings = better.findings if better is not None else current.findings
        # a refinement that ended because the design window was spent is settled: nothing more will be tried for this spec
        settled = (report.get("refinement") or {}).get("stopped") == "time"
        if review in {"skipped", "deferred"} and not settled:
            with _FLIGHT_LOCK:
                _REFINE_BACKOFF[key] = time.monotonic()
        latest = published["spec"]
        store.annotate(key, latest, [finding_text(f) for f in findings] or latest.warnings,
                       reviewed=settled or review not in {"skipped", "deferred"}, refinement=report.get("refinement"))
    except Exception:
        logger.exception("background refinement of generation spec %s failed", key[:8])
    finally:
        with _REFINE_CHANGED:
            _REFINING.discard(key)
            _REFINE_REVISION.pop(key, None)
            _REFINE_STARTED.pop(key, None)
            _REFINE_CHANGED.notify_all()
        store.release_remote(_REFINE_PREFIX + key)


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
            _REFINE_STARTED.setdefault(key, time.monotonic())
    if not store.claim_remote(_REFINE_PREFIX + key, _claim_seconds()):        # another worker is refining this spec already
        with _FLIGHT_LOCK:
            _REFINE_SCHEDULED.discard(key)
            _REFINE_STARTED.pop(key, None)
        return
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
        store.release_remote(key)


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
        if not store.claim_remote(key, _claim_seconds()):          # another worker process is designing this very spec
            _release(key)
            return key
        try:
            threading.Thread(target=_background, args=(list(variables), dict(brief), key, hint_key), name=f"spec-{key[:8]}", daemon=True).start()
        except Exception:
            _release(key)
            store.release_remote(key)
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
    A spec that is still being refined is waited for as well (see ``_after_refinement``): every request for a scenario generates from the settled spec.
    """
    from config.runtime import SPEC_GENERATE_WAIT_SECONDS

    wait = float(SPEC_GENERATE_WAIT_SECONDS if wait_seconds is None else wait_seconds)
    began = time.monotonic()

    def left() -> float:
        return max(0.0, wait - (time.monotonic() - began))

    pinned_key = str(context.get("generation_spec_key") or "").strip() or None
    brief = make_brief(context)
    usable = [v for v in variables if isinstance(v, dict) and v.get("name")]
    names = _names(usable)
    pinned = store.load(pinned_key)
    if pinned is not None and names == set(pinned.delivered.values()):
        _schedule_refinement(usable, brief, pinned_key, pinned, compiler)
        pinned = _after_refinement(pinned_key, pinned, names, limit=left())
        return SpecResolution(pinned, pinned_key, "pinned", _describe(pinned, pinned_key, "pinned"))
    key = design_key(usable, brief)
    if key is None:
        return SpecResolution(None, None, "unusable", _describe(None, None, "unusable", ["no variables"]))
    found = _stored(usable, brief, key, pinned_key)
    if found is not None:
        if found.spec is not None:
            _schedule_refinement(usable, brief, key, found.spec, compiler)
            latest = _after_refinement(key, found.spec, names, limit=left())
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
        if in_flight(waiting_on):                               # designed by this process: wait for its end
            if _wait(waiting_on, timeout=max(0.0, deadline - time.monotonic())):
                break
        elif store.designing_elsewhere(waiting_on):             # designed by another worker: wait until its spec is stored
            if store.load(waiting_on, fresh=True) is not None:
                break
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        else:
            break
    spec = store.load(waiting_on, fresh=True)
    if spec is not None and names == set(spec.delivered.values()):
        status = "pinned" if waiting_on == pinned_key else "cached"
        spec = _after_refinement(waiting_on, spec, names, limit=left())
        return SpecResolution(spec, waiting_on, status, _describe(spec, waiting_on, status))
    if in_flight(waiting_on) or store.designing_elsewhere(waiting_on):
        raise SpecNotReady(waiting_on, retry_after=RETRY_AFTER_SECONDS)       # the request has waited its share already: ask to come back soon, not after another full wait
    return SpecResolution(None, key, "unavailable", _describe(None, key, "unavailable", _why(key, "the design of this spec did not produce a usable spec")))


# ------------------------------------------------------------------------------------------------------------------
def column_sequence(spec: GenerationSpec, variables: list[dict[str, Any]], delivered: dict[str, str]) -> list[str]:
    """The delivered column names in the order of the data's story: a ``sequence`` on a variable first, else the story order
    (``synth.sequence``) of the party, what it holds, its facts, its dates (a start before its end) and then each event."""
    from synth.sequence import story_order

    by_name = {str(v.get("name")): v for v in variables if isinstance(v, dict) and v.get("name")}
    names = list(delivered.values())
    definitions = [by_name.get(n) or {"name": n, "dtype": spec.columns[cid].dtype} for cid, n in delivered.items()]
    kinds = {n: ("entity" if spec.columns[cid].kind == "entity" else "event") for cid, n in delivered.items()}
    entity = spec.columns[spec.entity_column].column if spec.entity_column else None
    clock = spec.columns[spec.clock].column if spec.clock else None
    try:
        ordered = story_order(definitions, entity_key=entity, kinds=kinds, clock=clock)
    except Exception:                                   # the order is presentation: never let it stop a generation
        logger.exception("column order could not be computed; using the approved order")
        ordered = names
    return ordered + [n for n in names if n not in set(ordered)]


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
                        contract=build_contract(variables, delivered, spec.scenario.get("country")))
    order = column_sequence(spec, variables, delivered)
    records = [{name: r[name] for name in order if name in r} for r in records]
    summary = _summary_keys(report, count, per_entity, len(records), spec.warnings, _confined(spec))
    # columns that carry no value on any row stay visible: the design left them empty because the scenario excludes the fact
    summary["empty_columns"] = [name for name in delivered.values() if records and all(r.get(name) is None for r in records)]
    # columns that are empty on most rows are kind-specific facts (a deposit amount exists only on deposit rows): reported, not hidden
    if records:
        shares = {name: sum(1 for r in records if r.get(name) is None) / len(records) for name in delivered.values()}
        summary["sparse_columns"] = {name: round(share, 3) for name, share in shares.items() if 0.5 <= share < 1.0}
    open_errors = int(summary["open_findings"]["errors"])
    if report.get("verdict") == "pass" and open_errors:
        # the rows follow the spec's own rules, but the design review left errors open: that is not a clean pass
        report["spec_verdict"], report["verdict"] = "pass", "review"
    if open_errors and isinstance(report.get("overall"), (int, float)):
        # a score of 1.0 beside a known open error reads as "nothing is wrong": each open error takes a tenth off, at most half
        report["overall_before_open_findings"] = report["overall"]
        report["overall"] = round(report["overall"] * (1.0 - min(0.5, 0.1 * open_errors)), 4)
    report.update(seed=seed, as_of=ctx.as_of.isoformat(), engine=spec.engine, generation_engine=spec.engine,
                  spec_revision=spec.revision, spec_assumptions=spec.assumptions, **summary)
    clock = spec.columns[spec.clock].column if spec.clock else None
    entity_names = set(spec.entity_columns)
    return SpecGeneration(records=records, fields=list(records[0]) if records else list(order),
                          entity_columns=[n for n in order if n in entity_names],
                          clock_column=clock, validation_report=report)


# Rows a coverage search may simulate in total: small requests (where chance can leave a conditional fact
# unrepresented) get many attempts, large ones (where it cannot) get one.
_COVERAGE_ROW_BUDGET = 6_000
_COVERAGE_MAX_ATTEMPTS = 24


def _attempt_seed(seed: int | None, attempt: int) -> int | None:
    if attempt == 0 or seed is None:
        return seed
    return (int(seed) * 1_000_003 + attempt) % 2_147_483_647


def _degenerate(spec: GenerationSpec, columns: set[str], rows: list[dict[str, Any]]) -> int:
    """How badly a small dataset misrepresents its columns: a column empty on every row weighs two, a number or yes/no that is
    one single value on every row weighs one (a conditional fact that chance happened never to trigger reads as a broken column)."""
    score = 0
    for c in columns:
        present = [row[c] for row in rows if row.get(c) is not None]
        if not present:
            score += 2
        elif len(rows) >= 20 and spec.columns[c].dtype in ("integer", "float", "boolean") and len({repr(v) for v in present}) == 1:
            score += 1
    return score


def _simulate_with_coverage(engine, spec: GenerationSpec, columns: set[str], *, seed: int | None, as_of: Any, count: int,
                            per_entity: int):
    """Simulate, preferring a dataset in which every delivered column carries a value and no number or flag is stuck on one.

    A conditional fact (a recurring-top-up period, a suspension reason, a retry count) is empty or stuck on the rows it does
    not apply to, but in a small dataset chance alone can leave it so everywhere, which reads as a broken column. The
    attempts are a deterministic sequence derived from the seed (the first is the seed itself), so the same
    request always yields the same dataset; the first attempt that has no such column wins, otherwise the
    one with the least of them (a column that really is constant makes every attempt equal, and the first stays).
    """
    base_as_of = RunContext(seed=0, as_of=as_of, tz_name=spec.timezone).as_of
    attempts = max(1, min(_COVERAGE_MAX_ATTEMPTS, _COVERAGE_ROW_BUDGET // max(1, count * per_entity)))
    best = None
    for attempt in range(attempts):
        ctx = RunContext(seed=_attempt_seed(seed, attempt), as_of=base_as_of, tz_name=spec.timezone)
        rows = engine.simulate(spec, ctx, entities=count, per_entity=per_entity, hints={})
        score = _degenerate(spec, columns, rows)
        if best is None or score < best[0]:
            best = (score, ctx, rows)
        if score == 0:
            break
    return best[1], best[2]


_UNMET_WHEN_CONFINED = frozenset({"healed_empty_column", "healed_column", "healed_invariant", "healed_target"})


def _confined(spec: GenerationSpec) -> list[str]:
    """What the design repair removed from the spec to make it usable (a column's behaviour, a rule, a target), still unrepaired.

    The refinement drops this record when it models the removed behaviour again, so what is listed here is still missing.
    """
    healed = spec.design.get("healed") if isinstance(spec.design, dict) else None
    return [str(h.get("problem")) for h in healed or [] if isinstance(h, dict) and h.get("id") in _UNMET_WHEN_CONFINED and h.get("problem")]


def _summary_keys(report: dict[str, Any], count: int, per_entity: int, produced: int, warnings: list[str],
                  confined: list[str] | None = None) -> dict[str, Any]:
    """The keys clients already read, filled with values *measured* on the delivered rows (never constants)."""
    from config.runtime import MIN_LOGIC_QUALITY_PERCENT

    comps = report.get("components", {})
    expected = count * per_entity
    failed = int(report.get("rows_failed", 0))

    def pct(part: float, whole: float) -> float:
        return round(100.0 * part / whole, 3) if whole else 100.0

    conformance = pct(max(0, produced - int(report.get("structure", {}).get("contract_rows_failed", 0))), produced)
    clean = round(100.0 * comps.get("clean_rows", 0.0), 3)
    reported_errors = sum(1 for w in warnings if str(w).startswith("error:"))
    # What the repair removed to make the spec usable is a rule or behaviour the data does not follow: it stays an open error until
    # the refinement models it again, whatever the score of the remaining rules says.
    removed = confined or []
    listed = sum(1 for w in warnings if not str(w).startswith("error:") and any(text in str(w) for text in removed))
    open_errors = reported_errors + len(removed)
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
        # the checks above only prove the rows follow the spec's own rules: a known error the review left open is a failed target
        "quality_target_met": report.get("verdict") == "pass" and clean >= MIN_LOGIC_QUALITY_PERCENT and conformance >= MIN_LOGIC_QUALITY_PERCENT
                              and produced == expected and open_errors == 0,
        "open_findings": {"errors": open_errors, "warnings": max(0, len(warnings) - reported_errors - listed),
                          "items": [str(w) for w in warnings if str(w).startswith("error:")][:10] + removed[:10]},
        "contract_pass_rate": round(100.0 * comps.get("consistency", 0.0), 3),
        "recovered": 0,
        "algo_fixes": 0,
        "llm_fixes": 0,
        "llm_issues": len(warnings),
        "deterministic_checks": ["spec_invariants", "distribution_targets", "definition_contract", "structure"],
    }
