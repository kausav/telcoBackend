"""Orchestration: confirmed variables + behaviour pack -> dataset + honest validation report.

This is the single entry point the API layer calls. It knows nothing about FastAPI, MongoDB or the
legacy generator, which keeps it unit-testable and lets other industries plug in by adding a pack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synth.clock import RunContext
from synth.concepts import canonicalize
from synth.contract import build_contract
from synth.engines import get_engine
from synth.pack import BehaviorPack, find_pack, get_pack
from synth.projection import parse_columns, project
from synth.scorer import score_rows
from config.runtime import ENABLE_BEHAVIOR_PACKS


@dataclass
class PackGeneration:
    records: list[dict[str, Any]]          # flat rows, column name -> formatted value
    fields: list[str]
    entity_key: str | None
    validation_report: dict[str, Any]
    concept_columns: dict[str, str]


def resolve_pack(scenario_context: dict[str, Any]) -> BehaviorPack | None:
    """The pack (and version) pinned at proposal time, else None (=> legacy generator).

    A draft that names a pack that cannot be loaded is an error, never a silent downgrade: quietly
    switching engines - or pack versions - would change the data a caller already approved.
    """
    pack_id = str(scenario_context.get("behavior_pack_id") or "").strip()
    if not pack_id:
        return None
    if not ENABLE_BEHAVIOR_PACKS:
        raise ValueError(
            f"Confirmed scenario is pinned to domain-specific behaviour pack '{pack_id}'. "
            "Set ENABLE_BEHAVIOR_PACKS=true to run it, or propose and confirm the scenario again "
            "with behaviour packs disabled to use the industry-standard source schema."
        )
    version = scenario_context.get("behavior_pack_version")
    pack = get_pack(pack_id, int(version) if version not in (None, "") else None)
    if pack is None:
        raise ValueError(
            f"The confirmed scenario was built on behaviour pack '{pack_id}' (version {version}), which cannot be "
            "loaded from the 'behavior_packs' collection. Restore it (status active or retired) or propose "
            "the scenario again."
        )
    return pack


def propose_pack(industry_key: str, domain_key: str, scenario_id: str, use_case: str, country: str,
                 type_of_data: str) -> BehaviorPack | None:
    return find_pack(industry_key=industry_key, domain_key=domain_key, scenario_id=scenario_id,
                     use_case=use_case, country=country, type_of_data=type_of_data)


def _hints(pack: BehaviorPack, variables: list[dict[str, Any]], concept_columns: dict[str, str]) -> dict[str, Any]:
    """Identifier formats the DB already defines (e.g. prefix + digits) for concepts whose sampler opts in
    with ``format_hint`` are honoured by the engine."""
    by_name = {str(v.get("name")): v for v in variables}
    hints: dict[str, Any] = {}
    for emit in pack.emit:
        if not (emit.sample or {}).get("format_hint"):
            continue
        params = (by_name.get(concept_columns.get(emit.concept, "")) or {}).get("params") or {}
        hint = params.get("format_hint") if isinstance(params.get("format_hint"), dict) else params
        if isinstance(hint, dict) and hint.get("prefix") is not None and hint.get("digits"):
            hints[emit.concept] = {"prefix": str(hint["prefix"]), "digits": int(hint["digits"])}
    return hints


def resolve_columns(variables: list[dict[str, Any]], pack: BehaviorPack) -> dict[str, str]:
    """concept -> column for the confirmed variables (explicit ``concept`` tag first, else pack binds).

    Raises when a curated variable has no concept: generating without it would silently drop a column
    somebody asked for.
    """
    sources = {str(v.get("name")): str(v.get("source") or "") for v in variables}
    result = canonicalize(variables, pack, sources)
    if result.uncovered:
        raise ValueError(
            f"Behaviour pack '{pack.pack_id}' cannot generate these confirmed variables: "
            + ", ".join(u["name"] for u in result.uncovered)
            + ". Remove them from the scenario or extend the pack."
        )
    return result.concept_columns


def generate(
    pack: BehaviorPack,
    variables: list[dict[str, Any]],
    *,
    count: int,
    per_entity: int,
    seed: int | None = None,
    as_of: Any = None,
    mode: str = "mixed",
) -> PackGeneration:
    mode = mode if mode in pack.modes else "mixed"      # the pack declares which scenario modes it knows
    limit = int(pack.output.get("max_records_per_entity", 30))
    if per_entity > limit:
        raise ValueError(
            f"recordsPerUser={per_entity} exceeds what behaviour pack '{pack.pack_id}' can simulate within its "
            f"history window ({limit}). Lower recordsPerUser or raise output.max_history_days in the pack."
        )
    concept_columns = resolve_columns(variables, pack)
    if not concept_columns:
        raise ValueError("None of the confirmed variables map to the behaviour pack; nothing to generate.")
    engine = get_engine(pack.engine)
    params = pack.mode_params(mode)
    ctx, concept_rows = _simulate_with_coverage(
        engine, pack, params, set(concept_columns), seed=seed, as_of=as_of, count=count, per_entity=per_entity,
        hints=_hints(pack, variables, concept_columns))
    records = project(concept_rows, pack, concept_columns)

    # Score what is actually delivered: parse the projected columns back, so formatting bugs are visible too.
    parsed, parse_errors = parse_columns(records, pack, concept_columns)
    report = score_rows(pack, parsed, engine.reference_view(pack, params, ctx), mode=mode,
                        columns=set(concept_columns), parse_errors=parse_errors,
                        contract=build_contract(variables, concept_columns))
    report.update(seed=seed, as_of=ctx.as_of.isoformat(), engine=pack.engine,
                  concept_columns=concept_columns, **_legacy_keys(report, count, per_entity, len(records)))
    entity_key = concept_columns.get(pack.entity_concept)
    return PackGeneration(records=records, fields=list(records[0]) if records else [],
                          entity_key=entity_key, validation_report=report, concept_columns=concept_columns)


# Rows a coverage search may simulate in total: small requests (where chance can leave a conditional fact
# unrepresented) get many attempts, large ones (where it cannot) get one.
_COVERAGE_ROW_BUDGET = 20_000
_COVERAGE_MAX_ATTEMPTS = 40


def _attempt_seed(seed: int | None, attempt: int) -> int | None:
    if attempt == 0 or seed is None:
        return seed
    return (int(seed) * 1_000_003 + attempt) % 2_147_483_647


def _empty_columns(rows: list[dict[str, Any]], concepts: set[str]) -> set[str]:
    return {c for c in concepts if all(row.get(c) is None for row in rows)}


def _simulate_with_coverage(engine, pack: BehaviorPack, params: dict[str, Any], concepts: set[str], *, seed: int | None,
                            as_of: Any, count: int, per_entity: int, hints: dict[str, Any]):
    """Simulate, preferring a dataset in which every delivered column carries at least one value.

    A conditional fact (a recurring-top-up period, a suspension reason) is empty on the rows it does not apply
    to, but in a small dataset chance alone can leave it empty everywhere, which reads as a broken column. The
    attempts are a deterministic sequence derived from the seed (the first is the seed itself), so the same
    request always yields the same dataset; the first attempt that represents every column wins, otherwise the
    one with the fewest empty columns.
    """
    base_as_of = RunContext(seed=0, as_of=as_of, tz_name=pack.timezone).as_of
    attempts = max(1, min(_COVERAGE_MAX_ATTEMPTS, _COVERAGE_ROW_BUDGET // max(1, count * per_entity)))
    best = None
    for attempt in range(attempts):
        ctx = RunContext(seed=_attempt_seed(seed, attempt), as_of=base_as_of, tz_name=pack.timezone)
        rows = engine.simulate(pack, params, ctx, entities=count, per_entity=per_entity, hints=hints)
        empty = len(_empty_columns(rows, concepts))
        if best is None or empty < best[0]:
            best = (empty, ctx, rows)
        if empty == 0:
            break
    return best[1], best[2]


def _legacy_keys(report: dict[str, Any], count: int, per_entity: int, produced: int) -> dict[str, Any]:
    """Keep the keys existing clients read, but fill them with *measured* values."""
    comps = report.get("components", {})
    expected = count * per_entity
    return {
        "requested_records": expected,
        "total_valid": produced,
        "valid_record_rate": round(100.0 * produced / expected, 3) if expected else 100.0,
        "clean_record_rate": round(100.0 * comps.get("clean_rows", 0.0), 3),
        "contract_pass_rate": round(100.0 * comps.get("consistency", 0.0), 3),
        "quality_target_met": report.get("verdict") == "pass",
        "deterministic_checks": ["pack_invariants", "distribution_targets", "structure"],
    }
