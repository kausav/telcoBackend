"""Honest dataset scorer.

What the old ``validation_report`` did: it reported ``contract_pass_rate`` from a constant and
"valid records" from generators that could not fail, so it could never disagree with the data.

What this does: it evaluates, on the *final rows*,

* **invariants** - the pack's logical rules (record-, entity- and dataset-level), each written as a
  small expression in the pack, i.e. independently of the engine that produced the rows;
* **targets** - distribution expectations (shares, medians) with acceptance ranges;
* **structure** - required columns present, parseable types, *duplicate columns that output the
  same value on every row* (the "duplicate variables" problem, measured directly on the data), and the
  output contract of the curated DB definitions (allowed choices, numeric ranges, nullability).

What it does NOT do: prove the data matches the real world. It measures conformance to the pack, so a
score is only as meaningful as the pack's rules and ranges. Calibrate the pack with real aggregates
before quoting a score as "accuracy against reality".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from synth.contract import check_contract
from synth.expr import Expr
from synth.pack import BehaviorPack, Target

MAX_EXAMPLES = 3
WEIGHTS = {"consistency": 0.4, "clean_rows": 0.3, "realism": 0.3}


def _short(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def _evaluate(expr: Expr, env: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        return bool(expr(env)), None
    except Exception as exc:     # None-arithmetic etc. -> the rule is violated / not robust
        return False, f"{type(exc).__name__}: {exc}"


def score_rows(
    pack: BehaviorPack,
    rows: list[dict[str, Any]],
    ref: dict[str, Any],
    *,
    mode: str = "mixed",
    columns: set[str] | None = None,
    parse_errors: list[dict[str, Any]] | None = None,
    contract: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score ``rows`` (``concept -> value`` dicts, datetimes aware UTC) against ``pack``."""
    n = len(rows)
    present = set(columns) if columns is not None else (set(rows[0]) if rows else set())
    concept_names = set(pack.concepts)
    report: dict[str, Any] = {"rows": n, "mode": mode, "pack_id": pack.pack_id, "pack_version": pack.version}
    if n == 0:
        report.update(overall=0.0, verdict="fail", note="no rows")
        return report

    def env_row(r: dict[str, Any]) -> dict[str, Any]:
        env = {c: r.get(c) for c in present}
        env["REF"] = ref
        return env

    # ---- invariants ----------------------------------------------------------------------------
    entity_c = pack.entity_concept
    order_c = next((c for c, spec in pack.concepts.items() if spec.required and spec.dtype == "datetime"), None)
    by_entity: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    if entity_c in present:
        for r in rows:
            by_entity[r.get(entity_c)].append(r)
        if order_c in present:
            for grp in by_entity.values():
                grp.sort(key=lambda r: (r.get(order_c) is None, r.get(order_c)))

    results: list[dict[str, Any]] = []
    row_fail = [False] * n
    for inv in pack.invariants:
        expr = Expr(inv.expr)
        needed = set(inv.requires) | (expr.names & concept_names)
        absent = needed - present
        if inv.level == "entity" and entity_c not in present:
            absent = absent | {entity_c}
        if absent:
            results.append({"id": inv.id, "level": inv.level, "severity": inv.severity, "status": "skipped",
                            "reason": "columns absent: " + ", ".join(sorted(absent))})
            continue
        examples: list[dict[str, Any]] = []
        if inv.level == "record":
            failed = 0
            for i, r in enumerate(rows):
                ok, err = _evaluate(expr, env_row(r))
                if not ok:
                    failed += 1
                    if inv.severity == "error":
                        row_fail[i] = True
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"row": i, "error": err, **{c: _short(r.get(c)) for c in sorted(needed)}})
            total, passed = n, n - failed
        elif inv.level == "entity":
            failed = 0
            for key, grp in by_entity.items():
                env = {c: [g.get(c) for g in grp] for c in present}
                env["REF"] = ref
                ok, err = _evaluate(expr, env)
                if not ok:
                    failed += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"entity": key, "error": err})
            total, passed = len(by_entity), len(by_entity) - failed
        else:  # dataset
            env = {c: [r.get(c) for r in rows] for c in present}
            env["REF"] = ref
            ok, err = _evaluate(expr, env)
            total, passed, failed = 1, int(ok), int(not ok)
            if not ok:
                examples.append({"error": err})
        results.append({
            "id": inv.id, "level": inv.level, "severity": inv.severity, "status": "pass" if failed == 0 else "fail",
            "evaluated": total, "violations": failed, "pass_rate": passed / total if total else 1.0,
            "message": inv.message, "examples": examples,
        })

    evaluated = [r for r in results if r["status"] != "skipped"]
    consistency = (sum(r["pass_rate"] for r in evaluated) / len(evaluated)) if evaluated else 1.0
    clean_rows = 1.0 - (sum(row_fail) / n)

    # ---- targets -------------------------------------------------------------------------------
    target_results = [_score_target(t, rows, present, concept_names, ref, len(by_entity) or n) for t in pack.mode_targets(mode)]
    scored_t = [t for t in target_results if t["status"] in ("pass", "fail")]
    realism = (sum(1 for t in scored_t if t["status"] == "pass") / len(scored_t)) if scored_t else None

    # ---- structure -----------------------------------------------------------------------------
    missing_required = sorted(c for c, spec in pack.concepts.items() if spec.required and c not in present)
    dup = _duplicate_columns(rows, present, pack)
    contract_violations = check_contract(rows, contract or [])
    structure = {
        "contract_checks": len(contract or []),
        "contract_violations": contract_violations,
        "missing_required_concepts": missing_required,
        "parse_errors": (parse_errors or [])[:10],
        "parse_error_count": len(parse_errors or []),
        "duplicate_columns": dup,
    }
    structure_ok = not missing_required and not dup and not parse_errors and not contract_violations

    parts = {"consistency": consistency, "clean_rows": clean_rows}
    if realism is not None:
        parts["realism"] = realism
    wsum = sum(WEIGHTS[k] for k in parts)
    overall = sum(WEIGHTS[k] * v for k, v in parts.items()) / wsum
    error_failures = [r["id"] for r in evaluated if r["severity"] == "error" and r["status"] == "fail"]
    # Rows are consistent by construction, so a single error-level violation is a defect, not noise.
    verdict = "pass" if (not error_failures and (realism is None or realism >= 0.9) and structure_ok) else "fail"

    report.update(
        overall=round(overall, 4),
        components={k: round(v, 4) for k, v in parts.items()},
        weights={k: WEIGHTS[k] for k in parts},
        verdict=verdict,
        invariants={
            "evaluated": len(evaluated), "skipped": len(results) - len(evaluated),
            "failed": [r for r in evaluated if r["status"] == "fail"],
            "all": results,
        },
        targets={"scored": len(scored_t), "skipped": len(target_results) - len(scored_t), "results": target_results},
        structure=structure,
        note=("Scores measure conformance to the behaviour pack (logic + plausible ranges), not truth about a real "
              "population. Calibrate the pack with real aggregates to turn it into an accuracy claim."),
    )
    return report


def _wilson(p: float, n: float, z: float = 2.576) -> tuple[float, float]:
    """99% Wilson score interval for a proportion observed on ``n`` (effective) trials."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _score_target(t: Target, rows: list[dict[str, Any]], present: set[str], concepts: set[str],
                  ref: dict[str, Any], entities: int) -> dict[str, Any]:
    exprs = [Expr(s) for s in (t.where, t.condition) if s]
    needed = set().union(*(e.names & concepts for e in exprs)) if exprs else set()
    if t.concept:
        needed.add(t.concept)
    base = {"id": t.id, "stat": t.stat, "min": t.min, "max": t.max, "description": t.description}
    if not needed <= present:
        return {**base, "status": "skipped", "reason": "columns absent: " + ", ".join(sorted(needed - present))}
    where = Expr(t.where) if t.where else None
    cond = Expr(t.condition) if t.condition else None
    population = 0
    hits = 0
    values: list[float] = []
    for r in rows:
        env = {c: r.get(c) for c in present}
        env["REF"] = ref
        try:
            if where is not None and not where(env):
                continue
            if t.stat == "share":
                hit = bool(cond(env))
            else:
                v = r.get(t.concept)
                if v is None:
                    continue
                values.append(float(v))
                hit = True
        except Exception:
            continue          # not evaluable for this row (e.g. None) -> outside the population
        population += 1
        hits += 1 if hit else 0
    if population < t.min_n:
        return {**base, "status": "skipped", "n": population, "reason": f"only {population} rows qualify (min_n={t.min_n})"}
    interval = None
    if t.stat == "share":
        observed = hits / population
        # Rows of one entity are not independent draws (retries, habits), so the evidence is about
        # ``entities * 3`` observations, not ``rows``. A target fails only when the whole 99% interval
        # lies outside [min, max] - small datasets must not fail on sampling noise.
        lo, hi = _wilson(observed, min(population, entities * 3))
        interval = [round(lo, 4), round(hi, 4)]
        ok = hi >= t.min - 1e-9 and lo <= t.max + 1e-9      # tolerance: an exact 0 or 1 sits on the interval's float edge
        return {**base, "status": "pass" if ok else "fail", "observed": round(observed, 4), "n": population,
                "interval99": interval}
    else:
        values.sort()
        m = len(values)
        observed = values[m // 2] if t.stat == "median" and m % 2 else (
            (values[m // 2 - 1] + values[m // 2]) / 2 if t.stat == "median" else sum(values) / m)
    ok = t.min <= observed <= t.max
    return {**base, "status": "pass" if ok else "fail", "observed": round(observed, 4), "n": population}


def _reads(pack: BehaviorPack) -> dict[str, set[str]]:
    """concept -> concepts its emit reads (how the pack says one fact is derived from another)."""
    from synth.samplers import sampler_names

    reads: dict[str, set[str]] = {}
    for emit in pack.emit:
        names: set[str] = set()
        for src in (emit.when, emit.expr):
            if src:
                names |= Expr(src).names
        if emit.sample is not None:
            names |= sampler_names(emit.sample)
        reads[emit.concept] = names
    return reads


def _duplicate_columns(rows: list[dict[str, Any]], present: set[str], pack: BehaviorPack) -> list[dict[str, Any]]:
    """Pairs of non-entity columns that carry the same varying value on *every* row (same output).

    Not reported: constant columns (two columns that never vary are not the same variable), columns one of which
    the pack derives from the other (they may coincide on a small sample), and groups the pack declares in
    ``output.identical_concepts`` - columns the business defined as two names for one fact.
    """
    declared = [set(group) for group in pack.output.get("identical_concepts", [])]
    reads = _reads(pack)
    cols = [c for c in sorted(present) if c in pack.concepts and pack.concepts[c].kind != "entity"]
    sigs: dict[tuple, list[str]] = defaultdict(list)
    for c in cols:
        values = tuple(str(_short(r.get(c))) for r in rows)
        if len(set(values)) < 2:
            continue
        sigs[values].append(c)

    def related(names: list[str]) -> bool:
        return any(a in reads.get(b, ()) or b in reads.get(a, ()) for a in names for b in names if a != b)

    return [{"columns": names, "reason": "identical value on every row"} for names in sigs.values()
            if len(names) > 1 and not related(names) and not any(set(names) <= group for group in declared)]
