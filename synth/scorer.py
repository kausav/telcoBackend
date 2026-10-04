"""Honest dataset scorer.

What the old ``validation_report`` did: it reported ``contract_pass_rate`` from a constant and
"valid records" from generators that could not fail, so it could never disagree with the data.

What this does: it evaluates, on the *final rows*,

* **invariants** - the spec's logical rules (record-, entity- and dataset-level), each written as a
  small expression in the spec, i.e. independently of the engine that produced the rows;
* **targets** - distribution expectations (shares, medians) with acceptance ranges;
* **structure** - parseable types, *duplicate columns that output the same value on every row* (the
  "duplicate variables" problem, measured directly on the data), and the output contract of the variable
  definitions (allowed choices, numeric ranges, patterns, nullability) - which comes from the definitions and
  the industry source documents, not from the spec.

What it does NOT do: prove the data matches the real world. The spec's rules and targets are written from the
scenario's description, so they measure internal consistency and conformance to the definitions, not
calibration against a real population. Calibrate the parameters with real aggregates before quoting a score
as "accuracy against reality".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from synth.contract import check_contract
from synth.expr import Expr
from synth.spec import GenerationSpec, Target

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
    spec: GenerationSpec,
    rows: list[dict[str, Any]],
    ref: dict[str, Any],
    *,
    columns: set[str] | None = None,
    parse_errors: list[dict[str, Any]] | None = None,
    contract: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score ``rows`` (``column id -> value`` dicts, datetimes aware UTC) against ``spec``."""
    n = len(rows)
    present = set(columns) if columns is not None else (set(rows[0]) if rows else set())
    column_names = set(spec.columns)
    report: dict[str, Any] = {"rows": n, "spec_source": spec.source}
    if n == 0:
        report.update(overall=0.0, verdict="fail", note="no rows")
        return report

    def env_row(r: dict[str, Any]) -> dict[str, Any]:
        env = {c: r.get(c) for c in present}
        env["REF"], env["as_of"], env["P"] = ref, ref.get("as_of"), ref.get("model")
        return env

    # ---- invariants ----------------------------------------------------------------------------
    entity_c = spec.entity_column
    order_c = spec.clock
    by_entity: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    if entity_c is not None and entity_c in present:
        for r in rows:
            by_entity[r.get(entity_c)].append(r)
        if order_c in present:
            for grp in by_entity.values():
                grp.sort(key=lambda r: (r.get(order_c) is None, r.get(order_c)))

    results: list[dict[str, Any]] = []
    row_fail = [False] * n
    for inv in spec.invariants:
        expr = Expr(inv.expr)
        needed = set(inv.requires) | (expr.names & column_names)
        absent = needed - present
        if inv.level == "entity" and (entity_c is None or entity_c not in present):
            absent = absent | {entity_c or "<entity column>"}
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
                env["REF"], env["as_of"], env["P"] = ref, ref.get("as_of"), ref.get("model")
                ok, err = _evaluate(expr, env)
                if not ok:
                    failed += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"entity": key, "error": err})
            total, passed = len(by_entity), len(by_entity) - failed
        else:  # dataset
            env = {c: [r.get(c) for r in rows] for c in present}
            env["REF"], env["as_of"], env["P"] = ref, ref.get("as_of"), ref.get("model")
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
    target_results = [_score_target(t, rows, present, column_names, ref, len(by_entity) or n) for t in spec.targets]
    scored_t = [t for t in target_results if t["status"] in ("pass", "fail")]
    realism = (sum(1 for t in scored_t if t["status"] == "pass") / len(scored_t)) if scored_t else None

    # ---- structure -----------------------------------------------------------------------------
    dup = _duplicate_columns(rows, present, spec)
    contract_bad: set[int] = set()
    contract_violations = check_contract(rows, contract or [], contract_bad)
    structure = {
        "contract_checks": len(contract or []),
        "contract_violations": contract_violations,
        "contract_rows_failed": len(contract_bad),
        "parse_errors": (parse_errors or [])[:10],
        "parse_error_count": len(parse_errors or []),
        "duplicate_columns": dup,
    }
    structure_ok = not dup and not parse_errors and not contract_violations

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
        rows_failed=sum(1 for i in range(n) if row_fail[i] or i in contract_bad),
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
        note=("Scores measure consistency with the spec's own rules and conformance to the variable definitions (allowed "
              "values, ranges, patterns, nullability), not truth about a real population. Calibrate the spec's "
              "parameters with real aggregates to turn it into an accuracy claim."),
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


def _score_target(t: Target, rows: list[dict[str, Any]], present: set[str], columns: set[str],
                  ref: dict[str, Any], entities: int) -> dict[str, Any]:
    exprs = [Expr(s) for s in (t.where, t.condition) if s]
    needed = set().union(*(e.names & columns for e in exprs)) if exprs else set()
    if t.column:
        needed.add(t.column)
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
        env["REF"], env["as_of"], env["P"] = ref, ref.get("as_of"), ref.get("model")
        try:
            if where is not None and not where(env):
                continue
            if t.stat == "share":
                hit = bool(cond(env))
            else:
                v = r.get(t.column)
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
        # Rows of one entity are not independent draws, so with few entities a median or mean moves a lot from one run to the
        # next. A target fails only when the whole 99% interval of the statistic lies outside [min, max].
        effective = max(1.0, min(float(m), entities * 3.0))
        if t.stat == "median":
            half = 2.576 * 0.5 / effective ** 0.5
            lo = values[max(0, min(m - 1, int((0.5 - half) * (m - 1))))]
            hi = values[max(0, min(m - 1, int(round((0.5 + half) * (m - 1)))))]
        else:
            sd = (sum((v - observed) ** 2 for v in values) / max(1, m - 1)) ** 0.5
            lo, hi = observed - 2.576 * sd / effective ** 0.5, observed + 2.576 * sd / effective ** 0.5
        interval = [round(lo, 4), round(hi, 4)]
        ok = hi >= t.min - 1e-9 and lo <= t.max + 1e-9
        return {**base, "status": "pass" if ok else "fail", "observed": round(observed, 4), "n": population, "interval99": interval}


def _reads(spec: GenerationSpec) -> dict[str, set[str]]:
    """column -> columns its emit reads (how the spec says one fact is derived from another)."""
    from synth.samplers import sampler_names

    reads: dict[str, set[str]] = {}
    for emit in spec.emit:
        names: set[str] = set()
        for src in (emit.when, emit.expr):
            if src:
                names |= Expr(src).names
        if emit.sample is not None:
            names |= sampler_names(emit.sample)
        reads[emit.column] = names
    return reads


def _duplicate_columns(rows: list[dict[str, Any]], present: set[str], spec: GenerationSpec) -> list[dict[str, Any]]:
    """Pairs of non-entity columns that carry the same varying value on *every* row (same output).

    Not reported: constant columns (two columns that never vary are not the same variable), columns one of which
    the spec derives from the other (they may coincide on a small sample), and groups the spec declares in
    ``output.identical_columns`` - columns the business defined as two names for one fact.
    """
    declared = [set(group) for group in spec.output.get("identical_columns", [])]
    reads = _reads(spec)
    cols = [c for c in sorted(present) if c in spec.columns and spec.columns[c].kind == "event"]
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
