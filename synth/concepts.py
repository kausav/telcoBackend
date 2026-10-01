"""Concept canonicalisation: turn "every field the sources and the DB can offer" into "one column
per business fact the scenario needs".

Why this exists
---------------
The source standards (TMF654, TMF629, ...) describe *API shapes*. TMF654 repeats the same facets
(id, status, amount, bucket, channel, reason, requested/confirmation date, valid-for, ...) under
several sibling resources, and the
DB-recommended variables repeat them again under different spellings (``resourcex_status`` vs
``resource_x_status``). Name- or generator-signature de-duplication can never notice that two
such columns are the *same fact*, so the proposal carried 132 columns for ~26 facts.

The fix is to decide equivalence on meaning, not on spelling:

1. every column is bound to a *concept* of the behaviour pack (explicit tag > pack bind rules);
2. resources the scenario does not use are pruned by explicit, documented exclusions;
3. when several columns bind to one concept, exactly one survives - the highest-priority source
   (USER_SELECTED > DB_RECOMMENDED > MONGODB_JSON) wins, then the most specific bind;
4. columns that bind to nothing are dropped *and reported* (never silently generated as noise) -
   except *curated* columns (USER_SELECTED / DB_RECOMMENDED): a person chose those, so they are never
   dropped. Exclusions do not apply to them, and one that binds to nothing is reported as ``uncovered``;
   the caller then does not use the pack (``Canonicalization.covers`` is False).

Everything here is deterministic, dependency-free and data-driven (the pack is the only source of
domain knowledge), so the same code serves every industry that has a pack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from synth.pack import BehaviorPack, Bind

SOURCE_RANK = {"USER_SELECTED": 3, "DB_RECOMMENDED": 2, "MONGODB_JSON": 1}
CURATED = frozenset({"USER_SELECTED", "DB_RECOMMENDED"})
SCOPE_BY_KIND = {"entity": "entity", "event": "transaction", "derived": "derived"}


def normalise_name(name: str) -> str:
    """camelCase / kebab / dotted / spaced -> snake_case."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or "").strip())
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _prov(field_: dict[str, Any]) -> dict[str, Any]:
    prov = field_.get("provenance")
    return prov if isinstance(prov, dict) else {}


def _bind_specificity(bind: Bind, index: int, field_: dict[str, Any], norm: str) -> float | None:
    """Score of one bind against one column, or None when any given criterion fails."""
    prov = _prov(field_)
    score = 0.0
    if bind.name is not None:
        if not re.search(bind.name, norm):
            return None
        score += 10
    if bind.model is not None:
        model = str(prov.get("model") or prov.get("source_json_model") or "")
        if model.strip().lower() != bind.model.strip().lower():
            return None
        score += 10
    if bind.path is not None:
        path = str(prov.get("path") or prov.get("source_json_path") or "")
        if not re.search(bind.path, path):
            return None
        score += 6
    if bind.description is not None:
        if not re.search(bind.description, str(field_.get("description") or ""), re.IGNORECASE):
            return None
        score += 2
    if score == 0:
        return None
    return score - 0.01 * index   # earlier binds are the pack author's preferred spelling


def bind_column(pack: BehaviorPack, field_: dict[str, Any], *, honor_exclusions: bool = True
                ) -> tuple[str | None, float, str]:
    """Return (concept_id, specificity, how) for a column.

    ``how`` is ``explicit`` (a DB/user ``concept`` tag), ``bind`` (pack rule), ``excluded`` or
    ``none``. An explicit tag beats an exclusion; an exclusion beats a bind (unless
    ``honor_exclusions`` is off, which is how curated columns are bound).
    """
    norm = normalise_name(field_.get("name") or "")
    tag = field_.get("concept") or (field_.get("params") or {}).get("concept") or _prov(field_).get("concept")
    if tag:
        tag = str(tag)
        if tag in pack.concepts:
            return tag, 1000.0, "explicit"
    for ex in pack.exclusions if honor_exclusions else ():
        if re.search(ex.name, norm):
            return None, 0.0, "excluded"
    best: tuple[float, str] | None = None
    for cid, concept in pack.concepts.items():
        for i, bind in enumerate(concept.bind):
            score = _bind_specificity(bind, i, field_, norm)
            if score is not None and (best is None or score > best[0]):
                best = (score, cid)
    if best:
        return best[1], best[0], "bind"
    return None, 0.0, "none"


def pull_names(pack: BehaviorPack, rows: list[dict[str, Any]], covered_by: list[dict[str, Any]] | None = None) -> list[str]:
    """Names of source-catalog rows the pack binds to a concept, in a stable order (most specific first).

    A pack knows which facts it cannot run without, so it asks the catalog for exactly those columns
    instead of hoping a relevance ranking happens to pick them. One row per concept, best binding wins,
    ties broken by name, so the result never depends on iteration order.
    """
    covered = {bind_column(pack, v, honor_exclusions=False)[0] for v in covered_by or []}
    best: dict[str, tuple[float, str]] = {}
    for row in rows:
        concept, score, how = bind_column(pack, row)
        if concept is None or how != "bind" or concept in covered:
            continue
        name = str(row.get("name") or "")
        if name and (concept not in best or (-score, name) < (-best[concept][0], best[concept][1])):
            best[concept] = (score, name)
    return sorted(name for _score, name in best.values())


def exclusion_reason(pack: BehaviorPack, name: str) -> str:
    norm = normalise_name(name)
    for ex in pack.exclusions:
        if re.search(ex.name, norm):
            return ex.reason
    return ""


@dataclass
class Canonicalization:
    fields: list[dict[str, Any]]                        # kept columns, in final order
    dropped: list[dict[str, Any]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)      # derived columns created by the pack
    concept_columns: dict[str, str] = field(default_factory=dict)   # concept id -> column name
    missing_required: list[str] = field(default_factory=list)
    uncovered: list[dict[str, Any]] = field(default_factory=list)    # curated columns no concept covers
    optional_derived: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_count: int = 0

    @property
    def covers(self) -> bool:
        """True when the pack can serve every curated column and has every concept it cannot run without."""
        return not self.uncovered and not self.missing_required

    def skip_reason(self) -> str:
        parts = []
        if self.uncovered:
            parts.append("curated variables the pack has no concept for: " + ", ".join(u["name"] for u in self.uncovered))
        if self.missing_required:
            parts.append("required concepts without a column: " + ", ".join(self.missing_required))
        return "; ".join(parts)

    def report(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for d in self.dropped:
            by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
        return {
            "input_columns": self.input_count,
            "output_columns": len(self.fields),
            "dropped_by_kind": by_kind,
            "dropped": self.dropped,
            "derived_added": self.added,
            "concept_columns": self.concept_columns,
            "missing_required_concepts": self.missing_required,
            "uncovered_curated": self.uncovered,
            "optional_derived_columns": self.optional_derived,
            "warnings": self.warnings,
        }


def canonicalize(
    fields: list[dict[str, Any]],
    pack: BehaviorPack,
    sources: dict[str, str] | None = None,
) -> Canonicalization:
    """Collapse ``fields`` onto the pack's concepts. Pure function; inputs are not mutated."""
    sources = {str(k).casefold(): str(v).upper() for k, v in (sources or {}).items()}
    candidates: dict[str, list[tuple[tuple, dict[str, Any]]]] = {}
    result = Canonicalization(fields=[], input_count=len(fields))

    for index, raw in enumerate(fields):
        f = dict(raw)
        name = str(f.get("name") or "").strip()
        if not name:
            continue
        source = sources.get(name.casefold()) or str(f.get("source") or "").upper()
        curated = source in CURATED
        cid, specificity, how = bind_column(pack, f, honor_exclusions=not curated)
        if cid is None:
            if curated:
                result.uncovered.append({
                    "name": name, "source": source,
                    "reason": "Curated variable; the behaviour pack has no concept for it. Add a bind to the pack "
                              "(or a `concept` tag to the variable) so it can be generated consistently.",
                })
            elif how == "excluded":
                result.dropped.append({"name": name, "kind": "excluded", "reason": exclusion_reason(pack, name)})
            else:
                result.dropped.append({
                    "name": name, "kind": "unmapped",
                    "reason": "Not a fact the behaviour pack knows, and not a curated variable.",
                })
            continue
        rank = SOURCE_RANK.get(source, 0)
        key = (-rank, -specificity, index)            # sortable: best first
        candidates.setdefault(cid, []).append((key, f))

    winners: dict[str, dict[str, Any]] = {}
    for cid, items in candidates.items():
        items.sort(key=lambda x: x[0])
        winner = items[0][1]
        winners[cid] = winner
        for _, loser in items[1:]:
            result.dropped.append({
                "name": loser.get("name"), "kind": "duplicate", "concept": cid,
                "kept": winner.get("name"),
                "reason": f"Same business fact as '{winner.get('name')}' (concept '{cid}'); "
                          f"kept the higher-priority source.",
            })

    # Derived concepts the pack wants even when no source column exists for them.
    for cid, concept in pack.concepts.items():
        if concept.kind == "derived" and concept.default_on and cid not in winners and concept.column_name:
            winners[cid] = {
                "name": concept.column_name, "dtype": concept.dtype, "description": concept.description,
                "gen": "behavior_pack", "params": {}, "depends_on": [], "nullable": False,
                "required": False, "formula": None, "scope": "derived", "provenance": {"generated_from": "behavior_pack"},
                "source": "DERIVED", "_added": True,
            }

    # Stable, readable order: entity facts, then events in the pack's declaration order.
    for cid in pack.concepts:
        f = winners.get(cid)
        if f is None:
            continue
        concept = pack.concepts[cid]
        out = dict(f)
        added = bool(out.pop("_added", False))
        prov = dict(_prov(out))
        if out.get("gen") not in (None, "", "behavior_pack"):
            prov.setdefault("legacy_gen", out.get("gen"))
            prov.setdefault("legacy_params", out.get("params") or {})
        prov.update({"concept": cid, "behavior_pack_id": pack.pack_id, "behavior_pack_version": pack.version})
        out["provenance"] = prov
        out["concept"] = cid
        out["gen"] = "behavior_pack"
        out["params"] = {"concept": cid, "pack_id": pack.pack_id, "pack_version": pack.version}
        out["scope"] = SCOPE_BY_KIND[concept.kind]
        out["primary_timestamp"] = cid == pack.output.get("order_by")
        out["dtype"] = _compatible_dtype(str(out.get("dtype") or ""), concept.dtype, result, str(out.get("name")))
        out["nullable"] = not concept.required if not added else out.get("nullable", False)
        out["required"] = concept.required
        out["depends_on"] = []   # dependencies are explicit in the pack model, not name guesses
        out["formula"] = None
        result.fields.append(out)
        result.concept_columns[cid] = str(out["name"])
        if added:
            result.added.append(str(out["name"]))

    result.optional_derived = [
        {"concept": cid, "column": c.column_name or cid, "description": c.description}
        for cid, c in pack.concepts.items() if c.kind == "derived" and cid not in winners
    ]
    result.missing_required = [cid for cid, c in pack.concepts.items() if c.required and cid not in winners]
    if result.missing_required:
        result.warnings.append(
            "Required concepts have no column: " + ", ".join(result.missing_required) +
            ". The journey is still simulated, but these facts will be absent from the output and "
            "checks that depend on them are skipped."
        )
    return result


_DTYPE_FAMILY = {
    "string": "text", "str": "text", "text": "text", "categorical": "text",
    "integer": "num", "int": "num", "float": "num", "decimal": "num", "number": "num",
    "boolean": "bool", "bool": "bool", "datetime": "time", "date": "time", "timestamp": "time",
}


def _compatible_dtype(actual: str, wanted: str, result: Canonicalization, name: str) -> str:
    a, w = _DTYPE_FAMILY.get(actual.lower()), _DTYPE_FAMILY.get(wanted.lower())
    if a and w and a != w:
        result.warnings.append(f"Column '{name}' is declared {actual} but concept expects {wanted}; using {wanted}.")
        return wanted
    return actual or wanted
