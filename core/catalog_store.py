"""Application service for the scenario catalog: the industries, sub-domains, use cases, journeys and scenarios the UI offers.

The catalog lives in MongoDB (one collection per level, see ``models.catalog``) so the frontend holds no static lists. The
bundled ``seed/scenario_catalog.json`` is inserted the first time it is needed and never overwrites what MongoDB holds; the
import API changes or extends the catalog afterwards.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from core.catalog_models import (
    CatalogCountry,
    CatalogImportRequest,
    CatalogJourney,
    CatalogProposeDefaults,
    CatalogScenario,
    CatalogScenarioDetail,
    CatalogScenarioList,
    CatalogTree,
)
from models.catalog import IndustryModel, JourneyModel, ScenarioCatalogModel, SubDomainModel, UseCaseModel

_LEVELS = (
    ("industries", IndustryModel),
    ("subDomains", SubDomainModel),
    ("useCases", UseCaseModel),
    ("journeys", JourneyModel),
    ("scenarios", ScenarioCatalogModel),
)


class CatalogNotFound(LookupError):
    """The requested catalog entry does not exist (or is switched off)."""


# ------------------------------------------------------------------------------------------------------------------
def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-")
    return slug[:60] or "x-" + hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8]


def _flatten(request: CatalogImportRequest) -> dict[str, list[dict[str, Any]]]:
    """One row per entry of every level, each carrying the ids of everything above it."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _LEVELS}
    seen: dict[str, set[str]] = {name: set() for name, _ in _LEVELS}

    def add(level: str, row: dict[str, Any], key: str) -> None:
        if row[key] in seen[level]:
            raise ValueError(f"duplicate {level} id '{row[key]}' in the catalog")
        seen[level].add(row[key])
        rows[level].append(row)

    for industry in request.industries:
        industry_id = industry.id or _slug(industry.name)
        add("industries", {"industry_id": industry_id, "name": industry.name, "description": industry.description, "active": industry.active}, "industry_id")
        for sub in industry.subDomains:
            sub_id = sub.id or f"{industry_id}.{_slug(sub.name)}"
            add("subDomains", {"sub_domain_id": sub_id, "industry_id": industry_id, "name": sub.name, "description": sub.description, "active": sub.active}, "sub_domain_id")
            for case in sub.useCases:
                case_id = case.id or f"{sub_id}.{_slug(case.name)}"
                add("useCases", {"use_case_id": case_id, "sub_domain_id": sub_id, "industry_id": industry_id,
                                 "name": case.name, "description": case.description, "propose_value": case.proposeValue or case.name,
                                 "active": case.active}, "use_case_id")
                for journey in case.journeys:
                    journey_id = journey.id or f"{case_id}.{_slug(journey.name)}"
                    add("journeys", {"journey_id": journey_id, "use_case_id": case_id, "sub_domain_id": sub_id, "industry_id": industry_id,
                                     "name": journey.name, "description": journey.description, "active": journey.active,
                                     "data_types": list(dict.fromkeys(journey.dataTypes)),
                                     "industry_type": journey.industryType or industry.name, "domain": journey.domain or journey.name,
                                     "entity_key": journey.entityKey}, "journey_id")
                    for scenario in journey.scenarios:
                        add("scenarios", {"scenario_id": scenario.scenarioId, "journey_id": journey_id, "use_case_id": case_id,
                                          "sub_domain_id": sub_id, "industry_id": industry_id, "name": scenario.scenarioType,
                                          "scenario_type": scenario.scenarioType, "active": scenario.active,
                                          "business_scenario": scenario.businessScenario}, "scenario_id")
    return rows


def import_catalog(request: CatalogImportRequest, *, overwrite: bool = True) -> dict[str, dict[str, int]]:
    """Add the entries of ``request`` and, with ``overwrite``, update the ones that exist; nothing is ever deleted.

    Switch an entry off with ``active: false`` instead: it stays in MongoDB and leaves the dropdowns.
    """
    rows = _flatten(request)
    counts: dict[str, dict[str, int]] = {}
    for name, model in _LEVELS:
        created = sum(1 for row in rows[name] if model.save(row, overwrite=overwrite))
        counts[name] = {"created": created, "updated": (len(rows[name]) - created) if overwrite else 0}
    return counts


def seed_bundled() -> dict[str, int]:
    """Insert the bundled catalog entries MongoDB lacks; returns how many were inserted per collection."""
    from core.seed import SEED_DIR

    path = SEED_DIR / "scenario_catalog.json"
    if not path.is_file():
        return {}
    counts = import_catalog(CatalogImportRequest.model_validate(json.loads(path.read_text(encoding="utf-8"))), overwrite=False)
    return {f"catalog_{name}": value["created"] for name, value in counts.items() if value["created"]}


# ------------------------------------------------------------------------------------------------------------------
def _ready() -> None:
    from core.seed import ensure_seeded

    ensure_seeded()


def _item(row: dict[str, Any], key: str) -> dict[str, Any]:
    return {"id": row[key], "name": row["name"], "description": row.get("description"), "order": int(row.get("order") or 0)}


def _journey(row: dict[str, Any]) -> dict[str, Any]:
    return {**_item(row, "journey_id"), "dataTypes": list(row.get("data_types") or ["transactional"]),
            "industryType": row["industry_type"], "domain": row["domain"], "entityKey": row.get("entity_key")}


def _scenario(row: dict[str, Any]) -> dict[str, Any]:
    return {"scenarioId": row["scenario_id"], "scenarioType": row["scenario_type"], "label": f"{row['scenario_id']}: {row['scenario_type']}",
            "order": int(row.get("order") or 0), "businessScenario": row.get("business_scenario")}


def _children(parent_model: Any, parent_id: str, child_model: Any, build: Any) -> list[dict[str, Any]]:
    """The entries below one parent; the parent must exist, so a mistyped id is an error and not an empty dropdown."""
    _ready()
    if parent_model.get(parent_id) is None:
        raise CatalogNotFound(f"'{parent_id}' is not in the catalog")
    return [build(row) for row in child_model.rows(parent_id)]


def industries() -> list[dict[str, Any]]:
    _ready()
    return [_item(row, "industry_id") for row in IndustryModel.rows()]


def sub_domains(industry_id: str) -> list[dict[str, Any]]:
    return _children(IndustryModel, industry_id, SubDomainModel, lambda r: _item(r, "sub_domain_id"))


def use_cases(sub_domain_id: str) -> list[dict[str, Any]]:
    return _children(SubDomainModel, sub_domain_id, UseCaseModel, lambda r: _item(r, "use_case_id"))


def journeys(use_case_id: str) -> list[dict[str, Any]]:
    return _children(UseCaseModel, use_case_id, JourneyModel, _journey)


def scenarios(journey_id: str) -> dict[str, Any]:
    items = _children(JourneyModel, journey_id, ScenarioCatalogModel, _scenario)
    return CatalogScenarioList(journeyId=journey_id, count=len(items), allLabel=f"All {len(items)} Scenarios for Selected Journey",
                               scenarios=[CatalogScenario(**item) for item in items]).model_dump()


def scenario_detail(scenario_id: str) -> dict[str, Any]:
    """One scenario with everything above it and the /scenario/propose fields the catalog knows."""
    _ready()
    row = ScenarioCatalogModel.get(scenario_id)
    parents = None
    if row is not None:
        journey = JourneyModel.get(row["journey_id"])
        case = UseCaseModel.get(row["use_case_id"])
        sub = SubDomainModel.get(row["sub_domain_id"])
        industry = IndustryModel.get(row["industry_id"])
        parents = (journey, case, sub, industry) if all((journey, case, sub, industry)) else None
    if row is None or parents is None:
        raise CatalogNotFound(f"scenario '{scenario_id}' is not in the catalog")
    journey, case, sub, industry = parents
    base = _scenario(row)
    return CatalogScenarioDetail(
        **base, industry=_item(industry, "industry_id"), subDomain=_item(sub, "sub_domain_id"), useCase=_item(case, "use_case_id"),
        journey=CatalogJourney(**_journey(journey)),
        proposeRequest=CatalogProposeDefaults(
            requestedScenarioId=row["scenario_id"], scenarioType=row["scenario_type"], industryType=journey["industry_type"],
            domain=journey["domain"], useCase=case.get("propose_value") or case["name"], typeOfData=(journey.get("data_types") or ["transactional"])[0],
            businessScenario=row.get("business_scenario"), entityKey=journey.get("entity_key")),
    ).model_dump()


def countries() -> list[dict[str, Any]]:
    """The countries /scenario/propose accepts, from the generator's own country metadata."""
    from config.country_metadata import COUNTRY_BASE

    ordered = sorted(COUNTRY_BASE.items(), key=lambda item: (item[0] == "GLOBAL", str(item[1].get("country_name") or item[0])))
    return [CatalogCountry(code=code, name=str(meta.get("country_name") or code), currency=meta.get("currency") or None).model_dump()
            for code, meta in ordered]


def tree() -> dict[str, Any]:
    """The whole active catalog in one call (five queries, assembled in memory)."""
    _ready()
    by_parent: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, model in _LEVELS[1:]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        parent_key = model.parent_field
        for row in model.rows():
            grouped.setdefault(row[parent_key], []).append(row)
        by_parent[name] = grouped
    out = []
    for industry in IndustryModel.rows():
        subs = []
        for sub in by_parent["subDomains"].get(industry["industry_id"], []):
            cases = []
            for case in by_parent["useCases"].get(sub["sub_domain_id"], []):
                nodes = [{**_journey(journey), "scenarios": [_scenario(s) for s in by_parent["scenarios"].get(journey["journey_id"], [])]}
                         for journey in by_parent["journeys"].get(case["use_case_id"], [])]
                cases.append({**_item(case, "use_case_id"), "journeys": nodes})
            subs.append({**_item(sub, "sub_domain_id"), "useCases": cases})
        out.append({**_item(industry, "industry_id"), "subDomains": subs})
    return CatalogTree(industries=out).model_dump()
