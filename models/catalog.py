"""Persistence for the scenario catalog: the choices the UI offers, one collection per level.

industry -> sub-domain -> use case -> journey -> scenario. Every row carries the id of its parent, so one level is read with
one indexed query; names, order and the active flag are what the dropdowns show.
"""
from __future__ import annotations

import time
from typing import Any

from models.database import get_database


class _Level:
    collection: Any = None
    id_field = ""
    parent_field: str | None = None
    sort_fields: tuple[tuple[str, int], ...] = (("order", 1), ("created_at", 1))

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index(cls.id_field, unique=True)
        if cls.parent_field:
            cls.collection.create_index([(cls.parent_field, 1), ("order", 1)])

    @classmethod
    def rows(cls, parent_id: str | None = None, *, active_only: bool = True) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if cls.parent_field and parent_id is not None:
            query[cls.parent_field] = parent_id
        if active_only:
            query["active"] = True
        return list(cls.collection.find(query, {"_id": 0}).sort(list(cls.sort_fields)))

    @classmethod
    def get(cls, item_id: str, *, active_only: bool = True) -> dict[str, Any] | None:
        query: dict[str, Any] = {cls.id_field: item_id}
        if active_only:
            query["active"] = True
        return cls.collection.find_one(query, {"_id": 0})

    @classmethod
    def _next_order(cls, row: dict[str, Any]) -> int:
        """A new entry goes after its siblings, so importing more of the catalog never reshuffles the dropdowns."""
        query = {cls.parent_field: row[cls.parent_field]} if cls.parent_field else {}
        return max((int(r.get("order") or 0) for r in cls.collection.find(query, {"order": 1, "_id": 0})), default=-1) + 1

    @classmethod
    def save(cls, row: dict[str, Any], *, overwrite: bool) -> bool:
        """Store ``row``; True when it is new. Without ``overwrite`` an existing row is left exactly as it is.

        An existing entry keeps its position; only a new one is given one.
        """
        now = time.time()
        key = {cls.id_field: row[cls.id_field]}
        body = {k: v for k, v in row.items() if k != cls.id_field}
        if cls.collection.find_one(key, {"_id": 1}) is not None:
            if overwrite:
                cls.collection.update_one(key, {"$set": {**body, "updated_at": now}})
            return False
        update = {"$setOnInsert": {**body, "order": cls._next_order(row), "created_at": now, "updated_at": now}}
        return cls.collection.update_one(key, update, upsert=True).upserted_id is not None


class IndustryModel(_Level):
    collection = get_database()["catalog_industries"]
    id_field = "industry_id"


class SubDomainModel(_Level):
    collection = get_database()["catalog_sub_domains"]
    id_field = "sub_domain_id"
    parent_field = "industry_id"


class UseCaseModel(_Level):
    collection = get_database()["catalog_use_cases"]
    id_field = "use_case_id"
    parent_field = "sub_domain_id"


class JourneyModel(_Level):
    collection = get_database()["catalog_journeys"]
    id_field = "journey_id"
    parent_field = "use_case_id"


class ScenarioCatalogModel(_Level):
    collection = get_database()["catalog_scenarios"]
    id_field = "scenario_id"
    parent_field = "journey_id"


CATALOG_MODELS = (IndustryModel, SubDomainModel, UseCaseModel, JourneyModel, ScenarioCatalogModel)
