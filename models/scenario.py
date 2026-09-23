"""Persistence model for confirmed scenarios."""
from __future__ import annotations

import re
import time
from typing import Any
from pymongo.errors import DuplicateKeyError
from models._helpers import collection_name
from models.database import get_database


class ScenarioModel:
    collection = get_database()[collection_name("MONGODB_SCENARIOS_COLLECTION", "scenarios")]

    @classmethod
    def ensure_indexes(cls) -> None:
        # requested_scenario_id is the canonical business/source identifier for all new data.
        # Keep scenario_id as a compatibility alias, but never allocate a different id for a
        # confirmed scenario.
        cls.collection.create_index("requested_scenario_id", unique=True, sparse=True)
        cls.collection.create_index("scenario_id", unique=True, sparse=True)
        cls.collection.create_index("draft_id")

    @classmethod
    def next_id(cls) -> str:
        # Legacy helper retained for compatibility. New scenarios must supply the requested
        # scenario id and are never silently renamed.
        ids = cls.collection.find({}, {"requested_scenario_id": 1, "scenario_id": 1, "_id": 0})
        numbers = []
        for row in ids:
            value = row.get("requested_scenario_id") or row.get("scenario_id")
            if match := re.match(r"LB-(\d+)$", str(value)):
                numbers.append(int(match.group(1)))
        return f"LB-{(max(numbers) + 1) if numbers else 1:02d}"

    @classmethod
    def create(cls, scenario_id: str, requested_scenario_id: str | None, draft_id: str | None,
               meta: dict[str, Any], variables: list[dict[str, Any]], field_order: list[str],
               reassigned: bool = False) -> None:
        now = time.time()
        canonical = str(requested_scenario_id or "").strip() or None
        if not canonical:
            raise ValueError("requested_scenario_id is required")
        if str(scenario_id).strip() != canonical:
            raise ValueError("scenario_id must equal requested_scenario_id")
        persisted_meta = dict(meta)
        persisted_meta["requested_scenario_id"] = canonical
        persisted_meta["scenario_id"] = canonical
        cls.collection.insert_one({
            "scenario_id": canonical,
            "requested_scenario_id": canonical,
            "scenario_id_reassigned": reassigned,
            "draft_id": draft_id,
            "meta": persisted_meta,
            "variables": variables,
            "field_order": field_order,
            "created_at": now,
            "updated_at": now,
        })

    @classmethod
    def create_with_allocation(cls, requested_id: str | None, draft_id: str | None,
                               meta: dict[str, Any], variables: list[dict[str, Any]],
                               field_order: list[str]) -> tuple[str, bool]:
        requested = str(requested_id or "").strip() or None
        if not requested:
            raise ValueError("requested_scenario_id is required")
        # requested_scenario_id is the source of truth. Never replace it with an internally
        # generated/reassigned scenario id. A duplicate is a conflict that the caller must
        # resolve explicitly.
        if cls.collection.find_one({"requested_scenario_id": requested}, {"_id": 1}):
            raise ValueError(f"Scenario '{requested}' already exists")
        try:
            cls.create(requested, requested, draft_id, meta, variables, field_order, False)
        except DuplicateKeyError as exc:
            raise ValueError(f"Scenario '{requested}' already exists") from exc
        return requested, False

    @classmethod
    def by_draft_id(cls, draft_id: str) -> str | None:
        row = cls.collection.find_one(
            {"draft_id": draft_id},
            {"requested_scenario_id": 1, "scenario_id": 1},
        )
        return (row.get("requested_scenario_id") or row.get("scenario_id")) if row else None

    @classmethod
    def get(cls, requested_scenario_id: str) -> dict[str, Any] | None:
        row = cls.collection.find_one(
            {"requested_scenario_id": requested_scenario_id},
            {"meta": 1, "variables": 1, "field_order": 1, "requested_scenario_id": 1},
        )
        if not row:
            return None
        return {
            "meta": row.get("meta", {}),
            "variables": row.get("variables", []),
            "field_order": row.get("field_order", []),
        }

    @classmethod
    def exists(cls, requested_scenario_id: str) -> bool:
        return cls.collection.count_documents({"requested_scenario_id": requested_scenario_id}, limit=1) > 0

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        rows = cls.collection.find(
            {}, {"requested_scenario_id": 1, "scenario_id": 1, "meta": 1, "_id": 0}
        ).sort("requested_scenario_id", 1)
        return [
            {"id": row.get("requested_scenario_id") or row.get("scenario_id"), **(row.get("meta") or {})}
            for row in rows
        ]


ScenarioModel.ensure_indexes()
