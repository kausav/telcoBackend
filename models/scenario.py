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
        cls.collection.create_index("scenario_id", unique=True)
        cls.collection.create_index("draft_id")

    @classmethod
    def next_id(cls) -> str:
        ids = cls.collection.find({}, {"scenario_id": 1, "_id": 0})
        numbers = [
            int(match.group(1))
            for row in ids
            if (match := re.match(r"LB-(\d+)$", str(row.get("scenario_id"))))
        ]
        return f"LB-{(max(numbers) + 1) if numbers else 1:02d}"

    @classmethod
    def create(cls, scenario_id: str, requested_scenario_id: str | None, draft_id: str | None,
               meta: dict[str, Any], variables: list[dict[str, Any]], field_order: list[str],
               reassigned: bool = False) -> None:
        now = time.time()
        cls.collection.insert_one({
            "scenario_id": scenario_id,
            "requested_scenario_id": requested_scenario_id,
            "scenario_id_reassigned": reassigned,
            "draft_id": draft_id,
            "meta": meta,
            "variables": variables,
            "field_order": field_order,
            "created_at": now,
            "updated_at": now,
        })

    @classmethod
    def create_with_allocation(cls, requested_id: str | None, draft_id: str | None,
                               meta: dict[str, Any], variables: list[dict[str, Any]],
                               field_order: list[str]) -> tuple[str, bool]:
        requested = requested_id or None
        for _ in range(3):
            final_id = requested or cls.next_id()
            reassigned = bool(requested and final_id != requested)
            try:
                cls.create(final_id, requested, draft_id, meta, variables, field_order, reassigned)
                return final_id, reassigned
            except DuplicateKeyError:
                final_id = cls.next_id()
                try:
                    cls.create(final_id, requested, draft_id, meta, variables, field_order, True)
                    return final_id, True
                except DuplicateKeyError:
                    requested = None
        raise RuntimeError("Could not allocate a unique scenario_id")

    @classmethod
    def by_draft_id(cls, draft_id: str) -> str | None:
        row = cls.collection.find_one({"draft_id": draft_id}, {"scenario_id": 1})
        return row.get("scenario_id") if row else None

    @classmethod
    def get(cls, scenario_id: str) -> dict[str, Any] | None:
        row = cls.collection.find_one(
            {"scenario_id": scenario_id},
            {"meta": 1, "variables": 1, "field_order": 1},
        )
        if not row:
            return None
        return {
            "meta": row.get("meta", {}),
            "variables": row.get("variables", []),
            "field_order": row.get("field_order", []),
        }

    @classmethod
    def exists(cls, scenario_id: str) -> bool:
        return cls.collection.count_documents({"scenario_id": scenario_id}, limit=1) > 0

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        return [
            {"id": row["scenario_id"], **(row.get("meta") or {})}
            for row in cls.collection.find({}, {"scenario_id": 1, "meta": 1, "_id": 0}).sort("scenario_id", 1)
        ]


ScenarioModel.ensure_indexes()
