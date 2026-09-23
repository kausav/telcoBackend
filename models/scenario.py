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
        """Persist the latest confirmed draft under the requested scenario id.

        ``requested_scenario_id`` is the canonical business identifier. Confirming a
        different draft for the same requested id is an update/re-confirmation, not a
        duplicate scenario. The newest confirmed draft becomes the active scenario
        definition while the canonical requested id remains unchanged.
        """
        requested = str(requested_id or "").strip() or None
        if not requested:
            raise ValueError("requested_scenario_id is required")

        now = time.time()
        persisted_meta = dict(meta)
        persisted_meta["requested_scenario_id"] = requested
        persisted_meta["scenario_id"] = requested

        # Prefer the canonical field, but also recover legacy rows that only have
        # scenario_id. This keeps reconfirmation safe during migration.
        existing = cls.collection.find_one(
            {"$or": [{"requested_scenario_id": requested}, {"scenario_id": requested}]},
            {"_id": 1, "draft_id": 1},
        )

        document = {
            "scenario_id": requested,
            "requested_scenario_id": requested,
            "scenario_id_reassigned": False,
            "draft_id": draft_id,
            "meta": persisted_meta,
            "variables": variables,
            "field_order": field_order,
            "updated_at": now,
        }

        try:
            if existing:
                # A new draft with the same requested_scenario_id is a new confirmed
                # revision of that scenario. Replace the active definition rather than
                # returning a false duplicate conflict.
                document["previous_draft_id"] = existing.get("draft_id")
                cls.collection.update_one({"_id": existing["_id"]}, {"$set": document})
            else:
                document["created_at"] = now
                cls.collection.insert_one(document)
        except DuplicateKeyError as exc:
            # Handle a race where another request confirmed the same requested id
            # between our lookup and insert. Re-read and update that canonical row.
            raced = cls.collection.find_one(
                {"$or": [{"requested_scenario_id": requested}, {"scenario_id": requested}]},
                {"_id": 1, "draft_id": 1},
            )
            if not raced:
                raise ValueError(f"Could not persist scenario '{requested}'") from exc
            document["previous_draft_id"] = raced.get("draft_id")
            cls.collection.update_one({"_id": raced["_id"]}, {"$set": document})

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
