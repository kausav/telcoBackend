"""Persistence for locked proposal selections (one immutable record per distinct proposal input)."""
from __future__ import annotations

import time
from typing import Any

from pymongo import ReturnDocument

from models._helpers import collection_name
from models.database import get_database


class SelectionLockModel:
    collection = get_database()[collection_name("MONGODB_SELECTION_LOCKS_COLLECTION", "scenario_selection_locks")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("lock_key", unique=True)

    @classmethod
    def get(cls, lock_key: str) -> dict[str, Any] | None:
        row = cls.collection.find_one({"lock_key": lock_key}, {"_id": 0, "payload": 1})
        return dict(row["payload"]) if row else None

    @classmethod
    def put_if_absent(cls, lock_key: str, requested_scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Store ``payload`` unless a concurrent request already did; always return what is stored."""
        row = cls.collection.find_one_and_update(
            {"lock_key": lock_key},
            {"$setOnInsert": {"lock_key": lock_key, "requested_scenario_id": requested_scenario_id,
                              "payload": payload, "created_at": time.time()}},
            upsert=True, return_document=ReturnDocument.AFTER, projection={"_id": 0, "payload": 1},
        )
        return dict(row["payload"])


SelectionLockModel.ensure_indexes()
