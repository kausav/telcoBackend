"""Persistence model for temporary scenario drafts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any
import uuid
from models._helpers import collection_name
from models.database import get_database


DRAFT_TTL_SECONDS = int(os.getenv("DRAFT_TTL_SECONDS", "86400"))


class ScenarioDraftModel:
    collection = get_database()[collection_name("MONGODB_DRAFTS_COLLECTION", "scenario_drafts")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("draft_id", unique=True)
        cls.collection.create_index("created_at", expireAfterSeconds=DRAFT_TTL_SECONDS)

    @staticmethod
    def new_id() -> str:
        return f"draft-{uuid.uuid4().hex}"

    @classmethod
    def purge_expired(cls) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=DRAFT_TTL_SECONDS)
        cls.collection.delete_many({"created_at": {"$lt": cutoff}})

    @classmethod
    def save(cls, draft_id: str, data: dict[str, Any]) -> None:
        cls.purge_expired()
        stored = dict(data)
        requested = str(stored.get("requested_scenario_id") or stored.get("scenario_id") or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        stored["requested_scenario_id"] = requested
        stored["scenario_id"] = requested
        cls.collection.replace_one(
            {"draft_id": draft_id},
            {"draft_id": draft_id, "requested_scenario_id": requested or None, "data": stored, "created_at": datetime.now(timezone.utc)},
            upsert=True,
        )

    @classmethod
    def get(cls, draft_id: str) -> dict[str, Any] | None:
        cls.purge_expired()
        row = cls.collection.find_one({"draft_id": draft_id})
        return row.get("data") if row else None

    @classmethod
    def pop(cls, draft_id: str) -> dict[str, Any] | None:
        row = cls.collection.find_one_and_delete({"draft_id": draft_id})
        return row.get("data") if row else None


ScenarioDraftModel.ensure_indexes()
