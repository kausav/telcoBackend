"""Persistence model for conversation metadata."""
from __future__ import annotations

import time
import uuid
from models._helpers import collection_name
from models.database import get_database


class ConversationModel:
    collection = get_database()[collection_name("MONGODB_CONVERSATIONS_COLLECTION", "conversations")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("conversation_id", unique=True)
        cls.collection.create_index("requested_scenario_id")
        cls.collection.create_index([("user_id", 1), ("updated_at", -1)])

    @classmethod
    def ensure(cls, conversation_id: str | None = None, user_id: str | None = None,
               requested_scenario_id: str | None = None) -> str:
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        conversation_id = conversation_id or f"conv-{uuid.uuid4().hex}"
        now = time.time()
        existing = cls.collection.find_one(
            {"conversation_id": conversation_id},
            {"requested_scenario_id": 1, "_id": 0},
        )
        if existing:
            stored = str(existing.get("requested_scenario_id") or "").strip()
            if stored and stored != requested:
                raise ValueError(
                    f"conversation_id '{conversation_id}' belongs to requested_scenario_id '{stored}', not '{requested}'"
                )
        cls.collection.update_one(
            {"conversation_id": conversation_id},
            {"$setOnInsert": {"conversation_id": conversation_id, "created_at": now},
             "$set": {"updated_at": now, "user_id": user_id, "requested_scenario_id": requested}},
            upsert=True,
        )
        return conversation_id

    @classmethod
    def touch(cls, conversation_id: str, updated_at: float | None = None) -> None:
        cls.collection.update_one({"conversation_id": conversation_id}, {"$set": {"updated_at": updated_at or time.time()}})


ConversationModel.ensure_indexes()
