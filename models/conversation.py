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
        cls.collection.create_index([("user_id", 1), ("updated_at", -1)])

    @classmethod
    def ensure(cls, conversation_id: str | None = None, user_id: str | None = None) -> str:
        conversation_id = conversation_id or f"conv-{uuid.uuid4().hex}"
        now = time.time()
        cls.collection.update_one(
            {"conversation_id": conversation_id},
            {"$setOnInsert": {"conversation_id": conversation_id, "created_at": now},
             "$set": {"updated_at": now, "user_id": user_id}},
            upsert=True,
        )
        return conversation_id

    @classmethod
    def touch(cls, conversation_id: str, updated_at: float | None = None) -> None:
        cls.collection.update_one({"conversation_id": conversation_id}, {"$set": {"updated_at": updated_at or time.time()}})


ConversationModel.ensure_indexes()
