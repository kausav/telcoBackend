"""Persistence model for conversation messages."""
from __future__ import annotations

import time
from models._helpers import collection_name
from models.database import get_database


class ChatMessageModel:
    collection = get_database()[collection_name("MONGODB_CHAT_MESSAGES_COLLECTION", "chat_messages")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index([("conversation_id", 1), ("created_at", 1)])

    @classmethod
    def add(cls, conversation_id: str, role: str, content: str) -> None:
        cls.collection.insert_one({"conversation_id": conversation_id, "role": role, "content": content, "created_at": time.time()})


ChatMessageModel.ensure_indexes()
