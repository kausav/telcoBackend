"""Application service for conversation persistence."""
from __future__ import annotations

from models.conversation import ConversationModel
from models.chat_message import ChatMessageModel


def ensure_conversation(conversation_id: str | None = None, user_id: str | None = None) -> str:
    return ConversationModel.ensure(conversation_id, user_id)


def append_message(conversation_id: str, role: str, content: str) -> None:
    ChatMessageModel.add(conversation_id, role, content)
    ConversationModel.touch(conversation_id)
