"""Small shared helpers for MongoDB model modules."""
from __future__ import annotations

from config.constants import COLLECTIONS


def collection_name(key: str, default: str | None = None) -> str:
    """The MongoDB collection for ``key`` (see ``config.constants.COLLECTIONS``)."""
    return COLLECTIONS[key]
