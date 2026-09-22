"""Small shared helpers for MongoDB model modules."""
from __future__ import annotations

import os
from typing import Any


def collection_name(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).strip() or default


def without_id(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}
