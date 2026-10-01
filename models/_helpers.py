"""Small shared helpers for MongoDB model modules."""
from __future__ import annotations

import os


def collection_name(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).strip() or default


