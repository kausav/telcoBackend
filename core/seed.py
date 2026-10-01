"""Reference data the application ships with and keeps in MongoDB.

Behaviour packs (``behavior_packs``) and the domain vocabulary (``domain_lexicon``) are data, so they live in
MongoDB like the source documents and the scenario variables. A fresh deployment starts with neither, so the
bundled defaults in ``seed/`` are inserted the first time they are needed.

Only what is missing is inserted: a document that already exists (same pack id and version, same industry key)
is never overwritten, so edits made in MongoDB stay in force. Set ``SEED_REFERENCE_DATA=0`` to turn this off.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"
_SETS = (
    # file, collection env var, default collection, fields that identify a document
    ("behavior_packs.json", "MONGODB_BEHAVIOR_PACKS_COLLECTION", "behavior_packs", ("pack_id", "version")),
    ("domain_lexicon.json", "MONGODB_DOMAIN_LEXICON_COLLECTION", "domain_lexicon", ("industry_key",)),
)
_RETRY_SECONDS = 30.0
_LOCK = threading.Lock()
_state: dict[str, Any] = {"done": False, "next_try": 0.0}


def ensure_seeded() -> None:
    """Insert the bundled defaults that MongoDB lacks. Cheap after the first success; never raises."""
    if _state["done"] or os.getenv("SEED_REFERENCE_DATA", "1").strip().lower() in {"0", "false", "no"}:
        return
    with _LOCK:
        if _state["done"] or time.monotonic() < _state["next_try"]:
            return
        try:
            inserted = _seed()
        except Exception as exc:                       # the database is unreachable: try again shortly
            _state["next_try"] = time.monotonic() + _RETRY_SECONDS
            logger.warning("reference data not seeded: %s", exc)
            return
        _state["done"] = True
        if inserted:
            logger.info("Seeded MongoDB with bundled reference data: %s", inserted)


def _seed() -> dict[str, int]:
    from models._helpers import collection_name
    from models.database import get_database

    database = get_database()
    inserted: dict[str, int] = {}
    for filename, env_name, default, identity in _SETS:
        path = SEED_DIR / filename
        if not path.is_file():
            continue
        collection = database[collection_name(env_name, default)]
        count = 0
        for document in json.loads(path.read_text(encoding="utf-8")):
            key = {field: document[field] for field in identity}
            if collection.update_one(key, {"$setOnInsert": document}, upsert=True).upserted_id is not None:
                count += 1
        inserted[default] = count
    return {name: count for name, count in inserted.items() if count}

