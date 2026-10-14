"""Reference data the application ships with and keeps in MongoDB.

The domain vocabulary and the scenario catalog live in MongoDB alongside source documents and scenario variables. A fresh
deployment starts without them, so the bundled data in ``seed/`` is inserted the first time it is needed.

Only missing entries are inserted; existing MongoDB data is never overwritten. Set ``SEED_REFERENCE_DATA=0``
to turn this off.
"""
from __future__ import annotations

from config import constants as _constants

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
    # file, key in config.constants.COLLECTIONS, default collection name, fields that identify a document
    ("domain_lexicon.json", "domain_lexicon", "domain_lexicon", ("industry_key",)),
    ("country_limits.json", "country_limits", "country_operating_limits", ("country",)),
)
_RETRY_SECONDS = 30.0
_LOCK = threading.Lock()
_state: dict[str, Any] = {"done": False, "next_try": 0.0, "error": None}
_BUNDLED: dict[str, list[dict[str, Any]]] = {}


def bundled(collection: str) -> list[dict[str, Any]]:
    """Documents the application ships for ``collection`` (the defaults MongoDB falls back to when it has none)."""
    if collection not in _BUNDLED:
        documents: list[dict[str, Any]] = []
        for filename, _env, default, _identity in _SETS:
            if default == collection and (SEED_DIR / filename).is_file():
                documents = json.loads((SEED_DIR / filename).read_text(encoding="utf-8"))
        _BUNDLED[collection] = documents
    return _BUNDLED[collection]


def status() -> dict[str, Any]:
    """Whether the bundled reference data reached MongoDB, and the last reason it did not."""
    return {"seeded": bool(_state["done"]), "last_error": _state["error"]}


def ensure_seeded() -> None:
    """Insert the bundled defaults that MongoDB lacks. Cheap after the first success; never raises."""
    if _state["done"] or not _constants.SEED_REFERENCE_DATA:
        return
    with _LOCK:
        if _state["done"] or time.monotonic() < _state["next_try"]:
            return
        try:
            inserted = _seed()
        except Exception as exc:                       # the database is unreachable: try again shortly
            _state["next_try"] = time.monotonic() + _RETRY_SECONDS
            _state["error"] = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("reference data not seeded: %s", _state["error"])
            return
        _state["done"] = True
        if inserted:
            logger.info("Seeded MongoDB with bundled reference data: %s", inserted)


def _seed() -> dict[str, int]:
    from models._helpers import collection_name
    from models.database import get_database

    database = get_database()
    inserted: dict[str, int] = {}
    failures: list[str] = []
    for filename, env_name, default, identity in _SETS:
        path = SEED_DIR / filename
        if not path.is_file():
            continue
        collection = database[collection_name(env_name)]
        count = 0
        for document in json.loads(path.read_text(encoding="utf-8")):
            key = {field: document[field] for field in identity}
            try:
                if collection.update_one(key, {"$setOnInsert": {k: v for k, v in document.items() if k not in key}}, upsert=True).upserted_id is not None:
                    count += 1
            except Exception as exc:                   # one rejected document must not block the others
                failures.append(f"{default} {key}: {type(exc).__name__}: {exc}"[:200])
        inserted[default] = count
    try:
        from core.catalog_store import seed_bundled

        inserted.update(seed_bundled())
    except Exception as exc:                           # the catalog must not block the vocabulary, and is retried with it
        failures.append(f"catalog: {type(exc).__name__}: {exc}"[:200])
    if failures:
        raise RuntimeError("; ".join(failures[:3]) + (f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""))
    return {name: count for name, count in inserted.items() if count}

