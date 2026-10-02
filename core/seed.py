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
_state: dict[str, Any] = {"done": False, "next_try": 0.0, "error": None, "fingerprint": None}
_BUNDLED: dict[str, list[dict[str, Any]]] = {}
_bundled_for: list[Any] = [None]          # fingerprint of the seed files ``_BUNDLED`` was read from


def _fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Which version of the seed files is on disk, so a replaced file is noticed without restarting the server."""
    result = []
    for filename, *_rest in _SETS:
        path = SEED_DIR / filename
        stat = path.stat() if path.is_file() else None
        result.append((filename, stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0))
    return tuple(result)


def bundled(collection: str) -> list[dict[str, Any]]:
    """Documents the application ships for ``collection`` (the defaults MongoDB falls back to when it has none)."""
    fingerprint = _fingerprint()
    if _bundled_for[0] != fingerprint:
        _BUNDLED.clear()
        _bundled_for[0] = fingerprint
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
    """Insert the bundled defaults that MongoDB lacks. Cheap after the first success; never raises.

    Runs again when the seed files change on disk (a new bundle dropped in without a restart).
    """
    fingerprint = _fingerprint()
    if _state.get("fingerprint") not in (None, fingerprint):
        _state.update(done=False, next_try=0.0)
    if _state["done"] or os.getenv("SEED_REFERENCE_DATA", "1").strip().lower() in {"0", "false", "no"}:
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
        _state.update(done=True, error=None, fingerprint=fingerprint)
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
        collection = database[collection_name(env_name, default)]
        count = 0
        for document in json.loads(path.read_text(encoding="utf-8")):
            key = {field: document[field] for field in identity}
            try:
                if collection.update_one(key, {"$setOnInsert": {k: v for k, v in document.items() if k not in key}}, upsert=True).upserted_id is not None:
                    count += 1
            except Exception as exc:                   # one rejected document must not block the others
                failures.append(f"{default} {key}: {type(exc).__name__}: {exc}"[:200])
        inserted[default] = count
    if failures:
        raise RuntimeError("; ".join(failures[:3]) + (f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""))
    return {name: count for name, count in inserted.items() if count}

