"""Persistence model for verified generation specs (one document per compiled scenario behaviour)."""
from __future__ import annotations

import time
from typing import Any

from models._helpers import collection_name
from models.database import get_database


class GenerationSpecModel:
    collection = get_database()[collection_name("MONGODB_GENERATION_SPECS_COLLECTION", "generation_specs")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("spec_key", unique=True)

    @classmethod
    def get(cls, spec_key: str) -> dict[str, Any] | None:
        return cls.collection.find_one({"spec_key": spec_key}, {"_id": 0})

    @classmethod
    def put_if_absent(cls, spec_key: str, spec: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        """Store a spec once; returns the stored document (an existing one wins, so a key never changes meaning)."""
        document = {"spec_key": spec_key, "spec": spec, "meta": meta, "revision": 0, "created_at": time.time()}
        cls.collection.update_one({"spec_key": spec_key}, {"$setOnInsert": document}, upsert=True)
        return cls.get(spec_key) or document

    @classmethod
    def put_revision(cls, spec_key: str, spec: dict[str, Any], meta: dict[str, Any], revision: int) -> bool:
        """Replace a stored spec by a later revision of it (a refinement); an equal or newer stored revision is left alone."""
        result = cls.collection.update_one(
            {"spec_key": spec_key, "$or": [{"revision": {"$lt": revision}}, {"revision": {"$exists": False}}]},
            {"$set": {"spec": spec, "meta": meta, "revision": revision, "updated_at": time.time()}},
        )
        return bool(result.modified_count)

    @classmethod
    def set_review(cls, spec_key: str, warnings: list[str], reviewed: bool) -> None:
        """Record the outcome of a review on the stored spec without changing its behaviour (or its revision)."""
        cls.collection.update_one({"spec_key": spec_key}, {"$set": {"spec.warnings": warnings, "spec.reviewed": reviewed}})


class GenerationSpecClaimModel:
    """Who is designing (or refining) a spec right now, across every worker process: one short-lived document per key."""

    collection = get_database()[collection_name("MONGODB_GENERATION_SPEC_CLAIMS_COLLECTION", "generation_spec_claims")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("spec_key", unique=True)

    @classmethod
    def claim(cls, spec_key: str, owner: str, ttl_seconds: float) -> bool:
        """True when ``owner`` now holds the claim; False when another live claim exists. An expired claim is taken over."""
        from pymongo.errors import DuplicateKeyError

        now = time.time()
        cls.collection.delete_one({"spec_key": spec_key, "expires_at": {"$lt": now}})
        try:
            cls.collection.insert_one({"spec_key": spec_key, "owner": owner, "started_at": now, "expires_at": now + ttl_seconds})
            return True
        except DuplicateKeyError:
            return False

    @classmethod
    def active(cls, spec_key: str) -> bool:
        return cls.collection.find_one({"spec_key": spec_key, "expires_at": {"$gte": time.time()}}, {"_id": 1}) is not None

    @classmethod
    def release(cls, spec_key: str, owner: str) -> None:
        cls.collection.delete_one({"spec_key": spec_key, "owner": owner})


GenerationSpecModel.ensure_indexes()
GenerationSpecClaimModel.ensure_indexes()
