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
        document = {"spec_key": spec_key, "spec": spec, "meta": meta, "created_at": time.time()}
        cls.collection.update_one({"spec_key": spec_key}, {"$setOnInsert": document}, upsert=True)
        return cls.get(spec_key) or document


GenerationSpecModel.ensure_indexes()
