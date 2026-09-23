"""Persistence model for scenario-level recommended variables."""
from __future__ import annotations

from typing import Any
from models._helpers import collection_name
from models.database import get_database


class ScenarioVariableModel:
    collection = get_database()[collection_name("MONGODB_SCENARIO_VARIABLES_COLLECTION", "scenario_variables")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index(
            [("requested_scenario_id", 1), ("scenario_version", 1), ("variable_key", 1)],
            unique=True,
        )
        cls.collection.create_index([("scenario_key", 1), ("scenario_version", 1), ("variable_key", 1)])

    @classmethod
    def upsert_many(cls, requested_scenario_id: str, scenario_version: int,
                    variables: list[dict[str, Any]], actor_user_id: str | None = None) -> int:
        from core.agentic_models import GeneratedSchemaField
        import time
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        now = time.time()
        count = 0
        actor = actor_user_id or "system"
        for variable in variables:
            normalized = GeneratedSchemaField.model_validate(variable).model_dump()
            key = str(normalized.get("name") or "").strip()
            if not key:
                continue

            # Canonicalize the variable key so the same scenario/version/variable
            # is always updated instead of creating a duplicate due to casing.
            canonical_key = key.lower()
            version = int(scenario_version)
            base_filter = {
                "requested_scenario_id": requested,
                "scenario_version": version,
            }

            # Prefer the canonical key. If an older record was stored with a
            # different casing, reuse that document instead of inserting another.
            existing = cls.collection.find_one({
                **base_filter,
                "variable_key": canonical_key,
            }, {"_id": 1})
            if existing is None:
                import re
                existing = cls.collection.find_one({
                    **base_filter,
                    "variable_key": {"$regex": f"^{re.escape(key)}$", "$options": "i"},
                }, {"_id": 1})

            update = {
                "$set": {
                    "requested_scenario_id": requested,
                    "scenario_key": requested,
                    "variable_key": canonical_key,
                    "display_name": key,
                    "definition": normalized,
                    "source": "DB_RECOMMENDED",
                    "enabled": True,
                    "updated_at": now,
                    "updated_by": actor,
                },
                "$setOnInsert": {"created_at": now, "created_by": actor},
            }

            if existing and existing.get("_id") is not None:
                cls.collection.update_one({"_id": existing["_id"]}, update, upsert=False)
            else:
                cls.collection.update_one(
                    {**base_filter, "variable_key": canonical_key},
                    update,
                    upsert=True,
                )
            count += 1
        return count

    @classmethod
    def get_enabled(cls, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
        return [
            row["definition"]
            for row in cls.collection.find(
                {"requested_scenario_id": requested_scenario_id, "scenario_version": int(scenario_version), "enabled": True},
                {"definition": 1, "_id": 0},
            ).sort("variable_key", 1)
        ]


ScenarioVariableModel.ensure_indexes()
