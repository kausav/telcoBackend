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
            [("scenario_key", 1), ("scenario_version", 1), ("variable_key", 1)],
            unique=True,
        )

    @classmethod
    def upsert_many(cls, scenario_key: str, scenario_version: int,
                    variables: list[dict[str, Any]], actor_user_id: str | None = None) -> int:
        from core.agentic_models import GeneratedSchemaField
        import time
        now = time.time()
        count = 0
        actor = actor_user_id or "system"
        for variable in variables:
            normalized = GeneratedSchemaField.model_validate(variable).model_dump()
            key = str(normalized.get("name") or "").strip()
            if not key:
                continue
            cls.collection.update_one(
                {"scenario_key": scenario_key, "scenario_version": int(scenario_version), "variable_key": key},
                {"$set": {
                    "display_name": key,
                    "definition": normalized,
                    "source": "DB_RECOMMENDED",
                    "enabled": True,
                    "updated_at": now,
                    "updated_by": actor,
                }, "$setOnInsert": {"created_at": now, "created_by": actor}},
                upsert=True,
            )
            count += 1
        return count

    @classmethod
    def get_enabled(cls, scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
        return [
            row["definition"]
            for row in cls.collection.find(
                {"scenario_key": scenario_key, "scenario_version": int(scenario_version), "enabled": True},
                {"definition": 1, "_id": 0},
            ).sort("variable_key", 1)
        ]


ScenarioVariableModel.ensure_indexes()
