"""Persistence model for per-user scenario variable selections."""
from __future__ import annotations

import time
from typing import Any
from models._helpers import collection_name
from models.database import get_database


class ScenarioUserVariableModel:
    collection = get_database()[collection_name("MONGODB_USER_VARIABLES_COLLECTION", "scenario_user_variables")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index(
            [("user_id", 1), ("scenario_key", 1), ("scenario_version", 1), ("variable_key", 1)],
            unique=True,
        )
        cls.collection.create_index([("user_id", 1), ("scenario_key", 1), ("selection_state", 1)])

    @classmethod
    def upsert_many(cls, user_id: str, scenario_key: str, scenario_version: int,
                    variables: list[dict[str, Any]], state: str = "SELECTED") -> int:
        from core.agentic_models import GeneratedSchemaField
        if not user_id.strip():
            raise ValueError("user_id is required")
        if state not in {"SELECTED", "DESELECTED", "OVERRIDDEN"}:
            raise ValueError(f"Unsupported user variable state: {state}")
        now = time.time()
        count = 0
        for variable in variables:
            normalized = GeneratedSchemaField.model_validate(variable).model_dump()
            key = str(normalized.get("name") or "").strip()
            if not key:
                continue
            cls.collection.update_one(
                {"user_id": user_id, "scenario_key": scenario_key, "scenario_version": int(scenario_version), "variable_key": key},
                {"$set": {
                    "definition": normalized,
                    "selection_state": state,
                    "selection_source": "USER",
                    "is_active": state in {"SELECTED", "OVERRIDDEN"},
                    "updated_at": now,
                    "updated_by": user_id,
                    "deleted_at": None,
                }, "$setOnInsert": {"created_at": now, "selected_at": now}},
                upsert=True,
            )
            count += 1
        return count

    @classmethod
    def get_active(cls, user_id: str, scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
        return [
            row["definition"]
            for row in cls.collection.find(
                {"user_id": user_id, "scenario_key": scenario_key, "scenario_version": int(scenario_version),
                 "selection_state": {"$in": ["SELECTED", "OVERRIDDEN"]}, "is_active": True},
                {"definition": 1, "_id": 0},
            ).sort("variable_key", 1)
        ]

    @classmethod
    def delete(cls, user_id: str, scenario_key: str, scenario_version: int, variable_key: str) -> bool:
        now = time.time()
        result = cls.collection.update_one(
            {"user_id": user_id, "scenario_key": scenario_key, "scenario_version": int(scenario_version), "variable_key": variable_key},
            {"$set": {"selection_state": "DELETED", "is_active": False, "deleted_at": now, "updated_at": now, "updated_by": user_id}},
        )
        return result.matched_count > 0

    @classmethod
    def get_records(cls, user_id: str, scenario_key: str, scenario_version: int) -> list[dict[str, Any]]:
        return [{key: value for key, value in row.items() if key != "_id"} for row in cls.collection.find(
            {"user_id": user_id, "scenario_key": scenario_key, "scenario_version": int(scenario_version)}
        ).sort("variable_key", 1)]


ScenarioUserVariableModel.ensure_indexes()
