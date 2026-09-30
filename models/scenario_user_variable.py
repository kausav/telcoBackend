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
            [("user_id", 1), ("requested_scenario_id", 1), ("scenario_version", 1), ("variable_key", 1)],
            unique=True,
        )
        cls.collection.create_index([("user_id", 1), ("requested_scenario_id", 1), ("selection_state", 1)])

    @classmethod
    def upsert_many(cls, user_id: str, requested_scenario_id: str, scenario_version: int,
                    variables: list[dict[str, Any]], state: str = "SELECTED") -> int:
        from core.agentic_models import GeneratedSchemaField
        from core.variable_semantics import variable_semantic_identities
        from core.low_balance_variable_policy import validate_db_definition
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        if not user_id.strip():
            raise ValueError("user_id is required")
        if state not in {"SELECTED", "DESELECTED", "OVERRIDDEN"}:
            raise ValueError(f"Unsupported user variable state: {state}")
        now = time.time()
        count = 0
        incoming_by_alias: dict[str, str] = {}
        normalized_variables: list[dict[str, Any]] = []
        for variable in variables:
            normalized = GeneratedSchemaField.model_validate(variable).model_dump()
            validate_db_definition(normalized)
            key = str(normalized.get("name") or "").strip()
            if not key:
                continue
            aliases = variable_semantic_identities(normalized)
            conflicts = sorted({incoming_by_alias[alias] for alias in aliases if alias in incoming_by_alias and incoming_by_alias[alias].casefold() != key.casefold()})
            if conflicts:
                raise ValueError(
                    f"User variables '{key}' semantically duplicate existing selection(s): "
                    + ", ".join(conflicts)
                )
            for alias in aliases:
                incoming_by_alias[alias] = key
            normalized_variables.append(normalized)

        selected_keys = {str(item.get("name") or "").strip() for item in normalized_variables if str(item.get("name") or "").strip()}
        if selected_keys:
            cls.collection.update_many(
                {
                    "user_id": user_id, "requested_scenario_id": requested, "scenario_version": int(scenario_version),
                    "selection_state": {"$in": ["SELECTED", "OVERRIDDEN"]}, "is_active": True,
                    "variable_key": {"$nin": sorted(selected_keys)},
                },
                {"$set": {"selection_state": "DESELECTED", "is_active": False, "updated_at": now, "updated_by": user_id}},
            )

        for normalized in normalized_variables:
            key = str(normalized.get("name") or "").strip()
            cls.collection.update_one(
                {"user_id": user_id, "requested_scenario_id": requested, "scenario_version": int(scenario_version), "variable_key": key},
                {"$set": {
                    "requested_scenario_id": requested,
                    "scenario_key": requested,
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
    def get_active(cls, user_id: str, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
        return [
            row["definition"]
            for row in cls.collection.find(
                {"user_id": user_id, "requested_scenario_id": requested_scenario_id, "scenario_version": int(scenario_version),
                 "selection_state": {"$in": ["SELECTED", "OVERRIDDEN"]}, "is_active": True},
                {"definition": 1, "_id": 0},
            ).sort("variable_key", 1)
        ]

    @classmethod
    def delete(cls, user_id: str, requested_scenario_id: str, scenario_version: int, variable_key: str) -> bool:
        now = time.time()
        result = cls.collection.update_one(
            {"user_id": user_id, "requested_scenario_id": requested_scenario_id, "scenario_version": int(scenario_version), "variable_key": variable_key},
            {"$set": {"selection_state": "DELETED", "is_active": False, "deleted_at": now, "updated_at": now, "updated_by": user_id}},
        )
        return result.matched_count > 0

    @classmethod
    def get_records(cls, user_id: str, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
        return [{key: value for key, value in row.items() if key != "_id"} for row in cls.collection.find(
            {"user_id": user_id, "requested_scenario_id": requested_scenario_id, "scenario_version": int(scenario_version)}
        ).sort("variable_key", 1)]


ScenarioUserVariableModel.ensure_indexes()
