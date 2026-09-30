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
        from core.variable_semantics import variable_semantic_identities
        from core.low_balance_variable_policy import validate_db_definition
        import time
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        now = time.time()
        count = 0
        actor = actor_user_id or "system"
        incoming_by_alias: dict[str, str] = {}
        normalized_variables: list[dict[str, Any]] = []
        for variable in variables:
            normalized = GeneratedSchemaField.model_validate(variable).model_dump()
            validate_db_definition(normalized)
            key = str(normalized.get("name") or "").strip()
            if not key:
                continue
            aliases = variable_semantic_identities(normalized)
            conflict = sorted({incoming_by_alias[alias] for alias in aliases if alias in incoming_by_alias and incoming_by_alias[alias].casefold() != key.casefold()})
            if conflict:
                raise ValueError(
                    f"Scenario variables '{key}' semantically duplicate existing request variable(s): "
                    + ", ".join(conflict)
                )
            for alias in aliases:
                incoming_by_alias[alias] = key
            normalized_variables.append(normalized)

        # Recommendation means a selected subset. When a non-empty subset is supplied, deactivate
        # older enabled rows that are no longer selected so stale variables cannot reappear on the
        # next proposal/generation. Empty requests remain a no-op to avoid accidental clear-all calls.
        selected_keys = {str(item.get("name") or "").strip() for item in normalized_variables if str(item.get("name") or "").strip()}
        if selected_keys:
            cls.collection.update_many(
                {
                    "requested_scenario_id": requested,
                    "scenario_version": int(scenario_version),
                    "enabled": True,
                    "variable_key": {"$nin": sorted(selected_keys)},
                },
                {"$set": {"enabled": False, "updated_at": now, "updated_by": actor}},
            )

        for normalized in normalized_variables:
            key = str(normalized.get("name") or "").strip()
            cls.collection.update_one(
                {"requested_scenario_id": requested, "scenario_version": int(scenario_version), "variable_key": key},
                {"$set": {
                    "requested_scenario_id": requested,
                    "scenario_key": requested,
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
    def get_enabled(cls, requested_scenario_id: str, scenario_version: int) -> list[dict[str, Any]]:
        return [
            row["definition"]
            for row in cls.collection.find(
                {"requested_scenario_id": requested_scenario_id, "scenario_version": int(scenario_version), "enabled": True},
                {"definition": 1, "_id": 0},
            ).sort("variable_key", 1)
        ]


ScenarioVariableModel.ensure_indexes()
