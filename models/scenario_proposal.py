"""Persistence model for generated scenario proposals."""
from __future__ import annotations

import time
from typing import Any
from models._helpers import collection_name
from models.database import get_database


class ScenarioProposalModel:
    collection = get_database()[collection_name("MONGODB_PROPOSALS_COLLECTION", "scenario_proposals")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("request_id", unique=True)
        cls.collection.create_index([("requested_scenario_id", 1), ("created_at", -1)])
        cls.collection.create_index([("user_id", 1), ("requested_scenario_id", 1), ("created_at", -1)])

    @classmethod
    def save(cls, request_id: str, user_id: str | None, requested_scenario_id: str,
             scenario_version: int, payload: dict[str, Any]) -> None:
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        cls.collection.replace_one(
            {"request_id": request_id},
            {"request_id": request_id, "user_id": user_id, "requested_scenario_id": requested,
             "scenario_key": requested, "scenario_version": int(scenario_version),
             "payload": {**payload, "requested_scenario_id": requested, "scenario_id": requested}, "created_at": time.time()},
            upsert=True,
        )


ScenarioProposalModel.ensure_indexes()
