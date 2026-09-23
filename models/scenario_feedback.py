"""Persistence model for scenario feedback."""
from __future__ import annotations

import time
from models._helpers import collection_name
from models.database import get_database


class ScenarioFeedbackModel:
    collection = get_database()[collection_name("MONGODB_FEEDBACK_COLLECTION", "scenario_feedback")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("key")
        cls.collection.create_index("requested_scenario_id")
        cls.collection.create_index([("requested_scenario_id", 1), ("created_at", -1)])
        cls.collection.create_index([("domain", 1), ("business_scenario", 1)])

    @classmethod
    def add(cls, requested_scenario_id: str, domain: str, business_scenario: str, feedback: str) -> None:
        requested = str(requested_scenario_id or "").strip()
        if not requested:
            raise ValueError("requested_scenario_id is required")
        if not feedback:
            return
        cls.collection.insert_one({
            "key": f"{requested}::{domain.strip().lower()}::{business_scenario.strip().lower()}",
            "requested_scenario_id": requested,
            "domain": domain,
            "business_scenario": business_scenario,
            "feedback": feedback,
            "created_at": time.time(),
        })


ScenarioFeedbackModel.ensure_indexes()
