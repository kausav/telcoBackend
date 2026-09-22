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
        cls.collection.create_index([("domain", 1), ("business_scenario", 1)])

    @classmethod
    def add(cls, domain: str, business_scenario: str, feedback: str) -> None:
        if not feedback:
            return
        cls.collection.insert_one({
            "key": f"{domain.strip().lower()}::{business_scenario.strip().lower()}",
            "domain": domain,
            "business_scenario": business_scenario,
            "feedback": feedback,
            "created_at": time.time(),
        })


ScenarioFeedbackModel.ensure_indexes()
