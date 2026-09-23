"""Production audit for the requested_scenario_id source-of-truth contract.

Run this against a live database before/after deployment to find legacy scenario-scoped
documents that lack the canonical requested_scenario_id. This script does not mutate data.
"""
from __future__ import annotations

from models.database import get_database
from models._helpers import collection_name

COLLECTIONS = (
    ("scenarios", "MONGODB_SCENARIOS_COLLECTION"),
    ("scenario_drafts", "MONGODB_DRAFTS_COLLECTION"),
    ("scenario_variables", "MONGODB_SCENARIO_VARIABLES_COLLECTION"),
    ("scenario_user_variables", "MONGODB_USER_VARIABLES_COLLECTION"),
    ("scenario_proposals", "MONGODB_PROPOSALS_COLLECTION"),
    ("conversations", "MONGODB_CONVERSATIONS_COLLECTION"),
    ("chat_messages", "MONGODB_CHAT_MESSAGES_COLLECTION"),
    ("scenario_feedback", "MONGODB_FEEDBACK_COLLECTION"),
)

def main() -> int:
    db = get_database()
    failures = 0
    for default_name, env_name in COLLECTIONS:
        name = collection_name(env_name, default_name)
        collection = db[name]
        # Scenario draft keeps the canonical id at the document level and inside data.
        if name.endswith("scenarios"):
            query = {
                "$or": [
                    {"requested_scenario_id": {"$exists": False}},
                    {"requested_scenario_id": None},
                    {"requested_scenario_id": ""},
                    {"meta.requested_scenario_id": {"$exists": False}},
                    {"meta.requested_scenario_id": None},
                    {"meta.requested_scenario_id": ""},
                ]
            }
        elif name.endswith("scenario_drafts"):
            query = {
                "$or": [
                    {"requested_scenario_id": {"$exists": False}},
                    {"requested_scenario_id": None},
                    {"requested_scenario_id": ""},
                    {"data.requested_scenario_id": {"$exists": False}},
                    {"data.requested_scenario_id": None},
                    {"data.requested_scenario_id": ""},
                ]
            }
        elif name.endswith("chat_messages"):
            # Chat messages are scenario-scoped in the application and therefore must carry
            # the canonical id directly; this makes them independently queryable/auditable.
            query = {
                "$or": [
                    {"requested_scenario_id": {"$exists": False}},
                    {"requested_scenario_id": None},
                    {"requested_scenario_id": ""},
                ]
            }
        else:
            query = {
                "$or": [
                    {"requested_scenario_id": {"$exists": False}},
                    {"requested_scenario_id": None},
                    {"requested_scenario_id": ""},
                ]
            }
            if name.endswith("scenario_proposals"):
                query["$or"].extend([
                    {"payload.requested_scenario_id": {"$exists": False}},
                    {"payload.requested_scenario_id": None},
                    {"payload.requested_scenario_id": ""},
                ])
        count = collection.count_documents(query)
        if count:
            print(f"{name}: {count} document(s) missing requested_scenario_id")
            failures += count
        else:
            print(f"{name}: OK")

        # Compatibility aliases are allowed for old clients, but they must never disagree
        # with the canonical requested_scenario_id on a document that already has one.
        alias_fields: list[str] = []
        if name.endswith("scenarios"):
            alias_fields = ["scenario_id", "meta.scenario_id", "meta.requested_scenario_id"]
        elif name.endswith("scenario_drafts"):
            alias_fields = ["data.scenario_id"]
        elif name.endswith("scenario_variables") or name.endswith("scenario_user_variables"):
            alias_fields = ["scenario_key"]
        elif name.endswith("scenario_proposals"):
            alias_fields = ["scenario_key", "payload.scenario_id", "payload.requested_scenario_id"]

        for alias_field in alias_fields:
            conflicts = collection.count_documents({
                "$expr": {"$and": [
                    {"$ne": [f"${alias_field}", None]},
                    {"$ne": [f"${alias_field}", "${requested_scenario_id}"]},
                    {"$ne": ["$requested_scenario_id", None]},
                ]}
            })
            if conflicts:
                print(f"{name}: {conflicts} document(s) with {alias_field} != requested_scenario_id")
                failures += conflicts
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
