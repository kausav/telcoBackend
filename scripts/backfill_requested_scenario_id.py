"""Backfill requested_scenario_id for legacy scenario-scoped MongoDB documents.

Usage:
    python -m scripts.backfill_requested_scenario_id          # dry run
    python -m scripts.backfill_requested_scenario_id --apply  # mutate

The migration copies the legacy scenario identifier into requested_scenario_id. It never
changes an existing requested_scenario_id and refuses to overwrite a conflicting value.
Run this before enabling the strict requested_scenario_id indexes against a legacy database.
"""
from __future__ import annotations

import argparse
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

def _name(default_name: str, env_name: str) -> str:
    return collection_name(env_name, default_name)

def _legacy_value(doc: dict, collection: str) -> str | None:
    if collection.endswith("scenario_drafts"):
        data = doc.get("data") or {}
        return str(data.get("requested_scenario_id") or data.get("scenario_id") or "").strip() or None
    if collection.endswith("conversations"):
        return str(doc.get("requested_scenario_id") or "").strip() or None
    if collection.endswith("chat_messages"):
        return None
    if collection.endswith("scenario_feedback"):
        return None
    return str(
        doc.get("requested_scenario_id")
        or doc.get("scenario_key")
        or doc.get("scenario_id")
        or (doc.get("payload") or {}).get("requested_scenario_id")
        or (doc.get("payload") or {}).get("scenario_id")
        or ""
    ).strip() or None


def _scenario_lookup(db, conversation_id: str) -> str | None:
    """Recover a conversation's scenario only when its conversation id maps unambiguously."""
    if not conversation_id:
        return None
    collection = db[_name("scenarios", "MONGODB_SCENARIOS_COLLECTION")]
    matches = list(collection.find(
        {"$or": [
            {"requested_scenario_id": conversation_id},
            {"scenario_id": conversation_id},
        ]},
        {"requested_scenario_id": 1, "_id": 0},
    ).limit(2))
    if len(matches) != 1:
        return None
    return str(matches[0].get("requested_scenario_id") or "").strip() or None

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the backfill; default is dry-run")
    args = parser.parse_args()

    db = get_database()
    failures = 0
    updates = 0
    for default_name, env_name in COLLECTIONS:
        collection = db[_name(default_name, env_name)]
        for doc in collection.find({"$or": [{"requested_scenario_id": {"$exists": False}}, {"requested_scenario_id": None}]}):
            value = _legacy_value(doc, collection.name)
            if not value:
                if collection.name.endswith("conversations"):
                    value = _scenario_lookup(db, str(doc.get("conversation_id") or "").strip())
                if collection.name.endswith("chat_messages"):
                    conversation = db[_name("conversations", "MONGODB_CONVERSATIONS_COLLECTION")].find_one(
                        {"conversation_id": doc.get("conversation_id")},
                        {"requested_scenario_id": 1, "_id": 0},
                    )
                    value = str((conversation or {}).get("requested_scenario_id") or "").strip() or None
                elif collection.name.endswith("scenario_feedback"):
                    matches = list(db[_name("scenarios", "MONGODB_SCENARIOS_COLLECTION")].find(
                        {"meta.domain": doc.get("domain"), "meta.business_scenario": doc.get("business_scenario")},
                        {"requested_scenario_id": 1, "_id": 0},
                    ).limit(2))
                    if len(matches) == 1:
                        value = str(matches[0].get("requested_scenario_id") or "").strip() or None
                if not value:
                    print(f"{collection.name}: {_id(doc)} has no unambiguous recoverable scenario identifier")
                    failures += 1
                    continue
            if collection.name.endswith("scenario_drafts"):
                data = dict(doc.get("data") or {})
                existing = str(data.get("requested_scenario_id") or "").strip() or None
                if existing and existing != value:
                    print(f"{collection.name}: {_id(doc)} conflicting requested_scenario_id")
                    failures += 1
                    continue
                if args.apply:
                    collection.update_one({"_id": doc["_id"]}, {"$set": {
                        "requested_scenario_id": value,
                        "data.requested_scenario_id": value,
                        "data.scenario_id": value,
                    }})
                updates += 1
                continue

            if args.apply:
                set_doc = {"requested_scenario_id": value}
                if collection.name.endswith("scenarios"):
                    set_doc.update({"scenario_id": value, "meta.requested_scenario_id": value, "meta.scenario_id": value})
                elif collection.name.endswith("scenario_variables") or collection.name.endswith("scenario_user_variables"):
                    set_doc["scenario_key"] = value
                elif collection.name.endswith("scenario_proposals"):
                    set_doc.update({
                        "scenario_key": value,
                        "payload.requested_scenario_id": value,
                        "payload.scenario_id": value,
                    })
                collection.update_one({"_id": doc["_id"]}, {"$set": set_doc})
            updates += 1

    print(("Applied" if args.apply else "Would apply") + f" {updates} requested_scenario_id backfill(s)")
    if failures:
        print(f"Encountered {failures} migration issue(s); no automatic destructive changes were made.")
        return 1
    return 0

def _id(doc: dict) -> str:
    return str(doc.get("_id"))

if __name__ == "__main__":
    raise SystemExit(main())
