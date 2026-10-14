"""Persistence for the operating limits of a country's payment rails (one document per country).

A document looks like::

    {"country": "IN", "currency": "INR", "cash_denomination": 100, "note": "...",
     "rails": [{"id": "upi", "label": "UPI", "aliases": ["upi"], "kind": "instant_transfer",
                "min_amount": null, "max_amount": 100000, "fee": 0, "note": "..."}]}

``aliases`` are the words by which a column value names the rail (matched as whole words, case-insensitive). ``min_amount`` /
``max_amount`` are the floor and cap of one transaction on that rail, ``fee`` a fixed fee (0: the rail is free), in the document's currency.
"""
from __future__ import annotations

import time
from typing import Any

from models._helpers import collection_name
from models.database import get_database


class CountryLimitModel:
    collection = get_database()[collection_name("country_limits")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("country", unique=True)

    @classmethod
    def get(cls, country: str) -> dict[str, Any] | None:
        return cls.collection.find_one({"country": country}, {"_id": 0})

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        return list(cls.collection.find({}, {"_id": 0}).sort("country", 1))

    @classmethod
    def put(cls, country: str, document: dict[str, Any]) -> dict[str, Any]:
        """Create or replace the document of ``country``."""
        stored = {**document, "country": country, "updated_at": time.time()}
        cls.collection.replace_one({"country": country}, stored, upsert=True)
        return cls.get(country) or stored

    @classmethod
    def delete(cls, country: str) -> bool:
        return bool(cls.collection.delete_one({"country": country}).deleted_count)
