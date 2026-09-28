"""MongoDB model for industry/domain standard JSON source documents."""
from __future__ import annotations

from models._helpers import collection_name
from models.database import get_database


class IndustrySourceModel:
    collection = get_database()[
        collection_name("MONGODB_INDUSTRY_SOURCES_COLLECTION", "industry_source_documents")
    ]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index(
            [("industry_key", 1), ("domain_key", 1), ("sha256", 1)],
            unique=True,
            name="uq_industry_domain_sha256",
        )
        cls.collection.create_index(
            [("source_id", 1)],
            unique=True,
            name="uq_source_id",
        )
        cls.collection.create_index(
            [("industry_key", 1), ("domain_key", 1), ("active", 1), ("updated_at", -1)],
            name="ix_industry_domain_active_updated",
        )
        cls.collection.create_index(
            [("source_name", 1), ("version", 1)],
            name="ix_source_name_version",
        )

