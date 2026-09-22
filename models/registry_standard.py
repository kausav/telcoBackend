"""MongoDB model for telecom registry standards/artifacts."""
from __future__ import annotations
from models._helpers import collection_name
from models.database import get_database


class RegistryStandardModel:
    collection = get_database()[collection_name("MONGODB_REGISTRY_STANDARDS_COLLECTION", "registry_standards")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("artifact_id", unique=True)
        cls.collection.create_index("organization")


RegistryStandardModel.ensure_indexes()
