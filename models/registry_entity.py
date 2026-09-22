"""MongoDB model for telecom registry entities."""
from __future__ import annotations
from models._helpers import collection_name
from models.database import get_database


class RegistryEntityModel:
    collection = get_database()[collection_name("MONGODB_REGISTRY_ENTITIES_COLLECTION", "registry_entities")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("canonical_id", unique=True)
        cls.collection.create_index("aliases")
        cls.collection.create_index("domain")


RegistryEntityModel.ensure_indexes()
