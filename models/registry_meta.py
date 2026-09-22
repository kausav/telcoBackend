"""MongoDB model for telecom registry metadata."""
from __future__ import annotations
from models._helpers import collection_name
from models.database import get_database


class RegistryMetaModel:
    collection = get_database()[collection_name("MONGODB_REGISTRY_META_COLLECTION", "registry_meta")]

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("key", unique=True)


RegistryMetaModel.ensure_indexes()
