"""Central MongoDB connection management."""
from __future__ import annotations

from functools import lru_cache
import os
from pymongo import MongoClient
from pymongo.database import Database

DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "telco_scenario_db"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", DEFAULT_URI).strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is required")
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=_env_int("MONGODB_SERVER_SELECTION_TIMEOUT_MS", 5000),
        connectTimeoutMS=_env_int("MONGODB_CONNECT_TIMEOUT_MS", 5000),
        socketTimeoutMS=_env_int("MONGODB_SOCKET_TIMEOUT_MS", 10000),
        retryWrites=True,
        appname=os.getenv("MONGODB_APP_NAME", "telco-agentic-sdg"),
    )


@lru_cache(maxsize=1)
def get_database() -> Database:
    name = os.getenv("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME
    return get_client()[name]


def ping() -> None:
    get_client().admin.command("ping")


def close() -> None:
    get_client().close()
    get_database.cache_clear()
    get_client.cache_clear()
