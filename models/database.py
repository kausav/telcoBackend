"""Central MongoDB connection management."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

ROOT = Path(__file__).resolve().parents[1]

# Load the project .env here as well as from config.runtime. This module is imported
# through models.scenario during FastAPI application import, which can happen before
# config.runtime is imported. Loading here prevents an accidental localhost fallback.
load_dotenv(ROOT / ".env", override=False)

from config import constants as _c

DEFAULT_URI = _c.MONGODB_DEFAULT_URI
DEFAULT_DB_NAME = _c.MONGODB_DEFAULT_DB_NAME


def _normalize_mongodb_uri(uri: str) -> str:
    """Accept the project's existing escaped dotenv URI without requiring .env edits.

    The currently documented local .env may contain escaped colon and at-sign
    characters in the connection string. python-dotenv preserves those backslashes,
    but MongoDB's
    URI parser expects the literal URI delimiters. Normalize only these two
    escaped delimiters and leave all other characters untouched.
    """
    return uri.replace(r"\:", ":").replace(r"\@", "@")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = _normalize_mongodb_uri(os.getenv("MONGODB_URI", DEFAULT_URI).strip())
    if not uri:
        raise RuntimeError("MONGODB_URI is required")
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=_c.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=_c.MONGODB_CONNECT_TIMEOUT_MS,
        socketTimeoutMS=_c.MONGODB_SOCKET_TIMEOUT_MS,
        retryWrites=True,
        appname=_c.MONGODB_APP_NAME,
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
