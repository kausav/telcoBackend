"""Runtime paths and environment-backed application settings.

Official telecom standards are declared as a small source manifest under ``resources``.
Pinned artifacts are fetched into mutable ``runtime_data`` and normalized there; the
normalized runtime cache is never the source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

RESOURCE_ROOT = ROOT / "resources"
TELECOM_STANDARDS_MANIFEST = RESOURCE_ROOT / "telecom" / "standards" / "official_sources.json"
TELECOM_PROFILES_DIR = RESOURCE_ROOT / "telecom" / "generation_profiles"


def resolve_path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


RUNTIME_DATA_DIR = resolve_path(os.getenv("RUNTIME_DATA_DIR"), ROOT / "runtime_data")
RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mutable, generated runtime cache of official source artifacts and their internal index.
TELECOM_STANDARDS_CACHE_DIR = resolve_path(
    os.getenv("TELECOM_STANDARDS_CACHE_DIR"), RUNTIME_DATA_DIR / "official_standards"
)
TELECOM_STANDARDS_DIR = resolve_path(
    os.getenv("REGISTRY_STANDARDS_DIR"), TELECOM_STANDARDS_CACHE_DIR / "normalized"
)

REGISTRY_DB_PATH = resolve_path(os.getenv("REGISTRY_DB_PATH"), RUNTIME_DATA_DIR / "registry.db")
DYNAMIC_SCENARIOS_DB = resolve_path(os.getenv("DYNAMIC_SCENARIOS_DB"), RUNTIME_DATA_DIR / "dynamic_scenarios.db")
CONVERSATION_DB_PATH = resolve_path(os.getenv("CONVERSATION_DB_PATH"), RUNTIME_DATA_DIR / "conversations.db")

CORS_ALLOW_ORIGINS = [
    item.strip() for item in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",") if item.strip()
]

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024)))

# Official source sync behavior: auto downloads when a source is not already cached.
# Set OFFICIAL_STANDARDS_SYNC=disabled when an operator supplies REGISTRY_STANDARDS_DIR.
OFFICIAL_STANDARDS_SYNC = os.getenv("OFFICIAL_STANDARDS_SYNC", "auto").strip().lower()
OFFICIAL_STANDARDS_TIMEOUT_SEC = int(os.getenv("OFFICIAL_STANDARDS_TIMEOUT_SEC", "45"))
OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB = int(os.getenv("OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB", "100"))
