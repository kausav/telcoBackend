"""Environment-backed application settings.

Industry/domain standards are stored exclusively in MongoDB. This module contains
only generic application/runtime settings; it intentionally has no filesystem
standards directory or bundled industry-standard JSON configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def resolve_path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


RUNTIME_DATA_DIR = resolve_path(os.getenv("RUNTIME_DATA_DIR"), ROOT / "runtime_data")
try:
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    raise RuntimeError(
        f"Runtime data directory is not writable or could not be created: {RUNTIME_DATA_DIR}. "
        "Set RUNTIME_DATA_DIR to a writable directory for the service account."
    ) from exc

CORS_ALLOW_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if item.strip()
]

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024)))
INDUSTRY_SOURCE_MAX_JSON_BYTES = int(os.getenv("INDUSTRY_SOURCE_MAX_JSON_BYTES", str(8 * 1024 * 1024)))
JSON_SOURCE_LLM_CATALOG_LIMIT = int(os.getenv("JSON_SOURCE_LLM_CATALOG_LIMIT", "1200"))
# Number of representative fields shown per source business model to Gemini. The complete
# MongoDB catalog is still used by deterministic compilation; this only keeps propose latency low.
JSON_SOURCE_LLM_FIELDS_PER_MODEL = max(3, int(os.getenv("JSON_SOURCE_LLM_FIELDS_PER_MODEL", "6")))
# Process-local cache for immutable active source catalogs. Admin mutations invalidate the cache.
SOURCE_CATALOG_CACHE_TTL_SECONDS = max(5, int(os.getenv("SOURCE_CATALOG_CACHE_TTL_SECONDS", "60")))

SCHEMA_MAX_VARIABLES = int(os.getenv("SCHEMA_MAX_VARIABLES", "500"))
SCHEMA_MIN_VARIABLE_SCORE = float(os.getenv("SCHEMA_MIN_VARIABLE_SCORE", "42"))
