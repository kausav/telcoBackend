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
try:
    RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    raise RuntimeError(
        f"Runtime data directory is not writable or could not be created: {RUNTIME_DATA_DIR}. "
        "Set RUNTIME_DATA_DIR to a writable directory for the service account."
    ) from exc

# Mutable, generated runtime cache of official source artifacts and their internal index.
TELECOM_STANDARDS_CACHE_DIR = resolve_path(
    os.getenv("TELECOM_STANDARDS_CACHE_DIR"), RUNTIME_DATA_DIR / "official_standards"
)
TELECOM_STANDARDS_DIR = resolve_path(
    os.getenv("REGISTRY_STANDARDS_DIR"), TELECOM_STANDARDS_CACHE_DIR / "normalized"
)

CORS_ALLOW_ORIGINS = [
    item.strip() for item in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",") if item.strip()
]

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024)))

# Schema breadth is quality-gated rather than "include everything". The maximum is a
# preference: mandatory application/identity contracts are never dropped for size.
SCHEMA_MAX_VARIABLES = int(os.getenv("SCHEMA_MAX_VARIABLES", "100"))
SCHEMA_MIN_VARIABLE_SCORE = float(os.getenv("SCHEMA_MIN_VARIABLE_SCORE", "42"))

# Generation is a durable MongoDB-backed queue. API instances only enqueue work; dedicated
# worker processes claim jobs and execute the complete validation pipeline.
GENERATION_JOB_TTL_SECONDS = int(os.getenv("GENERATION_JOB_TTL_SECONDS", "86400"))
GENERATION_JOB_LEASE_SECONDS = int(os.getenv("GENERATION_JOB_LEASE_SECONDS", "900"))
GENERATION_WORKER_POLL_SECONDS = float(os.getenv("GENERATION_WORKER_POLL_SECONDS", "2"))
GENERATION_MAX_ATTEMPTS = int(os.getenv("GENERATION_MAX_ATTEMPTS", "3"))

# Official source sync behavior: auto downloads when a source is not already cached.
# Set OFFICIAL_STANDARDS_SYNC=disabled when an operator supplies REGISTRY_STANDARDS_DIR.
OFFICIAL_STANDARDS_SYNC = os.getenv("OFFICIAL_STANDARDS_SYNC", "auto").strip().lower()
OFFICIAL_STANDARDS_TIMEOUT_SEC = int(os.getenv("OFFICIAL_STANDARDS_TIMEOUT_SEC", "45"))
OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB = int(os.getenv("OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB", "100"))
