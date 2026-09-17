"""Runtime paths and environment-backed application settings.

Static model artifacts are shipped under ``resources/``. Mutable runtime state is
created under ``RUNTIME_DATA_DIR`` and must not be committed to source control.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

RESOURCE_ROOT = ROOT / "resources"
TELECOM_STANDARDS_DIR = RESOURCE_ROOT / "telecom" / "standards" / "normalized"
TELECOM_PROFILES_DIR = RESOURCE_ROOT / "telecom" / "generation_profiles"

def resolve_path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


RUNTIME_DATA_DIR = resolve_path(os.getenv("RUNTIME_DATA_DIR"), ROOT / "runtime_data")
RUNTIME_DATA_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_DB_PATH = resolve_path(os.getenv("REGISTRY_DB_PATH"), RUNTIME_DATA_DIR / "registry.db")
DYNAMIC_SCENARIOS_DB = resolve_path(os.getenv("DYNAMIC_SCENARIOS_DB"), RUNTIME_DATA_DIR / "dynamic_scenarios.db")
CONVERSATION_DB_PATH = resolve_path(os.getenv("CONVERSATION_DB_PATH"), RUNTIME_DATA_DIR / "conversations.db")

# Public configuration should be explicit in production. Local development defaults
# are intentionally narrow and can be overridden with comma-separated origins.
CORS_ALLOW_ORIGINS = [
    item.strip() for item in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",") if item.strip()
]

MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024)))
