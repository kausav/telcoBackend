#!/usr/bin/env python3
"""Synchronize official telecom models and rebuild the runtime registry.

Examples:
  python scripts/ingest_standards.py
  python scripts/ingest_standards.py --force
  python scripts/ingest_standards.py --openapi path/to/custom.swagger.json \
      --source-url https://... --organization "TM Forum"

Normal operation consumes the pinned official model sources in
resources/telecom/standards/official_sources.json. Custom OpenAPI ingestion remains
available for controlled/offline extensions, but it is not part of the official-source
bootstrap path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.runtime import (  # noqa: E402
    TELECOM_STANDARDS_MANIFEST,
    TELECOM_STANDARDS_CACHE_DIR,
    TELECOM_PROFILES_DIR,
    REGISTRY_DB_PATH,
    OFFICIAL_STANDARDS_TIMEOUT_SEC,
    OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB,
)
from core.official_standards import sync_official_standards  # noqa: E402
from core.standards_ingestion import normalize_openapi_file  # noqa: E402
from core.telecom_registry import RegistryBuilder, registry_fingerprint  # noqa: E402
from core.runtime_lock import RuntimeFileLock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync official telecom model artifacts and build the INGENII registry")
    parser.add_argument("--manifest", default=str(TELECOM_STANDARDS_MANIFEST))
    parser.add_argument("--cache-dir", default=str(TELECOM_STANDARDS_CACHE_DIR))
    parser.add_argument("--profiles-dir", default=str(TELECOM_PROFILES_DIR))
    parser.add_argument("--db", default=str(REGISTRY_DB_PATH))
    parser.add_argument("--force", action="store_true", help="Re-download pinned official sources even when cached")
    parser.add_argument("--openapi", help="Optional custom Swagger/OpenAPI JSON for a controlled extension")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--organization", default="TM Forum")
    parser.add_argument("--output-name", default="custom_openapi.json")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    normalized_dir = cache_dir / "normalized"
    profiles_dir = Path(args.profiles_dir).resolve()

    lock_path = cache_dir / "raw" / ".official-standards.sync.lock"
    with RuntimeFileLock(lock_path):
        if args.openapi:
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalize_openapi_file(
                args.openapi,
                normalized_dir / args.output_name,
                source_url=args.source_url,
                organization=args.organization,
            )
        else:
            result = sync_official_standards(
                args.manifest,
                cache_dir / "raw",
                normalized_dir,
                force=args.force,
                timeout=OFFICIAL_STANDARDS_TIMEOUT_SEC,
                max_download_mb=OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB,
                acquire_lock=False,
            )
            for item in result["sources"]:
                print(f"Synced: {item['organization']} | {item['artifact']} | {item['version']}")

        fingerprint = registry_fingerprint(normalized_dir, profiles_dir)
        RegistryBuilder(Path(args.db).resolve(), normalized_dir, profiles_dir).rebuild(fingerprint)
        print(f"Runtime registry rebuilt: {Path(args.db).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
