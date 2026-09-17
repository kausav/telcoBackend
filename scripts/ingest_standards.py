#!/usr/bin/env python3
"""Build the runtime registry from normalized standards artifacts.

Examples:
  python scripts/ingest_standards.py
  python scripts/ingest_standards.py --openapi path/to/TMF654.json \
      --source-url https://... --organization "TM Forum"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.standards_ingestion import normalize_openapi_file
from core.telecom_registry import DEFAULT_DB_PATH, DEFAULT_PROFILES_DIR, DEFAULT_STANDARDS_DIR, RegistryBuilder, registry_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest telecom standards artifacts into the INGENII runtime registry")
    parser.add_argument("--standards-dir", default=str(DEFAULT_STANDARDS_DIR))
    parser.add_argument("--profiles-dir", default=str(DEFAULT_PROFILES_DIR))
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--openapi", help="Optional Swagger/OpenAPI JSON to normalize into the standards directory")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--organization", default="TM Forum")
    parser.add_argument("--output-name", default="imported_openapi.json")
    args = parser.parse_args()

    standards_dir = Path(args.standards_dir)
    profiles_dir = Path(args.profiles_dir)
    standards_dir.mkdir(parents=True, exist_ok=True)
    if args.openapi:
        normalize_openapi_file(args.openapi, standards_dir / args.output_name, source_url=args.source_url, organization=args.organization)

    fingerprint = registry_fingerprint(standards_dir, profiles_dir)
    RegistryBuilder(Path(args.db), standards_dir, profiles_dir).rebuild(fingerprint)
    print(f"Runtime registry rebuilt: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
