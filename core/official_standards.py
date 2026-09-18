"""Fetch and materialize official telecom data-model artifacts at runtime.

The application treats the official source artifacts as the source of truth.  Only a
small manifest lives under ``resources``; downloaded source files and normalized
runtime views live under ``runtime_data`` and can be recreated at any time.

The module intentionally does not bundle 3GPP/MEF/TM Forum source files in the
application archive.  This keeps the distribution clean and lets deployments pin
an exact release/commit while preserving URL + SHA-256 provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from typing import Any, Iterable

from core.standards_ingestion import normalize_official_file


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    organization: str
    artifact: str
    version: str
    url: str
    source_page: str
    kind: str = "file"  # file | archive
    format: str = "json"  # json | yaml | asn1 | mixed
    parser: str = "auto"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    enabled: bool = True


class OfficialStandardsError(RuntimeError):
    """Raised when an official model source cannot be synchronized or parsed."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._") or "source"


def _download(url: str, destination: Path, max_bytes: int, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "INGENII-Telecom-Standards-Sync/1.0",
            "Accept": "application/json,application/yaml,application/octet-stream,text/plain,*/*",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise OfficialStandardsError(
                        f"Official source exceeds configured limit of {max_bytes} bytes: {url}"
                    )
                output.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OfficialStandardsError(f"Could not download official source {url}: {exc}") from exc


def load_source_manifest(path: str | Path) -> list[SourceSpec]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise OfficialStandardsError(f"Official standards manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise OfficialStandardsError(f"Official standards manifest has no sources: {manifest_path}")
    result: list[SourceSpec] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("source_id") or not item.get("url"):
            raise OfficialStandardsError(f"Invalid source entry in {manifest_path}: {item!r}")
        result.append(
            SourceSpec(
                source_id=str(item["source_id"]),
                organization=str(item.get("organization") or "Unknown"),
                artifact=str(item.get("artifact") or item["source_id"]),
                version=str(item.get("version") or "unknown"),
                url=str(item["url"]),
                source_page=str(item.get("source_page") or item["url"]),
                kind=str(item.get("kind") or "file"),
                format=str(item.get("format") or "json"),
                parser=str(item.get("parser") or "auto"),
                include=tuple(str(x) for x in item.get("include", []) or []),
                exclude=tuple(str(x) for x in item.get("exclude", []) or []),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return [item for item in result if item.enabled]


def _matches(path_name: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    normalized = path_name.replace("\\", "/")
    if include and not any(normalized.startswith(prefix) or prefix in normalized for prefix in include):
        return False
    if exclude and any(normalized.startswith(prefix) or prefix in normalized for prefix in exclude):
        return False
    return True


def _extract_archive(archive_path: Path, destination: Path, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename.replace("\\", "/")
                if not _matches(name, include, exclude):
                    continue
                # Do not allow archive members to escape destination.
                target = (destination / name).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise OfficialStandardsError(f"Unsafe archive member: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise OfficialStandardsError(f"Official source is not a valid ZIP archive: {archive_path}") from exc
    return extracted


def sync_official_standards(
    manifest_path: str | Path,
    raw_cache_dir: str | Path,
    normalized_dir: str | Path,
    *,
    force: bool = False,
    timeout: int | None = None,
    max_download_mb: int | None = None,
) -> dict[str, Any]:
    """Download pinned official sources and rebuild the normalized runtime cache.

    ``resources/`` remains the source *manifest* only.  Normalized JSON files are
    generated into runtime_data and are never authoritative copies of the standards.
    """
    raw_dir = Path(raw_cache_dir).resolve()
    normalized_root = Path(normalized_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)
    timeout_s = int(timeout if timeout is not None else os.getenv("OFFICIAL_STANDARDS_TIMEOUT_SEC", "45"))
    max_bytes = int(max_download_mb if max_download_mb is not None else os.getenv("OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB", "100")) * 1024 * 1024

    sources = load_source_manifest(manifest_path)
    manifest_hash = _sha256(Path(manifest_path))
    lock_file = raw_dir / ".manifest.json"

    normalized_outputs: list[Path] = []
    source_results: list[dict[str, Any]] = []

    for source in sources:
        source_dir = raw_dir / _safe_name(source.source_id)
        source_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = source_dir / "source.json"
        source_meta: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                source_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                source_meta = {}

        source_file_name = _safe_name(source.url.rsplit("/", 1)[-1] or source.source_id)
        downloaded = source_dir / source_file_name
        cached_ok = downloaded.exists() and downloaded.stat().st_size > 0 and source_meta.get("url") == source.url and source_meta.get("version") == source.version
        if force or not cached_ok:
            temp = Path(tempfile.mkstemp(prefix="official-source-", suffix=".download", dir=str(source_dir))[1])
            try:
                _download(source.url, temp, max_bytes, timeout_s)
                os.replace(temp, downloaded)
            finally:
                if temp.exists():
                    temp.unlink(missing_ok=True)

        checksum = _sha256(downloaded)
        metadata = {
            "source_id": source.source_id,
            "organization": source.organization,
            "artifact": source.artifact,
            "version": source.version,
            "url": source.url,
            "source_page": source.source_page,
            "kind": source.kind,
            "format": source.format,
            "parser": source.parser,
            "include": list(source.include),
            "exclude": list(source.exclude),
            "sha256": checksum,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }

        output_dir = normalized_root / _safe_name(source.source_id)
        existing_outputs = sorted(output_dir.glob("*.json")) if output_dir.exists() else []
        cache_matches = (
            not force
            and downloaded.exists()
            and metadata_path.exists()
            and output_dir.exists()
            and bool(existing_outputs)
        )
        if cache_matches:
            try:
                old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                cache_matches = (
                    old_meta.get("url") == source.url
                    and old_meta.get("version") == source.version
                    and old_meta.get("sha256") == checksum
                    and old_meta.get("parser") == source.parser
                    and old_meta.get("include") == list(source.include)
                    and old_meta.get("exclude") == list(source.exclude)
                )
            except (OSError, json.JSONDecodeError):
                cache_matches = False
        if cache_matches:
            source_results.append({**metadata, "normalized_files": [str(p.relative_to(normalized_root)) for p in existing_outputs]})
            normalized_outputs.extend(existing_outputs)
            continue

        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if source.kind == "archive":
            extracted_root = source_dir / "extracted"
            if extracted_root.exists():
                shutil.rmtree(extracted_root)
            extracted = _extract_archive(downloaded, extracted_root, source.include, source.exclude)
            if source.parser in {"asn1", "asn1_zip"} or source.format == "asn1":
                files = [p for p in extracted if p.suffix.lower() in {".asn", ".asn1", ".asn1p"}]
            elif source.parser in {"json_schema", "mef_schema"} or source.format in {"yaml", "json_schema"}:
                files = [p for p in extracted if p.suffix.lower() in {".yaml", ".yml", ".json"}]
            else:
                files = [p for p in extracted if p.is_file()]
        else:
            files = [downloaded]

        generated = normalize_official_file(
            files,
            output_dir,
            organization=source.organization,
            artifact=source.artifact,
            version=source.version,
            source_url=source.url,
            source_page=source.source_page,
            source_id=source.source_id,
            parser=source.parser,
        )
        normalized_outputs.extend(generated)
        source_results.append({**metadata, "normalized_files": [str(p.relative_to(normalized_root)) for p in generated]})

    # Write a compact audit manifest for operators and client evidence.
    lock_file.write_text(
        json.dumps({
            "manifest_sha256": manifest_hash,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "sources": source_results,
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "manifest_sha256": manifest_hash,
        "sources": source_results,
        "normalized_files": [str(p) for p in normalized_outputs],
    }
