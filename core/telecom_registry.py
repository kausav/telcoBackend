"""Standards-backed runtime registry for telecom schema compilation.

The application never stores the telecom entity catalogue in Python source code.
Runtime definitions are loaded from a SQLite registry database that is rebuilt from
pinned official standards artifacts and optional INGENII generation-policy profiles.
Generation profiles reference exact official source/model/field identities and may only
overlay generation behavior; they are never semantic sources.

Trust boundary:
    standards artifacts -> canonical runtime registry -> deterministic compiler

The LLM may request concepts, but it cannot add entities, attributes, relationships,
enums, generators or formulas to the registry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from threading import RLock

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None


class _CrossPlatformFileLock:
    """Small OS-level lock that works on both POSIX and Windows without extra deps."""

    def __init__(self, path: Path, timeout: float = 180.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        self._handle = handle
        deadline = __import__("time").monotonic() + self.timeout
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt  # type: ignore
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except (BlockingIOError, OSError):
                if __import__("time").monotonic() >= deadline:
                    handle.close()
                    self._handle = None
                    raise RegistryError(
                        f"Timed out waiting for the registry bootstrap lock: {self.path}"
                    )
                __import__("time").sleep(0.25)

    def __exit__(self, exc_type, exc, tb):
        handle = self._handle
        self._handle = None
        if handle is None:
            return False
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt  # type: ignore
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()
        return False


from config.runtime import (
    resolve_path,
    TELECOM_STANDARDS_DIR,
    TELECOM_STANDARDS_MANIFEST,
    TELECOM_STANDARDS_CACHE_DIR,
    TELECOM_PROFILES_DIR,
    REGISTRY_DB_PATH,
    OFFICIAL_STANDARDS_SYNC,
    OFFICIAL_STANDARDS_TIMEOUT_SEC,
    OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB,
)
from core.official_standards import sync_official_standards, OfficialStandardsError
from core.runtime_lock import RuntimeFileLock, RuntimeLockError


DEFAULT_STANDARDS_DIR = TELECOM_STANDARDS_DIR
DEFAULT_PROFILES_DIR = TELECOM_PROFILES_DIR
DEFAULT_DB_PATH = REGISTRY_DB_PATH
REGISTRY_SCHEMA_VERSION = "7"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributeDef:
    name: str
    dtype: str
    required: bool = False
    nullable: bool = False
    description: str = ""
    enum_values: tuple[str, ...] = ()
    generator: str = ""
    params: dict[str, Any] | None = None
    derived_formula: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationshipDef:
    target: str
    relation: str
    cardinality: str
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class EntityDef:
    canonical_id: str
    name: str
    aliases: tuple[str, ...]
    domain: str
    description: str
    sources: tuple[dict[str, Any], ...]
    attributes: tuple[AttributeDef, ...]
    relationships: tuple[RelationshipDef, ...]


class RegistryError(RuntimeError):
    """Raised when the standards-backed registry is invalid or unavailable."""


def _normalise(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _canonical_file_list(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.json") if p.is_file())


def registry_fingerprint(standards_dir: Path, profiles_dir: Path) -> str:
    """Hash registry inputs without depending on the deployment filesystem path."""
    hasher = hashlib.sha256()
    for label, directory in (("standards", standards_dir), ("profiles", profiles_dir)):
        for path in _canonical_file_list(directory):
            hasher.update(f"{label}/{path.relative_to(directory).as_posix()}".encode())
            hasher.update(path.read_bytes())
    hasher.update(REGISTRY_SCHEMA_VERSION.encode())
    return hasher.hexdigest()


class TelecomRegistry:
    """Read-only runtime view over the standards-backed SQLite registry."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        standards_dir: str | Path | None = None,
        profiles_dir: str | Path | None = None,
        auto_bootstrap: bool = True,
    ) -> None:
        self.db_path = resolve_path(str(db_path) if db_path is not None else os.getenv("REGISTRY_DB_PATH"), DEFAULT_DB_PATH)
        explicit_standards_dir = standards_dir is not None or bool(os.getenv("REGISTRY_STANDARDS_DIR"))
        self.standards_dir = resolve_path(
            str(standards_dir) if standards_dir is not None else os.getenv("REGISTRY_STANDARDS_DIR"),
            DEFAULT_STANDARDS_DIR,
        )
        self.profiles_dir = resolve_path(
            str(profiles_dir) if profiles_dir is not None else os.getenv("REGISTRY_PROFILES_DIR"),
            DEFAULT_PROFILES_DIR,
        )
        self.manifest_path = resolve_path(
            os.getenv("OFFICIAL_STANDARDS_MANIFEST"),
            TELECOM_STANDARDS_MANIFEST,
        )
        self.official_cache_dir = resolve_path(
            os.getenv("TELECOM_STANDARDS_CACHE_DIR"),
            TELECOM_STANDARDS_CACHE_DIR,
        )
        self._explicit_standards_dir = explicit_standards_dir
        self._cache_lock = RLock()
        self._entity_cache: dict[str, EntityDef] = {}
        self._catalog_cache: dict[tuple[str | None, int | None], list[dict[str, Any]]] = {}
        self._search_cache: dict[tuple[str, str | None, int | None], list[dict[str, Any]]] = {}
        if auto_bootstrap:
            self.ensure_current()

    @contextmanager
    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ensure_current(self) -> None:
        """Synchronize and build the shared registry exactly once per deployment."""
        bootstrap_lock = self.official_cache_dir / "raw" / ".official-standards.sync.lock"
        try:
            with RuntimeFileLock(bootstrap_lock):
                if not self._explicit_standards_dir and OFFICIAL_STANDARDS_SYNC != "disabled":
                    try:
                        sync_official_standards(
                            self.manifest_path,
                            self.official_cache_dir / "raw",
                            self.standards_dir,
                            timeout=OFFICIAL_STANDARDS_TIMEOUT_SEC,
                            max_download_mb=OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB,
                            acquire_lock=False,
                        )
                    except OfficialStandardsError as exc:
                        cached = _canonical_file_list(self.standards_dir)
                        if not cached:
                            raise RegistryError(
                                "Official telecom standards could not be synchronized and no cached model is available: "
                                f"{exc}"
                            ) from exc
                if not self.standards_dir.exists():
                    raise RegistryError(f"Standards directory does not exist: {self.standards_dir}")
                standards = _canonical_file_list(self.standards_dir)
                if not standards:
                    raise RegistryError(f"No official standards model artifacts found in {self.standards_dir}")

                fingerprint = registry_fingerprint(self.standards_dir, self.profiles_dir)
                try:
                    with self._connect() as conn:
                        row = conn.execute("SELECT value FROM registry_meta WHERE key='fingerprint'").fetchone()
                        schema_row = conn.execute("SELECT value FROM registry_meta WHERE key='schema_version'").fetchone()
                        if row and schema_row and row[0] == fingerprint and schema_row[0] == REGISTRY_SCHEMA_VERSION:
                            return
                except sqlite3.OperationalError:
                    pass
                RegistryBuilder(self.db_path, self.standards_dir, self.profiles_dir).rebuild(fingerprint)
        except RuntimeLockError as exc:
            raise RegistryError(str(exc)) from exc

    def health(self) -> dict[str, Any]:
        with self._connect() as conn:
            standards = conn.execute("SELECT COUNT(*) FROM standards").fetchone()[0]
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            attributes = conn.execute("SELECT COUNT(*) FROM attributes").fetchone()[0]
            relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            sources = conn.execute("SELECT COUNT(*) FROM entity_sources").fetchone()[0]
            meta = dict(conn.execute("SELECT key, value FROM registry_meta").fetchall())
        return {
            "healthy": entities > 0,
            "standards": standards,
            "entities": entities,
            "attributes": attributes,
            "relationships": relationships,
            "provenance_records": sources,
            "schema_version": meta.get("schema_version"),
            "fingerprint": meta.get("fingerprint"),
        }

    def catalog_summary(self, domain: str | None = None, limit: int | None = 200) -> list[dict[str, Any]]:
        sql = "SELECT canonical_id, name, domain, description FROM entities"
        args: list[Any] = []
        if domain:
            sql += " WHERE domain = ?"
            args.append(domain)
        # ``None`` means no application-side result cap. This is intentionally used by
        # the agentic proposal path so the complete approved registry is available.
        effective_limit = None if limit is None else max(1, int(limit))
        cache_key = (domain, effective_limit)
        with self._cache_lock:
            cached = self._catalog_cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]
        sql += " ORDER BY canonical_id"
        if effective_limit is not None:
            sql += " LIMIT ?"
            args.append(effective_limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
            result = []
            for canonical_id, name, entity_domain, description in rows:
                aliases = [r[0] for r in conn.execute("SELECT alias FROM aliases WHERE canonical_id=? ORDER BY alias", (canonical_id,)).fetchall()]
                source_rows = conn.execute(
                    "SELECT DISTINCT standard FROM entity_sources WHERE canonical_id=? ORDER BY standard",
                    (canonical_id,),
                ).fetchall()
                result.append({
                    "canonical_id": canonical_id,
                    "name": name,
                    "aliases": aliases,
                    "domain": entity_domain,
                    "description": description,
                    "source_standards": [r[0] for r in source_rows],
                })
            with self._cache_lock:
                self._catalog_cache[cache_key] = [dict(item) for item in result]
            return result

    def llm_catalog_context(self, query: str | None = None) -> dict[str, Any]:
        """Return official-model-derived grounding context for intent understanding.

        All registered official source metadata/URLs are always included. When a query is
        supplied, entity/attribute/relationship payload is restricted to entities that are
        lexically relevant to that request; there is no numeric entity/field cap. The compiler
        still operates against the full registry graph and can expand every approved relation.
        This avoids blowing the LLM context window on unrelated portions of a complete standards
        library while preserving all official models in the backend registry.
        """
        with self._connect() as conn:
            standard_rows = conn.execute(
                """SELECT artifact_id, organization, title, version, status, source_kind, source_url, source_page
                   FROM standards ORDER BY organization, artifact_id"""
            ).fetchall()
            if query and str(query).strip():
                selected_ids = [item["canonical_id"] for item in self.search(str(query), limit=None)]
                if selected_ids:
                    placeholders = ",".join("?" for _ in selected_ids)
                    entity_rows = conn.execute(
                        f"""SELECT canonical_id, name, domain, description
                            FROM entities WHERE canonical_id IN ({placeholders})
                            ORDER BY canonical_id""",
                        selected_ids,
                    ).fetchall()
                else:
                    entity_rows = []
            else:
                entity_rows = conn.execute(
                    """SELECT canonical_id, name, domain, description
                       FROM entities ORDER BY canonical_id"""
                ).fetchall()

            standards = [
                {
                    "artifact_id": row[0],
                    "organization": row[1],
                    "title": row[2],
                    "version": row[3],
                    "status": row[4],
                    "source_kind": row[5],
                    "source_url": row[6],
                    "source_page": row[7],
                }
                for row in standard_rows
            ]

            entities: list[dict[str, Any]] = []
            for canonical_id, name, domain, description in entity_rows:
                aliases = [
                    row[0]
                    for row in conn.execute(
                        "SELECT alias FROM aliases WHERE canonical_id=? ORDER BY alias",
                        (canonical_id,),
                    ).fetchall()
                ]
                sources = [
                    dict(zip(("standard", "artifact", "version", "reference", "url", "source_page", "source_role"), row))
                    for row in conn.execute(
                        """SELECT standard, artifact, version, reference, url, source_page, source_role
                           FROM entity_sources WHERE canonical_id=?
                           ORDER BY standard, artifact, reference""",
                        (canonical_id,),
                    ).fetchall()
                ]
                attributes = []
                for row in conn.execute(
                    """SELECT name, dtype, required, nullable, description, enum_values_json,
                              generator, params_json, derived_formula, depends_on_json
                       FROM attributes WHERE canonical_id=? ORDER BY ordinal""",
                    (canonical_id,),
                ).fetchall():
                    attributes.append({
                        "name": row[0],
                        "dtype": row[1],
                        "required": bool(row[2]),
                        "nullable": bool(row[3]),
                        "description": row[4] or "",
                        "enum_values": json.loads(row[5] or "[]"),
                        "generator": row[6] or "",
                        "params": json.loads(row[7] or "{}"),
                        "derived_formula": row[8],
                        "depends_on": json.loads(row[9] or "[]"),
                    })
                relationships = [
                    dict(zip(("target_entity", "relation", "cardinality", "required", "description"), row))
                    for row in conn.execute(
                        """SELECT target_entity, relation, cardinality, required, description
                           FROM relationships WHERE source_entity=? ORDER BY ordinal""",
                        (canonical_id,),
                    ).fetchall()
                ]
                entities.append({
                    "canonical_id": canonical_id,
                    "name": name,
                    "aliases": aliases,
                    "domain": domain,
                    "description": description,
                    "sources": sources,
                    "attributes": attributes,
                    "relationships": relationships,
                })

        # Deduplicate the source URLs across artifact/entity provenance while retaining
        # their standard/artifact context. These URLs are supplied as grounding references;
        # the code does not pretend to fetch live standards during every proposal request.
        sources_by_url: dict[str, dict[str, Any]] = {}
        for standard in standards:
            url = standard.get("source_url")
            if url:
                sources_by_url.setdefault(url, {
                    "url": url,
                    "standard": standard["organization"],
                    "artifact_ids": [],
                })["artifact_ids"].append(standard["artifact_id"])
        for entity in entities:
            for source in entity["sources"]:
                url = source.get("url")
                if not url:
                    continue
                item = sources_by_url.setdefault(url, {
                    "url": url,
                    "standard": source.get("standard"),
                    "artifact_ids": [],
                })
                artifact_name = source.get("artifact")
                if artifact_name and artifact_name not in item["artifact_ids"]:
                    item["artifact_ids"].append(artifact_name)

        return {
            "standards": standards,
            "source_urls": sorted(sources_by_url.values(), key=lambda item: item["url"]),
            "entities": entities,
            "registry_health": self.health(),
        }

    def entities_with_attribute(self, attribute_name: str) -> list[EntityDef]:
        """Return registry entities exposing an exact attribute name, deterministically."""
        normalized = str(attribute_name or "").strip().lower()
        if not normalized:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT canonical_id FROM attributes WHERE lower(name)=? ORDER BY canonical_id",
                (normalized,),
            ).fetchall()
        return [self.get_entity(row[0]) for row in rows]

    def related_entities(self, entity_id: str) -> list[EntityDef]:
        """Return entities connected to *entity_id* by approved registry relationships.

        Both outgoing and incoming edges are considered. Results are deterministic and
        remain fully registry-backed; this method never creates or infers entities.
        """
        key = _normalise(entity_id).replace(" ", "_")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT target_entity FROM relationships WHERE source_entity = ?
                   UNION
                   SELECT DISTINCT source_entity FROM relationships WHERE target_entity = ?
                   ORDER BY 1""",
                (key, key),
            ).fetchall()
        result = []
        for (candidate,) in rows:
            try:
                result.append(self.get_entity(candidate))
            except KeyError:
                continue
        return result

    def resolve_entity(self, value: str) -> EntityDef | None:
        key = _normalise(value)
        with self._connect() as conn:
            row = conn.execute("SELECT canonical_id FROM aliases WHERE alias=? ORDER BY canonical_id LIMIT 1", (key,)).fetchone()
            if row is None:
                row = conn.execute("SELECT canonical_id FROM entities WHERE canonical_id=?", (key.replace(" ", "_"),)).fetchone()
            if row is None:
                return None
        return self.get_entity(row[0])

    def get_entity(self, entity_id: str) -> EntityDef:
        key = _normalise(entity_id).replace(" ", "_")
        with self._cache_lock:
            cached = self._entity_cache.get(key)
            if cached is not None:
                return cached
        with self._connect() as conn:
            entity = conn.execute(
                "SELECT canonical_id, name, domain, description FROM entities WHERE canonical_id=?", (key,)
            ).fetchone()
            if entity is None:
                raise KeyError(entity_id)
            canonical_id, name, domain, description = entity
            aliases = tuple(r[0] for r in conn.execute("SELECT alias FROM aliases WHERE canonical_id=? ORDER BY alias", (canonical_id,)).fetchall())
            sources = tuple(
                dict(zip(("standard", "artifact", "version", "reference", "url", "source_page", "source_role"), row))
                for row in conn.execute(
                    "SELECT standard, artifact, version, reference, url, source_page, source_role FROM entity_sources WHERE canonical_id=? ORDER BY standard, artifact, reference",
                    (canonical_id,),
                ).fetchall()
            )
            attrs: list[AttributeDef] = []
            attr_rows = conn.execute(
                """SELECT name, dtype, required, nullable, description, enum_values_json,
                          generator, params_json, derived_formula, depends_on_json
                   FROM attributes WHERE canonical_id=? ORDER BY ordinal""", (canonical_id,)
            ).fetchall()
            for row in attr_rows:
                attrs.append(AttributeDef(
                    name=row[0], dtype=row[1], required=bool(row[2]), nullable=bool(row[3]), description=row[4] or "",
                    enum_values=tuple(json.loads(row[5] or "[]")), generator=row[6] or "",
                    params=json.loads(row[7] or "{}"), derived_formula=row[8], depends_on=tuple(json.loads(row[9] or "[]")),
                ))
            relationships = tuple(
                RelationshipDef(target=row[0], relation=row[1], cardinality=row[2], required=bool(row[3]), description=row[4] or "")
                for row in conn.execute(
                    "SELECT target_entity, relation, cardinality, required, description FROM relationships WHERE source_entity=? ORDER BY ordinal",
                    (canonical_id,),
                ).fetchall()
            )
        result = EntityDef(canonical_id, name, aliases, domain, description, sources, tuple(attrs), relationships)
        with self._cache_lock:
            self._entity_cache[key] = result
        return result

    def entity_dict(self, entity_id: str) -> dict[str, Any]:
        e = self.get_entity(entity_id)
        return {
            "canonical_id": e.canonical_id,
            "name": e.name,
            "aliases": list(e.aliases),
            "domain": e.domain,
            "description": e.description,
            "sources": [dict(s) for s in e.sources],
            "attributes": [
                {
                    **asdict(a),
                    "enum_values": list(a.enum_values),
                    "params": dict(a.params or {}),
                    "depends_on": list(a.depends_on),
                }
                for a in e.attributes
            ],
            "relationships": [asdict(r) for r in e.relationships],
        }

    def standards_for_entities(self, entity_ids: list[str] | tuple[str, ...] | set[str]) -> list[dict[str, Any]]:
        """Derive applicable standards from resolved entity provenance.

        Standards are never selected by the LLM. A standard is applicable only when
        an approved registry entity has provenance to that standards family.
        """
        normalized_ids: list[str] = []
        for value in entity_ids:
            entity = self.resolve_entity(value)
            if entity and entity.canonical_id not in normalized_ids:
                normalized_ids.append(entity.canonical_id)
        if not normalized_ids:
            return []

        placeholders = ",".join("?" for _ in normalized_ids)
        sql = f"""
            SELECT DISTINCT standard, artifact, version, reference, url
            FROM entity_sources
            WHERE canonical_id IN ({placeholders})
            ORDER BY standard, artifact, reference
        """
        with self._connect() as conn:
            rows = conn.execute(sql, normalized_ids).fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        for standard, artifact, version, reference, url in rows:
            group = grouped.setdefault(standard, {"standard": standard, "artifacts": []})
            group["artifacts"].append({
                "artifact": artifact,
                "version": version,
                "reference": reference,
                "url": url,
            })
        return list(grouped.values())

    def entity_exists(self, entity_id: str) -> bool:
        try:
            self.get_entity(entity_id)
            return True
        except KeyError:
            return False

    def search(self, query: str, domain: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Deterministic lexical search over names and aliases; one result per entity."""
        normalized = _normalise(query)
        effective_limit = None if limit is None else max(1, int(limit))
        cache_key = (normalized, domain, effective_limit)
        with self._cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]
        tokens = [token for token in normalized.split() if token]
        best_by_entity: dict[str, tuple[int, dict[str, Any]]] = {}

        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.canonical_id, e.name, e.domain, e.description,
                          GROUP_CONCAT(DISTINCT a.alias), GROUP_CONCAT(DISTINCT es.artifact)
                   FROM entities e
                   LEFT JOIN aliases a ON a.canonical_id=e.canonical_id
                   LEFT JOIN entity_sources es ON es.canonical_id=e.canonical_id
                   GROUP BY e.canonical_id, e.name, e.domain, e.description"""
            ).fetchall()

            source_cache: dict[str, list[str]] = {}
            for canonical_id, name, entity_domain, description, aliases_text, artifacts_text in rows:
                if domain and entity_domain != domain:
                    continue
                aliases = [item for item in str(aliases_text or "").split(",") if item]
                search_text = _normalise(" ".join([
                    str(name or ""), str(canonical_id or ""), str(description or ""),
                    *aliases, *(str(artifacts_text or "").split(",")),
                ]))
                score = 0
                if normalized and normalized in search_text:
                    score += 45
                for token in tokens:
                    if token in search_text:
                        score += 10
                if any(_normalise(alias) == normalized for alias in aliases):
                    score += 100
                if score <= 0:
                    continue

                if canonical_id not in source_cache:
                    source_cache[canonical_id] = [
                        row[0]
                        for row in conn.execute(
                            "SELECT DISTINCT standard FROM entity_sources WHERE canonical_id=? ORDER BY standard",
                            (canonical_id,),
                        ).fetchall()
                    ]

                item = {
                    "canonical_id": canonical_id,
                    "name": name,
                    "domain": entity_domain,
                    "description": description,
                    "source_standards": source_cache[canonical_id],
                }
                current = best_by_entity.get(canonical_id)
                if current is None or score > current[0]:
                    best_by_entity[canonical_id] = (score, item)

        scored = list(best_by_entity.values())
        scored.sort(key=lambda item: (-item[0], item[1]["canonical_id"]))
        result = scored if effective_limit is None else scored[:effective_limit]
        result = [item for _, item in result]
        with self._cache_lock:
            self._search_cache[cache_key] = [dict(item) for item in result]
        return result


class RegistryBuilder:
    """Build the runtime SQLite registry from generated indexes of official artifacts."""

    def __init__(self, db_path: Path, standards_dir: Path, profiles_dir: Path) -> None:
        self.db_path = db_path
        self.standards_dir = standards_dir
        self.profiles_dir = profiles_dir

    def rebuild(self, fingerprint: str) -> None:
        artifacts = []
        for path in _canonical_file_list(self.standards_dir):
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if "artifact" not in document or "entities" not in document:
                raise RegistryError(f"Invalid normalized standards artifact: {path}")
            artifacts.append((path, document))
        profiles = self._load_profiles()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("BEGIN IMMEDIATE")
            self._create_tables(conn)
            for table in (
                "standards", "entities", "aliases", "entity_sources", "attributes", "relationships", "generation_profiles"
            ):
                conn.execute(f"DELETE FROM {table}")

            entity_ids: set[str] = set()
            for path, document in artifacts:
                artifact = document["artifact"]
                self._insert_artifact(conn, artifact, path)
                for entity in document["entities"]:
                    cid = self._normalise_canonical_id(entity["canonical_id"])
                    if cid in entity_ids:
                        raise RegistryError(f"Duplicate canonical entity '{cid}' across standards artifacts")
                    entity_ids.add(cid)
                    self._insert_entity(conn, cid, entity, artifact, path)

            for profile in profiles:
                self._insert_profile(conn, profile)

            self._apply_generation_profiles(conn)
            self._validate_graph(conn)
            now = str(__import__("time").time())
            conn.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES('schema_version',?)", (REGISTRY_SCHEMA_VERSION,))
            conn.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES('fingerprint',?)", (fingerprint,))
            conn.execute("INSERT OR REPLACE INTO registry_meta(key,value) VALUES('built_at',?)", (now,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        self._migrate_legacy_schema(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS standards(
                artifact_id TEXT PRIMARY KEY,
                organization TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT,
                status TEXT,
                source_kind TEXT,
                source_url TEXT,
                source_page TEXT,
                source_sha256 TEXT
            );
            CREATE TABLE IF NOT EXISTS entities(
                canonical_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                description TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                FOREIGN KEY(artifact_id) REFERENCES standards(artifact_id)
            );
            CREATE TABLE IF NOT EXISTS aliases(
                alias TEXT NOT NULL,
                canonical_id TEXT NOT NULL,
                PRIMARY KEY(alias, canonical_id),
                FOREIGN KEY(canonical_id) REFERENCES entities(canonical_id)
            );
            CREATE TABLE IF NOT EXISTS entity_sources(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_id TEXT NOT NULL,
                standard TEXT NOT NULL,
                artifact TEXT NOT NULL,
                version TEXT,
                reference TEXT NOT NULL,
                url TEXT,
                source_page TEXT,
                source_role TEXT NOT NULL DEFAULT 'semantic-source',
                FOREIGN KEY(canonical_id) REFERENCES entities(canonical_id)
            );
            CREATE TABLE IF NOT EXISTS attributes(
                canonical_id TEXT NOT NULL,
                name TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                required INTEGER NOT NULL,
                nullable INTEGER NOT NULL,
                description TEXT NOT NULL,
                enum_values_json TEXT NOT NULL,
                generator TEXT NOT NULL DEFAULT '',
                params_json TEXT NOT NULL DEFAULT '{}',
                derived_formula TEXT,
                depends_on_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(canonical_id, name),
                FOREIGN KEY(canonical_id) REFERENCES entities(canonical_id)
            );
            CREATE TABLE IF NOT EXISTS relationships(
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                relation TEXT NOT NULL,
                cardinality TEXT NOT NULL,
                required INTEGER NOT NULL,
                description TEXT NOT NULL,
                PRIMARY KEY(source_entity, target_entity, relation, cardinality),
                FOREIGN KEY(source_entity) REFERENCES entities(canonical_id),
                FOREIGN KEY(target_entity) REFERENCES entities(canonical_id)
            );
            CREATE TABLE IF NOT EXISTS generation_profiles(
                profile_id TEXT NOT NULL,
                canonical_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                generator TEXT NOT NULL,
                params_json TEXT NOT NULL DEFAULT '{}',
                derived_formula TEXT,
                depends_on_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY(profile_id, canonical_id, field_name)
            );
            CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON aliases(canonical_id);
            CREATE INDEX IF NOT EXISTS idx_attributes_entity ON attributes(canonical_id, ordinal);
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity, ordinal);
            """
        )

    @staticmethod
    def _migrate_legacy_schema(conn: sqlite3.Connection) -> None:
        """Upgrade the small runtime SQLite schema used by prior application versions."""
        def columns(table: str) -> dict[str, int]:
            try:
                return {row[1]: row[5] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.OperationalError:
                return {}

        # Older builds made alias globally unique, which is unsafe once multiple official
        # TMF/MEF model files contain the same model name. Recreate the tiny table.
        alias_cols = columns("aliases")
        if alias_cols and alias_cols.get("alias") == 1:
            conn.execute("DROP TABLE IF EXISTS aliases")

        if "source_page" not in columns("standards") and columns("standards"):
            conn.execute("ALTER TABLE standards ADD COLUMN source_page TEXT")
        if "source_page" not in columns("entity_sources") and columns("entity_sources"):
            conn.execute("ALTER TABLE entity_sources ADD COLUMN source_page TEXT")

    @staticmethod
    def _normalise_canonical_id(value: str) -> str:
        return _normalise(value).replace(" ", "_")

    def _insert_artifact(self, conn: sqlite3.Connection, artifact: dict[str, Any], path: Path) -> None:
        sources = artifact.get("sources") or []
        default_url = sources[0].get("url") if sources else artifact.get("source_url")
        default_page = sources[0].get("source_page") if sources else artifact.get("source_page")
        default_version = sources[0].get("version") if sources else artifact.get("version")
        conn.execute(
            "INSERT INTO standards(artifact_id, organization, title, version, status, source_kind, source_url, source_page, source_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                artifact["artifact_id"], artifact.get("organization", "Unknown"), artifact.get("title", artifact["artifact_id"]),
                artifact.get("artifact_version", default_version), artifact.get("status"), artifact.get("source_kind", "official-machine-readable-model"),
                default_url, default_page, hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )

    def _insert_entity(self, conn: sqlite3.Connection, cid: str, entity: dict[str, Any], artifact: dict[str, Any], path: Path) -> None:
        conn.execute(
            "INSERT INTO entities(canonical_id,name,domain,description,artifact_id) VALUES(?,?,?,?,?)",
            (cid, entity["name"], entity.get("domain", "telecom"), entity.get("description", ""), artifact["artifact_id"]),
        )
        aliases = set(entity.get("aliases") or []) | {cid, entity["name"]}
        for alias in aliases:
            conn.execute("INSERT OR IGNORE INTO aliases(alias,canonical_id) VALUES(?,?)", (_normalise(alias), cid))
        source_records = entity.get("sources") or artifact.get("sources") or []
        for source in source_records:
            conn.execute(
                "INSERT INTO entity_sources(canonical_id,standard,artifact,version,reference,url,source_page,source_role) VALUES(?,?,?,?,?,?,?,?)",
                (
                    cid, source.get("standard", artifact.get("organization", "Unknown")), source.get("artifact", artifact.get("title", artifact["artifact_id"])),
                    source.get("version") or artifact.get("artifact_version"), source.get("reference", artifact["artifact_id"]),
                    source.get("url") or artifact.get("source_url"), source.get("source_page") or artifact.get("source_page"),
                    source.get("source_role", "semantic-source"),
                ),
            )
        for ordinal, attribute in enumerate(entity.get("attributes") or []):
            conn.execute(
                """INSERT INTO attributes(
                    canonical_id,name,ordinal,dtype,required,nullable,description,enum_values_json,
                    generator,params_json,derived_formula,depends_on_json
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cid, attribute["name"], ordinal, attribute.get("dtype", "string"), int(bool(attribute.get("required"))),
                    int(bool(attribute.get("nullable"))), attribute.get("description", ""), _json(attribute.get("enum_values") or []),
                    attribute.get("generator", "") or "", _json(attribute.get("params") or {}),
                    attribute.get("derived_formula"), _json(attribute.get("depends_on") or []),
                ),
            )
        for ordinal, relationship in enumerate(entity.get("relationships") or []):
            target = self._normalise_canonical_id(relationship["target"])
            conn.execute(
                """INSERT OR REPLACE INTO relationships(source_entity,target_entity,ordinal,relation,cardinality,required,description)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    cid, target, ordinal, relationship["relation"], relationship["cardinality"], int(bool(relationship.get("required"))), relationship.get("description", ""),
                ),
            )

    def _load_profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for path in _canonical_file_list(self.profiles_dir):
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            profile = document.get("profile") if isinstance(document, dict) else None
            targets = document.get("targets") if isinstance(document, dict) else None
            if not isinstance(profile, dict) or not profile.get("profile_id") or not isinstance(targets, list):
                raise RegistryError(
                    f"Invalid generation profile: {path}. Expected {{profile: {{profile_id: ...}}, targets: [...]}}."
                )
            for index, target in enumerate(targets):
                if not isinstance(target, dict) or not target.get("source_id") or not target.get("model") or not target.get("field") or not target.get("generator"):
                    raise RegistryError(
                        f"Invalid generation profile target at {path} index {index}: expected source_id, model, field and generator."
                    )
            profiles.append(document)
        return profiles

    def _insert_profile(self, conn: sqlite3.Connection, document: dict[str, Any]) -> None:
        profile = document["profile"]
        profile_id = profile["profile_id"]
        for target in document.get("targets") or []:
            source_id = self._normalise_canonical_id(str(target["source_id"]))
            model = str(target["model"]).strip()
            field = str(target["field"]).strip()
            if not field:
                raise RegistryError(f"Generation profile '{profile_id}' contains an empty field target")

            if model == "*":
                entity_rows = conn.execute(
                    "SELECT canonical_id FROM entities WHERE artifact_id=? ORDER BY canonical_id",
                    (source_id,),
                ).fetchall()
            else:
                entity_rows = conn.execute(
                    "SELECT canonical_id FROM entities WHERE artifact_id=? AND lower(name)=lower(?) ORDER BY canonical_id",
                    (source_id, model),
                ).fetchall()

            if not entity_rows:
                message = (
                    f"Generation profile '{profile_id}' target does not exist in the official registry: "
                    f"{source_id}:{model}.{field}"
                )
                if bool(target.get("required", False)):
                    raise RegistryError(message)
                logger.warning(message + "; skipping optional generation policy target")
                continue

            matched_fields = []
            for (canonical_id,) in entity_rows:
                row = conn.execute(
                    "SELECT 1 FROM attributes WHERE canonical_id=? AND name=?",
                    (canonical_id, field),
                ).fetchone()
                if row is not None:
                    matched_fields.append(canonical_id)

            if not matched_fields:
                message = (
                    f"Generation profile '{profile_id}' targets an attribute that is not present in the official registry: "
                    f"{source_id}:{model}.{field}"
                )
                if bool(target.get("required", False)):
                    raise RegistryError(message)
                logger.warning(message + "; skipping optional generation policy target")
                continue

            for canonical_id in matched_fields:
                conn.execute(
                    "INSERT INTO generation_profiles(profile_id,canonical_id,field_name,generator,params_json,derived_formula,depends_on_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        profile_id, canonical_id, field, target.get("generator", ""), _json(target.get("params") or {}),
                        target.get("derived_formula"), _json(target.get("depends_on") or []),
                    ),
                )

    def _apply_generation_profiles(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT profile_id, canonical_id, field_name, generator, params_json, derived_formula, depends_on_json FROM generation_profiles ORDER BY profile_id, canonical_id, field_name"
        ).fetchall()
        for _, cid, field, generator, params_json, formula, depends_on_json in rows:
            conn.execute(
                "UPDATE attributes SET generator=?, params_json=?, derived_formula=?, depends_on_json=? WHERE canonical_id=? AND name=?",
                (generator, params_json, formula, depends_on_json, cid, field),
            )

    def _validate_graph(self, conn: sqlite3.Connection) -> None:
        entity_ids = {r[0] for r in conn.execute("SELECT canonical_id FROM entities").fetchall()}
        if not entity_ids:
            raise RegistryError("Standards registry contains no entities")
        for source, target, *_ in conn.execute("SELECT source_entity, target_entity, relation, cardinality, required, description FROM relationships"):
            if source not in entity_ids or target not in entity_ids:
                raise RegistryError(f"Broken standards relationship: {source} -> {target}")
        allowed_generators = {"", "unique_id", "reference", "msisdn", "timestamp", "constant", "range", "weighted_choice", "dependent_choice", "semantic_event"}
        for cid, name, generator in conn.execute("SELECT canonical_id, name, generator FROM attributes"):
            if generator not in allowed_generators:
                raise RegistryError(f"Unsupported generator in registry: {cid}.{name}={generator}")


_default_registry: TelecomRegistry | None = None


def get_registry() -> TelecomRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = TelecomRegistry()
    return _default_registry


def catalog_summary() -> list[dict[str, Any]]:
    return get_registry().catalog_summary()


def resolve_entity(value: str) -> EntityDef | None:
    return get_registry().resolve_entity(value)


def get_entity(entity_id: str) -> EntityDef:
    return get_registry().get_entity(entity_id)


def entity_dict(entity_id: str) -> dict[str, Any]:
    return get_registry().entity_dict(entity_id)