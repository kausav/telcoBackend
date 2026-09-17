"""Standards-backed runtime registry for telecom schema compilation.

The application never stores the telecom entity catalogue in Python source code.
Runtime definitions are loaded from a SQLite registry database that is rebuilt from
versioned standards artifacts and INGENII generation profiles.

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
import os
import sqlite3
from contextlib import contextmanager

from config.runtime import TELECOM_STANDARDS_DIR, TELECOM_PROFILES_DIR, REGISTRY_DB_PATH


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STANDARDS_DIR = TELECOM_STANDARDS_DIR
DEFAULT_PROFILES_DIR = TELECOM_PROFILES_DIR
DEFAULT_DB_PATH = REGISTRY_DB_PATH
REGISTRY_SCHEMA_VERSION = "2"


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
        self.db_path = Path(db_path or os.getenv("REGISTRY_DB_PATH") or DEFAULT_DB_PATH)
        self.standards_dir = Path(standards_dir or os.getenv("REGISTRY_STANDARDS_DIR") or DEFAULT_STANDARDS_DIR).expanduser().resolve()
        self.profiles_dir = Path(profiles_dir or os.getenv("REGISTRY_PROFILES_DIR") or DEFAULT_PROFILES_DIR).expanduser().resolve()
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
        if not self.standards_dir.exists():
            raise RegistryError(f"Standards directory does not exist: {self.standards_dir}")
        standards = _canonical_file_list(self.standards_dir)
        if not standards:
            raise RegistryError(f"No normalized standards artifacts found in {self.standards_dir}")

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

    def catalog_summary(self, domain: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT canonical_id, name, domain, description FROM entities"
        args: list[Any] = []
        if domain:
            sql += " WHERE domain = ?"
            args.append(domain)
        sql += " ORDER BY canonical_id LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
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
            return result

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
            row = conn.execute("SELECT canonical_id FROM aliases WHERE alias=?", (key,)).fetchone()
            if row is None:
                row = conn.execute("SELECT canonical_id FROM entities WHERE canonical_id=?", (key.replace(" ", "_"),)).fetchone()
            if row is None:
                return None
        return self.get_entity(row[0])

    def get_entity(self, entity_id: str) -> EntityDef:
        key = _normalise(entity_id).replace(" ", "_")
        with self._connect() as conn:
            entity = conn.execute(
                "SELECT canonical_id, name, domain, description FROM entities WHERE canonical_id=?", (key,)
            ).fetchone()
            if entity is None:
                raise KeyError(entity_id)
            canonical_id, name, domain, description = entity
            aliases = tuple(r[0] for r in conn.execute("SELECT alias FROM aliases WHERE canonical_id=? ORDER BY alias", (canonical_id,)).fetchall())
            sources = tuple(
                dict(zip(("standard", "artifact", "version", "reference", "url", "source_role"), row))
                for row in conn.execute(
                    "SELECT standard, artifact, version, reference, url, source_role FROM entity_sources WHERE canonical_id=? ORDER BY standard, artifact, reference",
                    (canonical_id,),
                ).fetchall()
            )
            attrs: list[AttributeDef] = []
            attr_rows = conn.execute(
                """SELECT name, dtype, required, nullable, description, enum_values_json,
                          generator, params_json, derived_formula, depends_on_json
                   FROM attributes WHERE canonical_id=? ORDER BY ordinal""",
                (canonical_id,),
            ).fetchall()
            for row in attr_rows:
                attrs.append(AttributeDef(
                    name=row[0], dtype=row[1], required=bool(row[2]), nullable=bool(row[3]),
                    description=row[4] or "", enum_values=tuple(json.loads(row[5] or "[]")),
                    generator=row[6] or "", params=json.loads(row[7] or "{}"),
                    derived_formula=row[8], depends_on=tuple(json.loads(row[9] or "[]")),
                ))
            rels = tuple(
                RelationshipDef(target=row[0], relation=row[1], cardinality=row[2], required=bool(row[3]), description=row[4] or "")
                for row in conn.execute(
                    "SELECT target_entity, relation, cardinality, required, description FROM relationships WHERE source_entity=? ORDER BY ordinal",
                    (canonical_id,),
                ).fetchall()
            )
        return EntityDef(canonical_id, name, aliases, domain, description, sources, tuple(attrs), rels)

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

    def relationship_exists(self, source_entity: str, target_entity: str, relation: str, cardinality: str) -> bool:
        source = self.resolve_entity(source_entity)
        target = self.resolve_entity(target_entity)
        if not source or not target:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM relationships WHERE source_entity=? AND target_entity=? AND relation=? AND cardinality=?",
                (source.canonical_id, target.canonical_id, relation, cardinality),
            ).fetchone()
        return row is not None

    def search(self, query: str, domain: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Deterministic lexical search over names and aliases; one result per entity."""
        normalized = _normalise(query)
        tokens = [token for token in normalized.split() if token]
        best_by_entity: dict[str, tuple[int, dict[str, Any]]] = {}

        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.canonical_id, e.name, e.domain, e.description, a.alias
                   FROM entities e JOIN aliases a ON a.canonical_id=e.canonical_id"""
            ).fetchall()

            source_cache: dict[str, list[str]] = {}
            for canonical_id, name, entity_domain, description, alias in rows:
                if domain and entity_domain != domain:
                    continue
                alias_norm = _normalise(alias)
                score = 0
                if alias_norm == normalized:
                    score += 100
                if normalized and normalized in alias_norm:
                    score += 40
                score += sum(10 for token in tokens if token in alias_norm)
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
        return [item for _, item in scored[: max(1, min(limit, 100))]]


class RegistryBuilder:
    """Build the runtime SQLite registry from versioned JSON artifacts."""

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
            CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, canonical_id TEXT NOT NULL, FOREIGN KEY(canonical_id) REFERENCES entities(canonical_id));
            CREATE TABLE IF NOT EXISTS entity_sources(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_id TEXT NOT NULL,
                standard TEXT NOT NULL,
                artifact TEXT NOT NULL,
                version TEXT,
                reference TEXT NOT NULL,
                url TEXT,
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
    def _normalise_canonical_id(value: str) -> str:
        return _normalise(value).replace(" ", "_")

    def _insert_artifact(self, conn: sqlite3.Connection, artifact: dict[str, Any], path: Path) -> None:
        sources = artifact.get("sources") or []
        default_url = sources[0].get("url") if sources else artifact.get("source_url")
        default_version = sources[0].get("version") if sources else artifact.get("version")
        conn.execute(
            "INSERT INTO standards(artifact_id, organization, title, version, status, source_kind, source_url, source_sha256) VALUES(?,?,?,?,?,?,?,?)",
            (
                artifact["artifact_id"], artifact.get("organization", "Unknown"), artifact.get("title", artifact["artifact_id"]),
                artifact.get("artifact_version", default_version), artifact.get("status"), artifact.get("source_kind", "normalized"),
                default_url, hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )

    def _insert_entity(self, conn: sqlite3.Connection, cid: str, entity: dict[str, Any], artifact: dict[str, Any], path: Path) -> None:
        conn.execute(
            "INSERT INTO entities(canonical_id,name,domain,description,artifact_id) VALUES(?,?,?,?,?)",
            (cid, entity["name"], entity.get("domain", "telecom"), entity.get("description", ""), artifact["artifact_id"]),
        )
        aliases = set(entity.get("aliases") or []) | {cid, entity["name"]}
        for alias in aliases:
            conn.execute("INSERT OR REPLACE INTO aliases(alias,canonical_id) VALUES(?,?)", (_normalise(alias), cid))
        source_records = entity.get("sources") or artifact.get("sources") or []
        for source in source_records:
            conn.execute(
                "INSERT INTO entity_sources(canonical_id,standard,artifact,version,reference,url,source_role) VALUES(?,?,?,?,?,?,?)",
                (
                    cid, source.get("standard", artifact.get("organization", "Unknown")), source.get("artifact", artifact.get("title", artifact["artifact_id"])),
                    source.get("version") or artifact.get("artifact_version"), source.get("reference", artifact["artifact_id"]),
                    source.get("url") or artifact.get("source_url"), source.get("source_role", "semantic-source"),
                ),
            )
        for ordinal, attribute in enumerate(entity.get("attributes") or []):
            conn.execute(
                """INSERT INTO attributes(canonical_id,name,ordinal,dtype,required,nullable,description,enum_values_json)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    cid, attribute["name"], ordinal, attribute.get("dtype", "string"), int(bool(attribute.get("required"))),
                    int(bool(attribute.get("nullable"))), attribute.get("description", ""), _json(attribute.get("enum_values") or []),
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
            if "profile" not in document or "fields" not in document:
                raise RegistryError(f"Invalid generation profile: {path}")
            profiles.append(document)
        return profiles

    def _insert_profile(self, conn: sqlite3.Connection, document: dict[str, Any]) -> None:
        profile = document["profile"]
        profile_id = profile["profile_id"]
        for key, config in (document.get("fields") or {}).items():
            if "." not in key:
                raise RegistryError(f"Generation profile key must be '<entity>.<field>': {key}")
            entity, field = key.split(".", 1)
            conn.execute(
                "INSERT INTO generation_profiles(profile_id,canonical_id,field_name,generator,params_json,derived_formula,depends_on_json) VALUES(?,?,?,?,?,?,?)",
                (
                    profile_id, self._normalise_canonical_id(entity), field, config.get("generator", ""), _json(config.get("params") or {}),
                    config.get("derived_formula"), _json(config.get("depends_on") or []),
                ),
            )

    def _apply_generation_profiles(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT profile_id, canonical_id, field_name, generator, params_json, derived_formula, depends_on_json FROM generation_profiles ORDER BY profile_id"
        ).fetchall()
        for _, cid, field, generator, params_json, formula, depends_on_json in rows:
            exists = conn.execute("SELECT 1 FROM attributes WHERE canonical_id=? AND name=?", (cid, field)).fetchone()
            if not exists:
                raise RegistryError(f"Generation profile references unknown standards attribute: {cid}.{field}")
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
