"""Standards-backed telecom registry persisted in MongoDB."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib, json, logging, os
from threading import RLock

from config.runtime import TELECOM_STANDARDS_DIR, TELECOM_STANDARDS_MANIFEST, TELECOM_STANDARDS_CACHE_DIR, TELECOM_PROFILES_DIR, OFFICIAL_STANDARDS_SYNC, OFFICIAL_STANDARDS_TIMEOUT_SEC, OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB
from core.official_standards import sync_official_standards, OfficialStandardsError
from core.runtime_lock import RuntimeFileLock, RuntimeLockError
from models.registry_entity import RegistryEntityModel
from models.registry_standard import RegistryStandardModel
from models.registry_meta import RegistryMetaModel

DEFAULT_STANDARDS_DIR=TELECOM_STANDARDS_DIR
DEFAULT_PROFILES_DIR=TELECOM_PROFILES_DIR
REGISTRY_SCHEMA_VERSION="8-mongo"
logger=logging.getLogger(__name__)

@dataclass(frozen=True)
class AttributeDef:
    name: str; dtype: str; required: bool=False; nullable: bool=False; description: str=""; enum_values: tuple[str,...]=(); generator: str=""; params: dict[str,Any]|None=None; derived_formula: str|None=None; depends_on: tuple[str,...]=()
@dataclass(frozen=True)
class RelationshipDef:
    target: str; relation: str; cardinality: str; required: bool=False; description: str=""
@dataclass(frozen=True)
class EntityDef:
    canonical_id: str; name: str; aliases: tuple[str,...]; domain: str; description: str; sources: tuple[dict[str,Any],...]; attributes: tuple[AttributeDef,...]; relationships: tuple[RelationshipDef,...]
class RegistryError(RuntimeError): pass

def _normalise(value:str)->str: return " ".join(str(value or "").strip().lower().replace("_"," ").split())
def _canonical(value:str)->str: return _normalise(value).replace(" ","_")
def _canonical_file_list(directory:Path)->list[Path]: return sorted(p for p in directory.rglob("*.json") if p.is_file()) if directory.exists() else []
def _json_hash(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def registry_fingerprint(standards_dir:Path, profiles_dir:Path)->str:
    h=hashlib.sha256()
    for label,d in (("standards",standards_dir),("profiles",profiles_dir)):
        for p in _canonical_file_list(d):
            h.update(f"{label}/{p.relative_to(d).as_posix()}".encode()); h.update(p.read_bytes())
    h.update(REGISTRY_SCHEMA_VERSION.encode()); return h.hexdigest()

class TelecomRegistry:
    def __init__(self, db_name=None, standards_dir=None, profiles_dir=None, auto_bootstrap=True):
        self.entities=RegistryEntityModel.collection
        self.standards=RegistryStandardModel.collection
        self.meta=RegistryMetaModel.collection
        self.standards_dir=Path(standards_dir or os.getenv("REGISTRY_STANDARDS_DIR") or DEFAULT_STANDARDS_DIR)
        self.profiles_dir=Path(profiles_dir or os.getenv("REGISTRY_PROFILES_DIR") or DEFAULT_PROFILES_DIR)
        self.manifest_path=Path(os.getenv("OFFICIAL_STANDARDS_MANIFEST") or TELECOM_STANDARDS_MANIFEST)
        self.official_cache_dir=Path(os.getenv("TELECOM_STANDARDS_CACHE_DIR") or TELECOM_STANDARDS_CACHE_DIR)
        self._explicit_standards_dir=standards_dir is not None or bool(os.getenv("REGISTRY_STANDARDS_DIR"))
        self._cache_lock=RLock(); self._entity_cache={}; self._catalog_cache={}; self._search_cache={}
        RegistryEntityModel.ensure_indexes()
        RegistryStandardModel.ensure_indexes()
        RegistryMetaModel.ensure_indexes()
        if auto_bootstrap: self.ensure_current()

    def ensure_current(self):
        lock=self.official_cache_dir/"raw"/".official-standards.sync.lock"
        try:
            with RuntimeFileLock(lock):
                if not self._explicit_standards_dir and OFFICIAL_STANDARDS_SYNC!="disabled":
                    try: sync_official_standards(self.manifest_path,self.official_cache_dir/"raw",self.standards_dir,timeout=OFFICIAL_STANDARDS_TIMEOUT_SEC,max_download_mb=OFFICIAL_STANDARDS_MAX_DOWNLOAD_MB,acquire_lock=False)
                    except OfficialStandardsError as exc:
                        if not _canonical_file_list(self.standards_dir): raise RegistryError(f"Official telecom standards could not be synchronized and no cached model is available: {exc}") from exc
                if not _canonical_file_list(self.standards_dir): raise RegistryError(f"No official standards model artifacts found in {self.standards_dir}")
                fp=registry_fingerprint(self.standards_dir,self.profiles_dir)
                m={x["key"]:x["value"] for x in self.meta.find({}, {"key":1,"value":1,"_id":0})}
                if m.get("fingerprint")==fp and m.get("schema_version")==REGISTRY_SCHEMA_VERSION and self.entities.count_documents({},limit=1): return
                RegistryBuilder(self,self.standards_dir,self.profiles_dir).rebuild(fp)
        except RuntimeLockError as exc: raise RegistryError(str(exc)) from exc

    def health(self):
        m={x["key"]:x["value"] for x in self.meta.find({}, {"key":1,"value":1,"_id":0})}
        return {"healthy":self.entities.count_documents({},limit=1)>0,"standards":self.standards.count_documents({}),"entities":self.entities.count_documents({}),"attributes":sum(len(x.get("attributes",[])) for x in self.entities.find({}, {"attributes":1,"_id":0})),"relationships":sum(len(x.get("relationships",[])) for x in self.entities.find({}, {"relationships":1,"_id":0})),"provenance_records":sum(len(x.get("sources",[])) for x in self.entities.find({}, {"sources":1,"_id":0})),"schema_version":m.get("schema_version"),"fingerprint":m.get("fingerprint")}

    def _to_entity(self, doc):
        attrs=tuple(AttributeDef(a["name"],a.get("dtype","string"),bool(a.get("required")),bool(a.get("nullable")),a.get("description","") or "",tuple(a.get("enum_values",[])),a.get("generator","") or "",dict(a.get("params") or {}),a.get("derived_formula"),tuple(a.get("depends_on",[]))) for a in doc.get("attributes",[]))
        rels=tuple(RelationshipDef(r["target"],r["relation"],r["cardinality"],bool(r.get("required")),r.get("description","") or "") for r in doc.get("relationships",[]))
        return EntityDef(doc["canonical_id"],doc["name"],tuple(doc.get("aliases",[])),doc.get("domain","telecom"),doc.get("description","") or "",tuple(doc.get("sources",[])),attrs,rels)

    def get_entity(self, entity_id:str)->EntityDef:
        key=_canonical(entity_id)
        with self._cache_lock:
            if key in self._entity_cache:return self._entity_cache[key]
        doc=self.entities.find_one({"canonical_id":key})
        if not doc: raise KeyError(entity_id)
        e=self._to_entity(doc)
        with self._cache_lock:self._entity_cache[key]=e
        return e
    def resolve_entity(self,value:str):
        key=_normalise(value); doc=self.entities.find_one({"aliases":key},{"canonical_id":1}) or self.entities.find_one({"canonical_id":key.replace(" ","_")},{"canonical_id":1})
        return self.get_entity(doc["canonical_id"]) if doc else None
    def entity_exists(self,entity_id):
        try:self.get_entity(entity_id);return True
        except KeyError:return False
    def entity_dict(self,entity_id):
        e=self.get_entity(entity_id); return {"canonical_id":e.canonical_id,"name":e.name,"aliases":list(e.aliases),"domain":e.domain,"description":e.description,"sources":[dict(x) for x in e.sources],"attributes":[{**asdict(a),"enum_values":list(a.enum_values),"params":dict(a.params or {}),"depends_on":list(a.depends_on)} for a in e.attributes],"relationships":[asdict(r) for r in e.relationships]}
    def entities_with_attribute(self,attribute_name):
        n=str(attribute_name or "").strip().lower(); return [self.get_entity(x["canonical_id"]) for x in self.entities.find({"attributes.name":{"$regex":f"^{__import__('re').escape(n)}$","$options":"i"}},{"canonical_id":1}).sort("canonical_id",1)] if n else []
    def related_entities(self,entity_id):
        key=_canonical(entity_id); ids=set(x["canonical_id"] for x in self.entities.find({"relationships.target":key},{"canonical_id":1})); doc=self.entities.find_one({"canonical_id":key},{"relationships.target":1}) or {}; ids.update(r["target"] for r in doc.get("relationships",[])); return [self.get_entity(x) for x in sorted(ids) if self.entity_exists(x)]
    def catalog_summary(self,domain=None,limit=200):
        q={"domain":domain} if domain else {}; cur=self.entities.find(q,{"canonical_id":1,"name":1,"domain":1,"description":1,"aliases":1,"sources":1}).sort("canonical_id",1); 
        if limit is not None: cur=cur.limit(max(1,int(limit)))
        return [{"canonical_id":x["canonical_id"],"name":x["name"],"aliases":x.get("aliases",[]),"domain":x.get("domain"),"description":x.get("description",""),"source_standards":sorted({s.get("standard") for s in x.get("sources",[]) if s.get("standard")})} for x in cur]
    def search(self,query,domain=None,limit=None):
        normalized=_normalise(query); tokens=normalized.split(); results=[]
        for x in self.entities.find({"domain":domain} if domain else {},{"canonical_id":1,"name":1,"domain":1,"description":1,"aliases":1,"sources":1}):
            text=_normalise(" ".join([x.get("name",""),x.get("canonical_id",""),x.get("description","")]+x.get("aliases",[])+[s.get("artifact","") for s in x.get("sources",[])])); score=(45 if normalized and normalized in text else 0)+sum(10 for t in tokens if t in text)+(100 if any(_normalise(a)==normalized for a in x.get("aliases",[])) else 0)
            if score>0: results.append((score,x))
        results.sort(key=lambda z:(-z[0],z[1]["canonical_id"])); out=[]
        for _,x in results[:(None if limit is None else max(1,int(limit)))]: out.append({"canonical_id":x["canonical_id"],"name":x["name"],"domain":x.get("domain"),"description":x.get("description",""),"aliases":x.get("aliases",[]),"source_standards":sorted({s.get("standard") for s in x.get("sources",[]) if s.get("standard")})})
        return out
    def standards_for_entities(self,entity_ids):
        grouped={}
        for value in entity_ids:
            e=self.resolve_entity(value)
            if not e: continue
            for s in e.sources:
                st=s.get("standard"); grouped.setdefault(st,{"standard":st,"artifacts":[]})["artifacts"].append({"artifact":s.get("artifact"),"version":s.get("version"),"reference":s.get("reference"),"url":s.get("url")})
        return list(grouped.values())
    def llm_catalog_context(self,query=None):
        standards=[{k:v for k,v in x.items() if k!="_id"} for x in self.standards.find({}).sort("organization",1)]
        ids=[x["canonical_id"] for x in self.search(query,limit=None)] if query else None
        docs=self.entities.find({"canonical_id":{"$in":ids}} if ids else {},{"_id":0})
        entities=[]
        for x in docs: entities.append({"canonical_id":x["canonical_id"],"name":x["name"],"domain":x.get("domain"),"description":x.get("description",""),"aliases":x.get("aliases",[])})
        return {"standards":standards,"source_urls":sorted({s.get("url") for x in self.entities.find({}, {"sources":1,"_id":0}) for s in x.get("sources",[]) if s.get("url")}),"entities":entities,"registry_health":self.health()}

class RegistryBuilder:
    def __init__(self,registry,standards_dir,profiles_dir): self.registry=registry; self.standards_dir=standards_dir; self.profiles_dir=profiles_dir
    @staticmethod
    def _load_profiles(directory):
        profiles=[]
        for path in _canonical_file_list(directory):
            d=json.loads(path.read_text(encoding="utf-8")); profile=d.get("profile") if isinstance(d,dict) else None; targets=d.get("targets") if isinstance(d,dict) else None
            if not isinstance(profile,dict) or not profile.get("profile_id") or not isinstance(targets,list): raise RegistryError(f"Invalid generation profile: {path}")
            profiles.append(d)
        return profiles
    def rebuild(self,fingerprint):
        artifacts=[]
        for path in _canonical_file_list(self.standards_dir):
            d=json.loads(path.read_text(encoding="utf-8"));
            if "artifact" not in d or "entities" not in d: raise RegistryError(f"Invalid normalized standards artifact: {path}")
            artifacts.append((path,d))
        profiles=self._load_profiles(self.profiles_dir); entity_docs={}; standard_docs={}
        for path,d in artifacts:
            a=d["artifact"]; sources=a.get("sources") or []; default_url=sources[0].get("url") if sources else a.get("source_url"); default_page=sources[0].get("source_page") if sources else a.get("source_page"); default_version=sources[0].get("version") if sources else a.get("version")
            standard_docs[a["artifact_id"]]={"artifact_id":a["artifact_id"],"organization":a.get("organization","Unknown"),"title":a.get("title",a["artifact_id"]),"version":a.get("artifact_version",default_version),"status":a.get("status"),"source_kind":a.get("source_kind","official-machine-readable-model"),"source_url":default_url,"source_page":default_page,"source_sha256":_json_hash(path)}
            for e in d["entities"]:
                cid=_canonical(e["canonical_id"])
                if cid in entity_docs: raise RegistryError(f"Duplicate canonical entity '{cid}' across standards artifacts")
                srcs=e.get("sources") or a.get("sources") or []
                attrs=[]
                for i,at in enumerate(e.get("attributes") or []): attrs.append({"name":at["name"],"ordinal":i,"dtype":at.get("dtype","string"),"required":bool(at.get("required")),"nullable":bool(at.get("nullable")),"description":at.get("description","") or "","enum_values":at.get("enum_values") or [],"generator":at.get("generator","") or "","params":at.get("params") or {},"derived_formula":at.get("derived_formula"),"depends_on":at.get("depends_on") or []})
                rels=[{"target":_canonical(r["target"]),"ordinal":i,"relation":r["relation"],"cardinality":r["cardinality"],"required":bool(r.get("required")),"description":r.get("description","") or ""} for i,r in enumerate(e.get("relationships") or [])]
                aliases=sorted({_normalise(x) for x in set(e.get("aliases") or [])|{cid,e["name"]}})
                entity_docs[cid]={"canonical_id":cid,"name":e["name"],"domain":e.get("domain","telecom"),"description":e.get("description","") or "","artifact_id":a["artifact_id"],"aliases":aliases,"sources":[{"standard":s.get("standard",a.get("organization","Unknown")),"artifact":s.get("artifact",a.get("title",a["artifact_id"])),"version":s.get("version") or a.get("artifact_version"),"reference":s.get("reference",a["artifact_id"]),"url":s.get("url") or a.get("source_url"),"source_page":s.get("source_page") or a.get("source_page"),"source_role":s.get("source_role","semantic-source")} for s in srcs],"attributes":attrs,"relationships":rels}
        for profile in profiles:
            pid=profile["profile"]["profile_id"]
            for target in profile.get("targets") or []:
                source_id=_canonical(str(target["source_id"])); model=str(target["model"]).strip(); field=str(target["field"]).strip()
                candidates=[e for e in entity_docs.values() if e["artifact_id"]==source_id and (model=="*" or e["name"].lower()==model.lower())]
                if not candidates:
                    if target.get("required"): raise RegistryError(f"Generation profile '{pid}' target does not exist: {source_id}:{model}.{field}")
                    continue
                for e in candidates:
                    for at in e["attributes"]:
                        if at["name"]==field:
                            at["generator"]=target.get("generator",""); at["params"]=target.get("params") or {}; at["derived_formula"]=target.get("derived_formula"); at["depends_on"]=target.get("depends_on") or []
        ids=set(entity_docs)
        allowed={"","unique_id","reference","msisdn","timestamp","constant","range","weighted_choice","dependent_choice","semantic_event"}
        for cid,e in entity_docs.items():
            for r in e["relationships"]:
                if r["target"] not in ids: raise RegistryError(f"Broken standards relationship: {cid} -> {r['target']}")
            for a in e["attributes"]:
                if a["generator"] not in allowed: raise RegistryError(f"Unsupported generator in registry: {cid}.{a['name']}={a['generator']}")
        self.registry.entities.delete_many({}); self.registry.standards.delete_many({})
        if standard_docs:self.registry.standards.insert_many(list(standard_docs.values()))
        if entity_docs:self.registry.entities.insert_many(list(entity_docs.values()))
        self.registry.meta.update_one({"key":"schema_version"},{"$set":{"key":"schema_version","value":REGISTRY_SCHEMA_VERSION}},upsert=True)
        self.registry.meta.update_one({"key":"fingerprint"},{"$set":{"key":"fingerprint","value":fingerprint}},upsert=True)
        self.registry.meta.update_one({"key":"built_at"},{"$set":{"key":"built_at","value":str(__import__('time').time())}},upsert=True)
        with self.registry._cache_lock:self.registry._entity_cache.clear();self.registry._catalog_cache.clear();self.registry._search_cache.clear()

_default_registry=None
def get_registry():
    global _default_registry
    if _default_registry is None:_default_registry=TelecomRegistry()
    return _default_registry
def catalog_summary():return get_registry().catalog_summary()
def resolve_entity(value):return get_registry().resolve_entity(value)
def get_entity(entity_id):return get_registry().get_entity(entity_id)
def entity_dict(entity_id):return get_registry().entity_dict(entity_id)
