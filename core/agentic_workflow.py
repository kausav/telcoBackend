"""Agentic telecom scenario proposal and HITL validation.

Public API design: scenario/propose creates a normal draft; scenario/confirm is the
HITL approval/edit action; scenario/generate executes only confirmed scenarios.
"""
from __future__ import annotations

from typing import Any
import hashlib
import json
import logging

from core.agentic_models import ScenarioImportResponse, ScenarioProposeRequest, ScenarioSchema, ScenarioIntent, GeneratedSchemaField
from core.conversation_store import append_message, ensure_conversation
from core.dynamic_scenarios import new_draft_id, save_draft
from core.errors import LLMUpstreamError
from core.runtime_cache import get_proposal, set_proposal
from core.scenario_variable_store import get_recommended, get_user_variables, save_proposal
from core.json_domain_policy import is_json_grounded_domain, source_manifest
from core.low_balance_variable_policy import validate_db_definition

logger = logging.getLogger(__name__)
from agents.intent_agent import GeminiIntentAgent
from agents.schema_compiler import SchemaCompiler
from core.industry_source_store import normalize_industry_key, semantic_exclusion_aliases, catalog_for_request


class AgenticSchemaWorkflow:
    """Build a standards-backed draft and validate HITL edits without an extra API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._intent_agent: GeminiIntentAgent | None = None
        self.compiler = SchemaCompiler()

    def _get_intent_agent(self) -> GeminiIntentAgent:
        if self._intent_agent is None:
            self._intent_agent = GeminiIntentAgent(api_key=self._api_key)
        return self._intent_agent

    @staticmethod
    def _infer_type_of_data(requested: str | None, schema: ScenarioSchema) -> str:
        if requested in {"transactional", "aggregational"}:
            return requested
        # Prefer the source-backed field grain rather than an industry-specific marker list.
        return "transactional" if any(str(field.grain or "").lower() == "entity" for field in schema.fields) else "aggregational"

    @staticmethod
    def _entity_key(requested: str | None, field_names: list[str], type_of_data: str) -> str | None:
        if type_of_data != "transactional":
            return None
        names = set(field_names)
        if requested:
            match = next((name for name in names if name.lower() == requested.lower()), None)
            if match:
                return match
        preferred = ("subscriber_id", "customer_id", "account_id", "prepaid_account_id", "user_id", "entity_id", "id")
        return next((name for name in preferred if name in names), next(iter(field_names), None))

    @staticmethod
    def _schema_to_variables(
        schema: ScenarioSchema,
        raw_persisted_by_name: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        variables: list[dict[str, Any]] = []
        field_order: list[str] = []
        persisted = raw_persisted_by_name or {}
        for field in schema.fields:
            key = field.name.strip().casefold()
            if key in persisted:
                # DB-owned variable definitions are returned exactly as stored.
                item = dict(persisted[key])
            else:
                item = field.model_dump()
                item.pop("provenance", None)
            variables.append(item)
            field_order.append(field.name)
        return variables, field_order

    @staticmethod
    def _variable_name_keys(variables: list[dict[str, Any]]) -> tuple[str, ...]:
        names = {
            str(item.get("name") or "").strip().casefold()
            for item in (variables or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        return tuple(sorted(names))

    @classmethod
    def _merge_db_variable_sources(
        cls,
        recommended: list[dict[str, Any]],
        user_selected: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge persisted DB variables by exact or conservative semantic identity.

        USER_SELECTED overrides DB_RECOMMENDED. A semantically equivalent DB field is represented
        once in the executable schema; the stored definition itself is never rewritten.
        """
        ordered: list[dict[str, Any]] = []
        source_by_index: list[str] = []
        by_name: dict[str, int] = {}
        by_alias: dict[str, set[int]] = {}

        def rebuild_alias_index() -> None:
            by_alias.clear()
            for index, item in enumerate(ordered):
                for alias in semantic_exclusion_aliases(item):
                    by_alias.setdefault(alias, set()).add(index)

        for source_name, items in (("DB_RECOMMENDED", recommended or []), ("USER_SELECTED", user_selected or [])):
            for raw in items:
                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid MongoDB scenario variable: {raw!r}")
                validate_db_definition(raw)
                name = str(raw.get("name") or "").strip()
                if not name:
                    raise ValueError("MongoDB scenario variable is missing its name")
                key = name.casefold()
                item = dict(raw)
                exact_index = by_name.get(key)
                alias_matches = sorted({index for alias in semantic_exclusion_aliases(item) for index in by_alias.get(alias, set())})
                match_indices = [exact_index] if exact_index is not None else alias_matches

                if match_indices:
                    index = match_indices[0]
                    # If one DB definition is semantically equivalent to multiple already-stored
                    # definitions, the input itself is ambiguous. Failing closed is safer than
                    # letting set/hash iteration choose a different replacement on each process.
                    if len(set(match_indices)) > 1:
                        existing_names = sorted(str(ordered[i].get("name") or "") for i in match_indices)
                        raise ValueError(
                            f"MongoDB scenario variable '{name}' semantically duplicates multiple existing variables: "
                            + ", ".join(existing_names)
                        )
                    old_key = str(ordered[index].get("name") or "").strip().casefold()
                    ordered[index] = item
                    if old_key and old_key != key:
                        by_name.pop(old_key, None)
                    by_name[key] = index
                    source_by_index[index] = source_name
                else:
                    index = len(ordered)
                    ordered.append(item)
                    source_by_index.append(source_name)
                    by_name[key] = index
                rebuild_alias_index()
        return ordered

    @classmethod
    def _cache_key(
        cls,
        req: ScenarioProposeRequest,
        db_variables: list[dict[str, Any]],
        source_sources: list[dict[str, Any]] | None = None,
    ) -> tuple:
        # The LLM output depends on the complete DB definitions because it must suppress
        # semantic duplicates, not merely exact-name duplicates. Hash the full definitions
        # deterministically so a changed DB definition cannot reuse a stale proposal.
        canonical_db = json.dumps(db_variables or [], sort_keys=True, separators=(",", ":"), default=str)
        db_fingerprint = hashlib.sha256(canonical_db.encode("utf-8")).hexdigest()
        source_fingerprint = hashlib.sha256(
            json.dumps(source_sources or source_manifest(req.industry_type, req.domain), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return (
            "agentic_proposal_v31_generic_source_semantic_boundary",
            req.industry_type.strip().lower(),
            req.country.strip().upper(),
            req.domain.strip().lower(),
            req.scenario_type.strip().lower(),
            req.type_of_data,
            req.use_case.strip().lower(),
            " ".join(req.business_scenario.split()).strip().lower(),
            db_fingerprint,
            source_fingerprint,
        )

    @staticmethod
    def _merge_persisted_variables(
        schema: ScenarioSchema,
        recommended: list[dict[str, Any]],
        user_selected: list[dict[str, Any]],
    ) -> tuple[ScenarioSchema, dict[str, str], dict[str, dict[str, Any]]]:
        """Merge DB-controlled variables with semantic precedence over source aliases.

        Precedence: USER_SELECTED > DB_RECOMMENDED > source JSON. A DB definition is never rewritten.
        When a DB variable represents the same business concept as a source field, the DB definition
        replaces that source field in-place so the final schema cannot contain two columns that model
        the same concept under different names.
        """
        source_by_name: dict[str, str] = {}
        raw_persisted_by_name: dict[str, dict[str, Any]] = {}
        ordered: list[GeneratedSchemaField] = []
        by_name: dict[str, GeneratedSchemaField] = {}
        by_semantic_alias: dict[str, str] = {}

        def rebuild_semantic_index() -> None:
            by_semantic_alias.clear()
            for field in ordered:
                key = field.name.strip().casefold()
                data = field.model_dump()
                prov = field.provenance or {}
                if prov.get("source_json_semantic_key"):
                    data["semantic_key"] = prov.get("source_json_semantic_key")
                for alias in semantic_exclusion_aliases(data):
                    by_semantic_alias.setdefault(alias, key)

        def add_schema_fields(items: list[GeneratedSchemaField]) -> None:
            for field in items or []:
                key = field.name.strip().casefold()
                if not key or key in by_name:
                    continue
                ordered.append(field)
                by_name[key] = field
            rebuild_semantic_index()
            for field in ordered:
                key = field.name.strip().casefold()
                generated_from = str(field.provenance.get("generated_from") or "").strip().lower()
                source_by_name.setdefault(key, "MONGODB_JSON" if generated_from == "mongodb_json_source" else "LLM_GENERATED")

        def add_persisted(items: list[dict[str, Any]], source: str) -> None:
            for raw in items or []:
                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid MongoDB scenario variable: {raw!r}")
                validate_db_definition(raw)
                key = str(raw.get("name") or "").strip().casefold()
                if not key:
                    raise ValueError("MongoDB scenario variable is missing its name")
                field = GeneratedSchemaField.model_validate(raw)
                aliases = semantic_exclusion_aliases(raw)
                if key in by_name:
                    replacement_key = key
                else:
                    semantic_matches = sorted({by_semantic_alias[a] for a in aliases if a in by_semantic_alias})
                    if len(semantic_matches) > 1:
                        raise ValueError(
                            f"MongoDB scenario variable '{key}' semantically overlaps multiple source variables: "
                            + ", ".join(sorted(semantic_matches))
                        )
                    replacement_key = semantic_matches[0] if semantic_matches else None
                if replacement_key and replacement_key in by_name:
                    idx = next(i for i, existing in enumerate(ordered) if existing.name.strip().casefold() == replacement_key)
                    old_key = ordered[idx].name.strip().casefold()
                    ordered[idx] = field
                    by_name.pop(old_key, None)
                    source_by_name.pop(old_key, None)
                    by_name[key] = field
                    source_by_name[key] = source
                    raw_persisted_by_name[key] = dict(raw)
                else:
                    ordered.append(field)
                    by_name[key] = field
                    source_by_name[key] = source
                    raw_persisted_by_name[key] = dict(raw)
                rebuild_semantic_index()

        add_schema_fields(list(schema.fields))
        add_persisted(recommended, "DB_RECOMMENDED")
        add_persisted(user_selected, "USER_SELECTED")
        merged = schema.model_copy(update={"fields": ordered})
        return merged, source_by_name, raw_persisted_by_name

    def propose(self, req: ScenarioProposeRequest) -> ScenarioImportResponse:
        prompt = req.business_scenario.strip()
        requested_scenario_id = req.requested_scenario_id.strip()
        industry_key = normalize_industry_key(req.industry_type)
        # Resolve the complete source catalog once. This supplies both the source fingerprint and
        # compiler input, avoiding repeated MongoDB queries/extraction work on the same proposal.
        source_catalog = catalog_for_request(req.industry_type, req.domain)
        source_sources = list(source_catalog.get("sources") or [])
        json_grounded = bool(source_sources)
        # Persisted MongoDB variables remain authoritative overlays for every domain. The source
        # selector receives their definitions up front so its breadth budget is spent only on new
        # source concepts rather than fields that will later be replaced by DB-owned variables.
        recommended = get_recommended(requested_scenario_id, 1)
        user_selected = (
            get_user_variables(req.user_id.strip(), requested_scenario_id, 1)
            if req.user_id and req.user_id.strip()
            else []
        )
        db_variables = self._merge_db_variable_sources(recommended, user_selected)
        if not json_grounded and not db_variables:
            raise ValueError(
                f"No active JSON source documents and no persisted MongoDB scenario variables exist for industryType='{req.industry_type}', domain='{req.domain}'. "
                "Upload at least one industry-standard JSON or save scenario variables before proposing/generating this scenario."
            )
        grounding_requirement = (
            "JSON-SOURCE REQUIREMENT: use only active MongoDB source documents for this exact industryType/domain pair. "
            "The source catalog is the only executable standards vocabulary; do not use external standards, URLs, static registries, profiles, templates, examples, memory, or generic industry knowledge. "
            "Every non-DB executable variable must be an exact scalar leaf from that catalog. "
            if json_grounded else
            "SCENARIO-VARIABLES REQUIREMENT: there is no registered JSON source for this exact industryType/domain pair. "
            "Use only the persisted MongoDB scenario variables supplied to you. Do not invent, rename, alias, or derive executable variable names. "
        )
        protected_name_set = set(self._variable_name_keys(db_variables))
        # Exact names remain a cheap provider-side exclusion. Semantic aliases are handled by the
        # deterministic source selector using the full DB definitions, so a source equivalent such
        # as `topup_balance.amount.amount` cannot consume breadth budget when DB already owns `recharge_amount`.
        protected_names = tuple(sorted(protected_name_set))
        agent_prompt = (
            f"Industry: {req.industry_type}\n"
            f"Business domain: {req.domain}\n"
            f"Use case: {req.use_case}\n"
            f"Scenario type: {req.scenario_type}\n"
            f"Data type: {req.type_of_data}\n"
            f"Country: {req.country}\n"
            f"Business scenario: {prompt}\n\n"
            "Variable-design requirement: propose a broad fresh semantic variable set without an artificial minimum count. Maximize DISTINCT analytical coverage rather than raw field count. "
            + grounding_requirement + " "
            "Do not copy reference CSV variable names. Include every variable genuinely needed to represent the business scenario, but do not add API href/referredType/reference metadata, display-only name/description fields, or semantic aliases solely to increase width. Prefer one canonical variable per business concept. "
            "Scenario type is a hard semantic signal: two requests with different scenarioType values must not be forced into the same variable set. "
            "Select variables that make the behavioral difference observable; do not use requestedScenarioId to achieve that difference. "
            "Use ALL request inputs except requestedScenarioId and entityKey as semantic/context signals: scenarioType, industryType, domain, "
            "businessScenario, typeOfData, country, and useCase must materially constrain the variable set, field parameters, scope, "
            "and generation behavior. Return only complementary source-backed variables from the supplied catalog; persisted MongoDB variables are supplied separately and must never be recreated or renamed."
        )

        cache_key = self._cache_key(req, db_variables, source_sources)
        cached = get_proposal(cache_key)
        if cached is not None:
            intent = ScenarioIntent.model_validate(cached["intent"])
            schema = ScenarioSchema.model_validate(cached["schema"])
            cid = ensure_conversation(requested_scenario_id, req.user_id, requested_scenario_id)
            logger.info("[AgenticSchemaWorkflow] Proposal cache hit; Gemini skipped.")
        else:
            cid = ensure_conversation(requested_scenario_id, req.user_id, requested_scenario_id)
            append_message(cid, "user", agent_prompt, requested_scenario_id=requested_scenario_id)
            intent = self._get_intent_agent().run(
                agent_prompt,
                country=req.country,
                industry_type=industry_key,
                domain_query=req.domain,
                excluded_variable_names=list(protected_names),
                persisted_variables=db_variables,
                source_catalog=source_catalog,
            )
            intent.industry_type = industry_key
            intent.domain = req.domain
            intent.subdomain = "unknown"
            if req.country:
                intent.country = req.country
            intent.scenario_type = req.scenario_type
            intent.type_of_data = req.type_of_data
            intent.entity_key = req.entity_key or ""
            intent.use_case = req.use_case

            # Defensive post-LLM pruning: exact persisted field names are already covered
            # by Mongo and must not enter compilation even if the provider ignores the
            # exclusion instruction. This reduces downstream work and prevents duplicate
            # candidate processing without weakening the final DB-overlay authority.
            if protected_names:
                intent.candidate_variables = [
                    idea
                    for idea in intent.candidate_variables
                    if str(idea.name or "").strip().casefold() not in protected_name_set
                ]

            if json_grounded:
                schema = self.compiler.compile(
                    intent,
                    max_variables=None,
                    domain_query=req.domain,
                    entity_key=req.entity_key,
                    industry_type=req.industry_type,
                    scenario_type=req.scenario_type,
                    type_of_data=req.type_of_data,
                    use_case=req.use_case,
                    business_scenario=req.business_scenario,
                    business_response="",
                    expected_outcome="",
                    country=req.country,
                    excluded_field_names=list(protected_names),
                    external_variable_names=set(protected_name_set),
                    external_variable_definitions=db_variables,
                )
            else:
                # No JSON source exists for this pair. The only executable definitions allowed
                # are the persisted MongoDB scenario variables; build the schema directly from
                # their stored contracts and never ask a registry/static source for replacements.
                db_fields = [GeneratedSchemaField.model_validate(dict(item)) for item in db_variables]
                names = [field.name for field in db_fields]
                resolved_entity_key = req.entity_key or (names[0] if req.type_of_data == "transactional" and names else None)
                schema = ScenarioSchema(
                    domain=req.domain,
                    subdomain="unknown",
                    applicable_standards=[],
                    entities=[],
                    relationships=[],
                    fields=db_fields,
                    hard_constraints=[
                        "Executable variables are restricted to persisted MongoDB scenario_variables because no industry/domain JSON source is registered for this pair.",
                        "The LLM is a reviewer only and cannot create, rename, alias, or change executable scenario variables.",
                    ] + ([f"'{resolved_entity_key}' is the authoritative entity key for transactional grouping."] if resolved_entity_key else []),
                    unresolved_items=[] if db_fields else ["No persisted MongoDB scenario variables are available."],
                    warnings=["No industry-standard JSON source is registered for this industry/domain pair; MongoDB scenario variables are the complete executable source."],
                )
            set_proposal(cache_key, {"intent": intent.model_dump(), "schema": schema.model_dump()})

        schema, variable_sources, raw_persisted_by_name = self._merge_persisted_variables(
            schema,
            recommended,
            user_selected,
        )

        unresolved_questions = self.compiler.approval_questions(intent, schema)
        variables, field_order = self._schema_to_variables(schema, raw_persisted_by_name)
        for variable in variables:
            key = str(variable.get("name") or "").strip().lower()
            persisted_source = variable_sources.get(key)
            if persisted_source:
                variable["source"] = persisted_source
                continue
            if json_grounded:
                variable["source"] = "MONGODB_JSON"
                continue
            variable["source"] = "LLM_GENERATED"
        type_of_data = self._infer_type_of_data(req.type_of_data, schema)
        entity_key = self._entity_key(req.entity_key, field_order, type_of_data)
        variable_source_ids = {}
        for field in schema.fields:
            provenance = field.provenance if isinstance(field.provenance, dict) else {}
            source_id = str(provenance.get("source_json_id") or "").strip()
            if source_id and field.name.strip():
                variable_source_ids[field.name.strip().casefold()] = source_id

        draft_id = new_draft_id()
        description = req.business_scenario
        draft = {
            "label": req.requested_scenario_id,
            "journey": req.domain,
            "description": description,
            "variables": variables,
            "field_order": field_order,
            "domain": req.domain,
            "business_scenario": req.business_scenario,
            "business_response": None,
            "expected_outcome": None,
            "use_case": req.use_case,
            "scenario_id": req.requested_scenario_id,
            "requested_scenario_id": req.requested_scenario_id,
            "scenario_type": req.scenario_type or "agentic",
            "industry_type": industry_key,
            "country": intent.country or req.country,
            "type_of_data": type_of_data,
            "entity_key": entity_key,
            "records_per_user": 10,
            "agentic": True,
            "conversation_id": cid,
            "intent": intent.model_dump(),
            "schema": schema.model_dump(),
            "approval_questions": unresolved_questions,
            "source_policy": "mongodb_industry_source_documents" if json_grounded else "scenario_variables",
            "source_documents": source_sources if json_grounded else [],
            "variable_sources": variable_sources,
            "variable_source_ids": variable_source_ids,
            "db_variable_names": sorted(raw_persisted_by_name.keys()),
            "db_variable_definitions": raw_persisted_by_name,
        }
        save_draft(draft_id, draft)
        save_proposal(
            request_id=draft_id,
            user_id=req.user_id.strip() if req.user_id else None,
            requested_scenario_id=requested_scenario_id,
            scenario_version=1,
            payload={
                "scenario_id": req.requested_scenario_id,
                "requested_scenario_id": requested_scenario_id,
                "variables": variables,
                "field_order": field_order,
                "intent": intent.model_dump(),
                "variable_sources": variable_sources,
            },
        )
        append_message(cid, "assistant", json.dumps({"intent": intent.model_dump(), "action": "schema_proposed"}, sort_keys=True), requested_scenario_id=requested_scenario_id)
        return ScenarioImportResponse(
            success=True,
            draft_id=draft_id,
            scenario_id=req.requested_scenario_id,
            requested_scenario_id=req.requested_scenario_id,
            journey=req.domain,
            description=description,
            variables=variables,
            field_order=field_order,
            typeOfData=type_of_data,
            entityKey=entity_key,
            variableSources=variable_sources,
        )

    @staticmethod
    def validate_hitl_changes(
        draft: dict[str, Any],
        add: list[dict[str, Any]],
        edit: list[Any],
        delete: list[str],
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, str], set[str]]:
        """Apply HITL edits while preserving the authoritative source/DB boundaries."""
        schema = ScenarioSchema.model_validate(draft["schema"])
        draft_variables = [
            dict(item)
            for item in (draft.get("variables") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        draft_source_by_name = {
            str(name).strip().casefold(): str(source).strip().upper()
            for name, source in (draft.get("variable_sources") or {}).items()
            if str(name).strip() and str(source).strip()
        }
        draft_db_names = {
            str(name).strip().casefold()
            for name in (draft.get("db_variable_names") or [])
            if str(name).strip()
        }
        draft_db_names.update(
            str(name).strip().casefold()
            for name in (draft.get("db_variable_definitions") or {}).keys()
            if str(name).strip()
        )

        # The proposal is already canonicalized against its source catalog and DB overlays. Do not
        # run the legacy Low Balance reconciliation layer here; it would reintroduce telecom-specific
        # naming and could discard legitimate source-backed fields from other scenarios.

        # Concept labels are soft hints. Only genuinely executable unresolved requirements block confirmation.
        # Reconcile compiler-stage diagnostics against the already merged executable variable set
        # before evaluating anything as a blocking confirmation requirement.
        external_db_names = set(draft_db_names)
        proposed_names = {
            str(item.get("name") or "").strip().casefold()
            for item in draft_variables
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        proposed_names.update(field.name.strip().casefold() for field in schema.fields if field.name.strip())
        blocking_unresolved = []
        for item in schema.unresolved_items:
            text = str(item).strip()
            lower = text.casefold()
            if lower.startswith("unknown concept ") or lower.startswith("unknown requested concept "):
                continue
            if "entity key '" in lower or "requested entity key '" in lower:
                marker = lower.split("entity key '", 1)[-1]
                candidate = marker.split("'", 1)[0].strip().casefold()
                if candidate in external_db_names or candidate in proposed_names or candidate == str(draft.get("entity_key") or "").strip().casefold():
                    continue
            blocking_unresolved.append(text)
        if blocking_unresolved:
            raise ValueError(
                "Cannot confirm an agentic draft with unresolved executable requirements: "
                + "; ".join(blocking_unresolved)
            )

        fields_by_name = {field.name: field for field in schema.fields}
        mandatory_entity_fields = {str(draft.get("entity_key") or "").strip()} - {""}
        delete_set = {str(name).strip() for name in delete if str(name).strip()}
        for name in delete_set:
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot delete unknown agentic field '{name}'")
            if name in mandatory_entity_fields:
                raise ValueError(f"HITL cannot delete the requested entity key field '{name}'")
            if fields_by_name[name].required:
                raise ValueError(f"HITL cannot delete required field '{name}'")

        for item in add:
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError("Agentic HITL additions require an existing field name")
            if name not in fields_by_name:
                raise ValueError(f"Agentic HITL cannot add field '{name}' because it is not in the proposed MongoDB source-backed schema")
            raise ValueError(f"Agentic HITL cannot add duplicate field '{name}'; revise existing fields instead")

        allowed_override_keys = {"nullable", "description", "params"}
        applied_edits: dict[str, dict[str, Any]] = {}
        for item in edit:
            name = item.name if hasattr(item, "name") else str(item.get("name") or "")
            changes = item.changes if hasattr(item, "changes") else dict(item.get("changes") or {})
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot edit unknown agentic field '{name}'")
            key = name.strip().casefold()
            if draft_source_by_name.get(key) in {"DB_RECOMMENDED", "USER_SELECTED"} and changes:
                raise ValueError(
                    f"MongoDB variable '{name}' is authoritative and cannot be renamed or edited; "
                    "change its DB definition instead"
                )
            unknown = sorted(set(changes) - allowed_override_keys)
            if unknown:
                raise ValueError(f"Agentic HITL cannot change executable schema semantics for '{name}': {unknown}")
            target = fields_by_name[name]
            if "nullable" in changes and not isinstance(changes["nullable"], bool):
                raise ValueError(f"HITL nullable override for '{name}' must be boolean")
            if "description" in changes:
                target.description = str(changes["description"])[:1000]
            if "nullable" in changes:
                target.nullable = changes["nullable"]
            if "params" in changes:
                params = changes["params"]
                if not isinstance(params, dict):
                    raise ValueError(f"HITL params override for '{name}' must be an object")
                unknown_params = sorted(set(params) - set(target.params or {}))
                if unknown_params:
                    raise ValueError(f"HITL cannot introduce generation parameters for '{name}': {unknown_params}")
                target.params = {**target.params, **params}
            applied_edits[key] = dict(changes)

        remaining = [field for field in schema.fields if field.name not in delete_set]
        remaining_names = {field.name for field in remaining}
        original_names = {field.name for field in schema.fields}
        for field in remaining:
            missing = sorted(
                dep for dep in field.depends_on
                if dep in original_names and dep not in remaining_names
            )
            if missing:
                raise ValueError(
                    f"HITL deletion would break field dependencies for '{field.name}': {missing}"
                )

        draft_raw_by_name = {
            str(item.get("name") or "").strip().casefold(): dict(item)
            for item in draft_variables
        }
        variables: list[dict[str, Any]] = []
        field_order: list[str] = []
        for field in remaining:
            key = field.name.strip().casefold()
            if draft_source_by_name.get(key) in {"DB_RECOMMENDED", "USER_SELECTED"} and key in draft_raw_by_name:
                data = dict(draft_raw_by_name[key])
                changes = applied_edits.get(key) or {}
                if "description" in changes:
                    data["description"] = field.description
                if "nullable" in changes:
                    data["nullable"] = field.nullable
                if "params" in changes:
                    data["params"] = dict(field.params or {})
            else:
                data = field.model_dump()
                data.pop("provenance", None)
            variables.append(data)
            field_order.append(field.name)

        if draft.get("type_of_data") == "transactional" and draft.get("entity_key") not in field_order:
            raise ValueError("HITL changes would remove the transactional entity key")
        return variables, field_order, draft_source_by_name, draft_db_names

_WORKFLOW_SINGLETONS: dict[str, AgenticSchemaWorkflow] = {}


def get_agentic_workflow(api_key: str | None = None) -> AgenticSchemaWorkflow:
    """Reuse the Gemini intent client across proposal requests.

    The cache key is the explicit API key (or a process-local default), never requestedScenarioId.
    This removes repeated Gemini client/provider construction from /scenario/propose.
    """
    key = api_key or "__default__"
    workflow = _WORKFLOW_SINGLETONS.get(key)
    if workflow is None:
        workflow = AgenticSchemaWorkflow(api_key=api_key)
        _WORKFLOW_SINGLETONS[key] = workflow
    return workflow
