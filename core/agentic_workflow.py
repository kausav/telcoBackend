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
from core.telecom_registry import TelecomRegistry, get_registry
from core.errors import LLMUpstreamError
from core.runtime_cache import get_proposal, set_proposal
from core.scenario_variable_store import get_recommended, get_user_variables, save_proposal
from core.json_domain_policy import is_json_grounded_domain, source_manifest
from core.low_balance_variable_policy import (
    dedupe_db_variable_sources,
    dedupe_schema_fields_against_db,
    validate_db_definition,
    validate_low_balance_variable_sources,
    validate_low_balance_required_identity_sources,
    filter_incompatible_low_balance_db_customer_overrides,
    reconcile_low_balance_schema,
    reconcile_low_balance_executable_variables,
    semantic_signature,
    official_catalog_by_name,
)

logger = logging.getLogger(__name__)
from agents.intent_agent import GeminiIntentAgent
from agents.schema_compiler import SchemaCompiler
from config.industry_profiles import match_industry_key


class AgenticSchemaWorkflow:
    """Build a standards-backed draft and validate HITL edits without an extra API."""

    def __init__(self, api_key: str | None = None, registry: TelecomRegistry | None = None):
        self.registry = registry or get_registry()
        self._api_key = api_key
        self._intent_agent: GeminiIntentAgent | None = None
        self.compiler = SchemaCompiler(self.registry)

    def _get_intent_agent(self) -> GeminiIntentAgent:
        if self._intent_agent is None:
            self._intent_agent = GeminiIntentAgent(api_key=self._api_key, registry=self.registry)
        return self._intent_agent

    @staticmethod
    def _infer_type_of_data(requested: str | None, schema: ScenarioSchema) -> str:
        if requested in {"transactional", "aggregational"}:
            return requested
        entity_ids = {entity.canonical_id for entity in schema.entities}
        transactional_markers = {
            "subscriber", "customer", "customer_account", "prepaid_account",
            "recharge", "usage_event", "charging_event",
        }
        return "transactional" if entity_ids & transactional_markers else "aggregational"

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
        """Return DB variables with user selection overriding an exact-name recommendation.

        Validation is strict and non-mutating: the stored Mongo definition is used unchanged
        after it has been checked for executability.
        """
        ordered: list[dict[str, Any]] = []
        positions: dict[str, int] = {}
        for raw in [*(recommended or []), *(user_selected or [])]:
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid MongoDB scenario variable: {raw!r}")
            validate_db_definition(raw)
            name = str(raw.get("name") or "").strip()
            if not name:
                raise ValueError("MongoDB scenario variable is missing its name")
            key = name.casefold()
            item = dict(raw)
            if key in positions:
                ordered[positions[key]] = item
            else:
                positions[key] = len(ordered)
                ordered.append(item)
        return ordered

    @classmethod
    def _cache_key(
        cls,
        req: ScenarioProposeRequest,
        db_variables: list[dict[str, Any]],
    ) -> tuple:
        # The LLM output depends on the complete DB definitions because it must suppress
        # semantic duplicates, not merely exact-name duplicates. Hash the full definitions
        # deterministically so a changed DB definition cannot reuse a stale proposal.
        canonical_db = json.dumps(db_variables or [], sort_keys=True, separators=(",", ":"), default=str)
        db_fingerprint = hashlib.sha256(canonical_db.encode("utf-8")).hexdigest()
        source_fingerprint = hashlib.sha256(
            json.dumps(source_manifest(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        return (
            "agentic_proposal_v28_low_balance_customer_override_reconciled_scenario_ranked_catalog",
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
        """Merge DB-controlled variables without allowing duplicate field names.

        Precedence: user-selected > DB-recommended > LLM-generated.
        """
        source_by_name: dict[str, str] = {}
        raw_persisted_by_name: dict[str, dict[str, Any]] = {}
        ordered: list[GeneratedSchemaField] = []
        by_name: dict[str, GeneratedSchemaField] = {}

        def add_schema_fields(items: list[GeneratedSchemaField]) -> None:
            for field in items or []:
                key = field.name.strip().casefold()
                if not key or key in by_name:
                    continue
                ordered.append(field)
                by_name[key] = field
                generated_from = str(field.provenance.get("generated_from") or "").strip().lower()
                source_by_name[key] = "OFFICIAL_JSON" if generated_from == "official_json_source" else "LLM_GENERATED"

        def add_persisted(items: list[dict[str, Any]], source: str) -> None:
            for raw in items or []:
                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid MongoDB scenario variable: {raw!r}")
                validate_db_definition(raw)
                key = str(raw.get("name") or "").strip().casefold()
                if not key:
                    raise ValueError("MongoDB scenario variable is missing its name")
                field = GeneratedSchemaField.model_validate(raw)
                if key in by_name:
                    idx = next(i for i, existing in enumerate(ordered) if existing.name.strip().casefold() == key)
                    ordered[idx] = field
                else:
                    ordered.append(field)
                by_name[key] = field
                source_by_name[key] = source
                # Keep the actual Mongo definition byte-for-byte at the dictionary level so
                # it can be persisted/returned without silently stripping DB-owned metadata.
                raw_persisted_by_name[key] = dict(raw)

        add_schema_fields(list(schema.fields))
        add_persisted(recommended, "DB_RECOMMENDED")
        add_persisted(user_selected, "USER_SELECTED")
        merged = schema.model_copy(update={"fields": ordered})
        return merged, source_by_name, raw_persisted_by_name

    def propose(self, req: ScenarioProposeRequest) -> ScenarioImportResponse:
        prompt = req.business_scenario.strip()
        requested_scenario_id = req.requested_scenario_id.strip()
        industry_key = match_industry_key(req.industry_type)
        if industry_key != "telecom":
            raise ValueError(
                "scenario/propose currently supports telecom industry aliases: "
                "Telecom, Telecommunication, or Telecommunications (case-insensitive)"
            )

        json_grounded = is_json_grounded_domain(req.domain)
        grounding_requirement = (
            "JSON-SOURCE REQUIREMENT: because this domain is Low Balance & Top-up, use ONLY the supplied TMF654 and TMF629 v4.0.0 Swagger/OpenAPI artifacts for official variable selection. "
            "Do not use PDFs, unrelated telecom standards, templates, CSV examples, memory, generic telecom knowledge, or application-specific hardcoded variables as a variable source. "
            "Every non-DB executable variable must be an exact scalar leaf from the supplied machine-readable catalog. The LLM is a selector/reviewer only; it cannot invent, rename, alias, or derive new executable variable names. "
            if json_grounded
            else "Use all relevant concepts from the complete approved telecom standards registry context, across all registered source URLs. "
        )
        # Mongo variables are authoritative inputs to the Low Balance variable set. Fetch both
        # scenario recommendations and the user's selected variables before the LLM/cache step.
        recommended = get_recommended(requested_scenario_id, 1)
        user_selected = (
            get_user_variables(req.user_id.strip(), requested_scenario_id, 1)
            if req.user_id and req.user_id.strip()
            else []
        )
        if json_grounded:
            recommended, user_selected, quarantined_customer_names = filter_incompatible_low_balance_db_customer_overrides(
                recommended, user_selected
            )
            if quarantined_customer_names:
                logger.warning(
                    "Low Balance quarantined incompatible MongoDB customer_id override(s); TMF629 Customer.id remains authoritative: %s",
                    quarantined_customer_names,
                )
            recommended, user_selected, suppressed_db_names = dedupe_db_variable_sources(
                recommended, user_selected
            )
            if suppressed_db_names:
                logger.info(
                    "Low Balance suppressed duplicate MongoDB variables before proposal: %s",
                    suppressed_db_names,
                )

        db_variables = self._merge_db_variable_sources(recommended, user_selected)
        if json_grounded:
            # Low Balance has two source families only. account_id and msisdn are absent from
            # the bundled TMF654/TMF629 catalog, so the exact DB variables are mandatory.
            validate_low_balance_required_identity_sources(db_variables)
        protected_name_set = set(self._variable_name_keys(db_variables))
        if json_grounded and db_variables:
            # Do semantic DB-vs-JSON deduplication BEFORE the deterministic breadth budget.
            # Otherwise a JSON alias such as topupbalance_is_auto_topup can consume one of the
            # maximum official slots, only to be removed later when is_automatic_topup wins.
            db_signatures = {
                semantic_signature(item) for item in db_variables if isinstance(item, dict)
            }
            protected_name_set.update({
                name for name, spec in official_catalog_by_name().items()
                if semantic_signature(spec) in db_signatures
            })
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
            "and generation behavior. For Low Balance & Top-up, return only complementary official JSON variables from the supplied catalog; persisted MongoDB variables are supplied separately and must never be recreated or renamed."
        )

        cid = ensure_conversation(requested_scenario_id, req.user_id, requested_scenario_id)
        append_message(cid, "user", agent_prompt, requested_scenario_id=requested_scenario_id)
        cache_key = self._cache_key(req, db_variables)
        cached = get_proposal(cache_key)
        if cached is not None:
            intent = ScenarioIntent.model_validate(cached["intent"])
            schema = ScenarioSchema.model_validate(cached["schema"])
            logger.info("[AgenticSchemaWorkflow] Proposal cache hit; Gemini skipped.")
        else:
            intent = self._get_intent_agent().run(
                agent_prompt,
                country=req.country,
                industry_type=industry_key,
                domain_query=req.domain,
                excluded_variable_names=list(protected_names),
                persisted_variables=db_variables,
            )
            intent.industry_type = industry_key
            intent.domain = req.domain
            intent.subdomain = req.use_case.strip().lower() if req.use_case.strip().lower() in {
                "prepaid", "postpaid", "charging", "usage", "customer", "network"
            } else "unknown"
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
            )
            set_proposal(cache_key, {"intent": intent.model_dump(), "schema": schema.model_dump()})

        # For Low Balance, remove only official JSON fields that duplicate an authoritative DB
        # variable by business use. DB fields are never modified or removed.
        if json_grounded:
            filtered_fields, duplicate_names = dedupe_schema_fields_against_db(
                list(schema.fields), db_variables
            )
            if duplicate_names:
                logger.info(
                    "Low Balance removed JSON variables duplicated by MongoDB definitions: %s",
                    duplicate_names,
                )
            schema = schema.model_copy(update={"fields": filtered_fields})

        schema, variable_sources, raw_persisted_by_name = self._merge_persisted_variables(
            schema,
            recommended,
            user_selected,
        )

        if json_grounded:
            # Final executable boundary: nothing outside the official catalog or MongoDB may
            # survive the proposal stage. This catches future code paths that bypass the compiler.
            validate_low_balance_variable_sources(
                self._schema_to_variables(schema, raw_persisted_by_name)[0],
                variable_sources,
                db_variable_names=set(raw_persisted_by_name),
            )

        unresolved_questions = self.compiler.approval_questions(intent, schema)
        variables, field_order = self._schema_to_variables(schema, raw_persisted_by_name)
        for variable in variables:
            key = str(variable.get("name") or "").strip().lower()
            persisted_source = variable_sources.get(key)
            if persisted_source:
                # Low Balance keeps DB definitions unchanged and records provenance separately
                # in the draft; other domains retain the legacy per-variable source annotation.
                if not json_grounded:
                    variable["source"] = persisted_source
                continue
            if is_json_grounded_domain(req.domain):
                # Do not add an application/LLM/derived variable source to Low Balance fields.
                # Every such field has already been proven to exist in the supplied JSON catalog.
                continue
            variable["source"] = "LLM_GENERATED"
        type_of_data = self._infer_type_of_data(req.type_of_data, schema)
        entity_key = self._entity_key(req.entity_key, field_order, type_of_data)

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
            "source_policy": "bundled_official_swagger_only" if json_grounded else "approved_telecom_standards_registry",
            "source_documents": source_manifest() if json_grounded else [],
            "variable_sources": variable_sources,
            "db_variable_names": sorted(raw_persisted_by_name.keys()) if json_grounded else [],
            "db_variable_definitions": raw_persisted_by_name if json_grounded else {},
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
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Apply safe HITL changes to an agentic draft. No new semantics can be introduced."""
        schema = ScenarioSchema.model_validate(draft["schema"])
        draft_source_by_name = {
            str(name).strip().casefold(): str(source).strip().upper()
            for name, source in (draft.get("variable_sources") or {}).items()
            if str(name).strip() and str(source).strip()
        }
        draft_raw_by_name = {
            str(item.get("name") or "").strip().casefold(): dict(item)
            for item in (draft.get("variables") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }

        # Reconcile legacy Low Balance drafts before unresolved-requirement checks. This is a
        # local executable-schema migration only: an incompatible DB customer_id is replaced by
        # the bundled TMF629 contract, while all unrelated Mongo variables remain unchanged.
        if is_json_grounded_domain(draft.get("domain")):
            schema, reconciled_variables, draft_source_by_name, reconciled_db_names, _ = reconcile_low_balance_schema(
                schema,
                list(draft_raw_by_name.values()),
                draft_source_by_name,
                set(draft.get("db_variable_names") or []),
                list(draft.get("field_order") or []),
            )
            draft_raw_by_name = {
                str(item.get("name") or "").strip().casefold(): dict(item)
                for item in reconciled_variables
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            }
        else:
            reconciled_db_names = set(draft.get("db_variable_names") or [])
        # Concept names extracted by the LLM are soft semantic hints. They may not have
        # one-to-one registry entities and must never block HITL confirmation. Only hard
        # executable failures (no usable fields / requested entity key not represented)
        # block confirmation. This also makes older drafts containing legacy
        # ``Unknown concept ...`` items confirmable without requiring a re-proposal.
        blocking_unresolved = []
        external_db_names = {
            str(name).strip().casefold()
            for name in reconciled_db_names
            if str(name).strip()
        }
        proposed_names = {
            str(item.get("name") or "").strip().casefold()
            for item in draft_raw_by_name.values()
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        for item in schema.unresolved_items:
            text = str(item).strip()
            lower = text.lower()
            if lower.startswith("unknown concept ") or lower.startswith("unknown requested concept "):
                continue
            if ("entity key '" in lower or "requested entity key '" in lower):
                marker = lower.split("entity key '", 1)[-1]
                candidate = marker.split("'", 1)[0].strip().casefold()
                # A requested entity key may legitimately be supplied by MongoDB even when
                # it is absent from the official JSON source catalog. Once it is present in
                # the proposed/draft variable set, the requirement is executable.
                if candidate in external_db_names or candidate in proposed_names:
                    continue
                if SchemaCompiler.is_mandatory_telecom_field(candidate):
                    continue
            blocking_unresolved.append(text)
        if blocking_unresolved:
            raise ValueError(
                "Cannot confirm an agentic draft with unresolved executable requirements: "
                + "; ".join(blocking_unresolved)
            )
        fields_by_name = {field.name: field for field in schema.fields}

        mandatory_telecom = {"subscriber_id", "account_id", "msisdn"}
        if is_json_grounded_domain(draft.get("domain")):
            mandatory_telecom = {"customer_id", "account_id", "msisdn"}
        for name in delete:
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot delete unknown agentic field '{name}'")
            if name in mandatory_telecom:
                raise ValueError(f"HITL cannot delete mandatory telecom field '{name}'")
            if fields_by_name[name].required:
                raise ValueError(f"HITL cannot delete required field '{name}'")

        # Agentic additions must reference an existing field contract in the proposed
        # schema. New entities are intentionally not created during confirmation.
        for item in add:
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError("Agentic HITL additions require an existing field name")
            if name not in fields_by_name:
                raise ValueError(f"Agentic HITL cannot add field '{name}' because it is not in the proposed registry-backed schema")
            raise ValueError(f"Agentic HITL cannot add duplicate field '{name}'; revise existing fields instead")

        allowed_override_keys = {"nullable", "description", "params"}
        applied_edits: dict[str, dict[str, Any]] = {}
        for edit in edit:
            name = edit.name if hasattr(edit, "name") else str(edit.get("name") or "")
            changes = edit.changes if hasattr(edit, "changes") else dict(edit.get("changes") or {})
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot edit unknown agentic field '{name}'")
            source_key = name.strip().casefold()
            if draft_source_by_name.get(source_key) in {"DB_RECOMMENDED", "USER_SELECTED"} and changes:
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
            applied_edits[name.strip().casefold()] = dict(changes)

        remaining = [field for field in schema.fields if field.name not in set(delete)]
        # Registry dependencies may reference entity canonical IDs (for example
        # subscriber.customer_id -> customer), not column names. Only enforce a
        # dependency here when the dependency explicitly names another field in
        # the compiled schema. Entity-level dependencies are validated by the
        # registry/compiler and are not broken merely because a column was removed.
        remaining_names = {field.name for field in remaining}
        for field in remaining:
            missing_field_dependencies = sorted(
                dep for dep in field.depends_on
                if dep in {f.name for f in schema.fields} and dep not in remaining_names
            )
            if missing_field_dependencies:
                raise ValueError(
                    f"HITL deletion would break field dependencies for '{field.name}': {missing_field_dependencies}"
                )

        variables: list[dict[str, Any]] = []
        field_order: list[str] = []
        for field in remaining:
            key = field.name.strip().casefold()
            if draft_source_by_name.get(key) in {"DB_RECOMMENDED", "USER_SELECTED"} and key in draft_raw_by_name:
                # Preserve the DB-owned definition as-is unless the user explicitly edited it.
                # Even then, only the already-allowed HITL keys are applied.
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

        if is_json_grounded_domain(draft.get("domain")):
            source_by_name = {
                name: source
                for name, source in draft_source_by_name.items()
                if name in {str(v.get("name") or "").strip().casefold() for v in variables if isinstance(v, dict)}
            }
            validate_low_balance_variable_sources(
                variables,
                source_by_name,
                db_variable_names=set(draft_raw_by_name),
            )

        if draft.get("type_of_data") == "transactional" and draft.get("entity_key") not in field_order:
            raise ValueError("HITL changes would remove the transactional entity key")
        return variables, field_order

_WORKFLOW_SINGLETONS: dict[str, AgenticSchemaWorkflow] = {}


def get_agentic_workflow(api_key: str | None = None, registry: TelecomRegistry | None = None) -> AgenticSchemaWorkflow:
    """Reuse the Gemini intent client/registry objects across proposal requests.

    The cache key is the explicit API key (or a process-local default), never requestedScenarioId.
    This removes repeated Gemini client/provider construction from /scenario/propose.
    """
    key = api_key or "__default__"
    workflow = _WORKFLOW_SINGLETONS.get(key)
    if workflow is None:
        workflow = AgenticSchemaWorkflow(api_key=api_key, registry=registry)
        _WORKFLOW_SINGLETONS[key] = workflow
    return workflow
