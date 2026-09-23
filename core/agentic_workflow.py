"""Agentic telecom scenario proposal and HITL validation.

Public API design: scenario/propose creates a normal draft; scenario/confirm is the
HITL approval/edit action; scenario/generate executes only confirmed scenarios.
"""
from __future__ import annotations

from typing import Any
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
    def _schema_to_variables(schema: ScenarioSchema) -> tuple[list[dict[str, Any]], list[str]]:
        variables: list[dict[str, Any]] = []
        field_order: list[str] = []
        for field in schema.fields:
            item = field.model_dump()
            item.pop("provenance", None)
            variables.append(item)
            field_order.append(field.name)
        return variables, field_order

    @staticmethod
    def _cache_key(req: ScenarioProposeRequest) -> tuple:
        return (
            "agentic_proposal_v13_quality_gated_schema",
            req.industry_type.strip().lower(),
            req.country.strip().upper(),
            req.domain.strip().lower(),
            req.scenario_type.strip().lower(),
            req.type_of_data,
            req.use_case.strip().lower(),
            " ".join(req.business_scenario.split()).strip().lower(),
        )

    @staticmethod
    def _merge_persisted_variables(schema: ScenarioSchema, recommended: list[dict[str, Any]], user_selected: list[dict[str, Any]]) -> tuple[ScenarioSchema, dict[str, str]]:
        """Merge DB-controlled variables without allowing duplicate field names.

        Precedence: user-selected > DB-recommended > LLM-generated.
        """
        source_by_name: dict[str, str] = {}
        ordered: list[GeneratedSchemaField] = []
        by_name: dict[str, GeneratedSchemaField] = {}

        def add(items: list[dict[str, Any]], source: str, replace: bool = False):
            for raw in items or []:
                try:
                    field = GeneratedSchemaField.model_validate(raw)
                except Exception:
                    logger.warning("Ignoring invalid persisted scenario variable '%s'", raw.get("name") if isinstance(raw, dict) else raw)
                    continue
                key = field.name.strip().lower()
                if not key:
                    continue
                if key in by_name and not replace:
                    continue
                if key in by_name and replace:
                    idx = next(i for i, existing in enumerate(ordered) if existing.name.strip().lower() == key)
                    ordered[idx] = field
                else:
                    ordered.append(field)
                by_name[key] = field
                source_by_name[key] = source

        # Start with the LLM/registry schema, then overlay DB recommendations, then user choices.
        add([field.model_dump() for field in schema.fields], "LLM_GENERATED")
        add(recommended, "DB_RECOMMENDED", replace=True)
        add(user_selected, "USER_SELECTED", replace=True)
        merged = schema.model_copy(update={"fields": ordered})
        return merged, source_by_name

    def propose(self, req: ScenarioProposeRequest) -> ScenarioImportResponse:
        prompt = req.business_scenario.strip()
        industry_key = match_industry_key(req.industry_type)
        if industry_key != "telecom":
            raise ValueError(
                "scenario/propose currently supports telecom industry aliases: "
                "Telecom, Telecommunication, or Telecommunications (case-insensitive)"
            )

        json_grounded = is_json_grounded_domain(req.domain)
        grounding_requirement = (
            "JSON-SOURCE REQUIREMENT: because this domain is Low Balance & Top-up, use the supplied TMF654 and TMF629 v4.0.0 Swagger/OpenAPI artifacts as the official standards grounding. "
            "Do not use PDFs, unrelated telecom standards, templates, CSV examples, memory, or general telecom knowledge as standards evidence. "
            "Standard-backed variables must come from those machine-readable models. Scenario-specific analytical variables may be added only when required by the business scenario and must be clearly treated as scenario-derived. "
            if json_grounded
            else "Use all relevant concepts from the complete approved telecom standards registry context, across all registered source URLs. "
        )
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
            "Select variables that make the behavioral difference observable; do not use scenarioId to achieve that difference. "
            "Use ALL request inputs except scenarioId and entityKey as semantic/context signals: scenarioType, industryType, domain, "
            "businessScenario, typeOfData, country, and useCase must materially constrain the variable set, field parameters, scope, "
            "and generation behavior. Return the widest relevant schema supported by the approved grounding; do not truncate it."
        )

        cid = ensure_conversation(req.scenario_id, req.user_id, req.scenario_id)
        append_message(cid, "user", agent_prompt, requested_scenario_id=req.scenario_id)
        cache_key = self._cache_key(req)
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
            )
            set_proposal(cache_key, {"intent": intent.model_dump(), "schema": schema.model_dump()})

        # Persisted scenario configuration is layered on top of the current LLM proposal.
        # The base LLM proposal remains cacheable and user-independent; only the merge is user-specific.
        requested_scenario_id = req.scenario_id.strip()
        if is_json_grounded_domain(req.domain):
            # Standards-grounded domain proposals must remain reproducible from the approved
            # source artifacts; do not overlay unrelated persisted recommendations.
            schema, variable_sources = schema, {}
        else:
            recommended = get_recommended(requested_scenario_id, 1)
            user_selected = get_user_variables(req.user_id.strip(), requested_scenario_id, 1) if req.user_id and req.user_id.strip() else []
            schema, variable_sources = self._merge_persisted_variables(schema, recommended, user_selected)

        unresolved_questions = self.compiler.approval_questions(intent, schema)
        variables, field_order = self._schema_to_variables(schema)
        for variable in variables:
            key = str(variable.get("name") or "").strip().lower()
            if is_json_grounded_domain(req.domain):
                field = next((candidate for candidate in schema.fields if candidate.name == variable.get("name")), None)
                if key in {"subscriber_id", "account_id", "msisdn"}:
                    variable["source"] = "APPLICATION_REQUIRED"
                elif field and (field.provenance.get("source_registry_attribute") or field.provenance.get("source_json_id")):
                    variable["source"] = "OFFICIAL_JSON_GROUNDED"
                else:
                    variable["source"] = "SCENARIO_DERIVED"
            else:
                variable["source"] = variable_sources.get(key, "LLM_GENERATED")
        type_of_data = self._infer_type_of_data(req.type_of_data, schema)
        entity_key = self._entity_key(req.entity_key, field_order, type_of_data)

        draft_id = new_draft_id()
        description = req.business_scenario
        draft = {
            "label": req.scenario_id,
            "journey": req.domain,
            "description": description,
            "variables": variables,
            "field_order": field_order,
            "domain": req.domain,
            "business_scenario": req.business_scenario,
            "business_response": None,
            "expected_outcome": None,
            "use_case": req.use_case,
            "scenario_id": req.scenario_id,
            "requested_scenario_id": req.scenario_id,
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
        }
        save_draft(draft_id, draft)
        save_proposal(
            request_id=draft_id,
            user_id=req.user_id.strip() if req.user_id else None,
            requested_scenario_id=requested_scenario_id,
            scenario_version=1,
            payload={"scenario_id": req.scenario_id, "requested_scenario_id": requested_scenario_id, "variables": variables, "field_order": field_order, "intent": intent.model_dump()},
        )
        append_message(cid, "assistant", json.dumps({"intent": intent.model_dump(), "action": "schema_proposed"}, sort_keys=True), requested_scenario_id=requested_scenario_id)
        return ScenarioImportResponse(
            success=True,
            draft_id=draft_id,
            scenario_id=req.scenario_id,
            requested_scenario_id=req.scenario_id,
            journey=req.domain,
            description=description,
            variables=variables,
            field_order=field_order,
            typeOfData=type_of_data,
            entityKey=entity_key,
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
        # Concept names extracted by the LLM are soft semantic hints. They may not have
        # one-to-one registry entities and must never block HITL confirmation. Only hard
        # executable failures (no usable fields / requested entity key not represented)
        # block confirmation. This also makes older drafts containing legacy
        # ``Unknown concept ...`` items confirmable without requiring a re-proposal.
        blocking_unresolved = []
        for item in schema.unresolved_items:
            text = str(item).strip()
            lower = text.lower()
            if lower.startswith("unknown concept ") or lower.startswith("unknown requested concept "):
                continue
            # subscriber_id/account_id/msisdn are application-level mandatory anchors,
            # not required literal attribute names in every official telecom model.
            # The compiler supplies executable fallbacks for them, so legacy/current
            # drafts containing this historical unresolved marker remain confirmable.
            if ("entity key '" in lower or "requested entity key '" in lower):
                marker = lower.split("entity key '", 1)[-1]
                candidate = marker.split("'", 1)[0].strip()
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
        for edit in edit:
            name = edit.name if hasattr(edit, "name") else str(edit.get("name") or "")
            changes = edit.changes if hasattr(edit, "changes") else dict(edit.get("changes") or {})
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot edit unknown agentic field '{name}'")
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
            data = field.model_dump()
            data.pop("provenance", None)
            variables.append(data)
            field_order.append(field.name)

        if draft.get("type_of_data") == "transactional" and draft.get("entity_key") not in field_order:
            raise ValueError("HITL changes would remove the transactional entity key")
        return variables, field_order

_WORKFLOW_SINGLETONS: dict[str, AgenticSchemaWorkflow] = {}


def get_agentic_workflow(api_key: str | None = None, registry: TelecomRegistry | None = None) -> AgenticSchemaWorkflow:
    """Reuse the Gemini intent client/registry objects across proposal requests.

    The cache key is the explicit API key (or a process-local default), never scenarioId.
    This removes repeated Gemini client/provider construction from /scenario/propose.
    """
    key = api_key or "__default__"
    workflow = _WORKFLOW_SINGLETONS.get(key)
    if workflow is None:
        workflow = AgenticSchemaWorkflow(api_key=api_key, registry=registry)
        _WORKFLOW_SINGLETONS[key] = workflow
    return workflow
