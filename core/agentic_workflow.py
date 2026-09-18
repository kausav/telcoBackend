"""Agentic telecom scenario proposal and HITL validation.

Public API design: scenario/propose creates a normal draft; scenario/confirm is the
HITL approval/edit action; scenario/generate executes only confirmed scenarios.
"""
from __future__ import annotations

from typing import Any
import json
import logging

from core.agentic_models import ScenarioImportResponse, ScenarioProposeRequest, ScenarioSchema, ScenarioIntent
from core.conversation_store import append_message, ensure_conversation
from core.dynamic_scenarios import new_draft_id, save_draft
from core.telecom_registry import TelecomRegistry, get_registry
from core.errors import LLMUpstreamError
from core.runtime_cache import get_proposal, set_proposal

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
            "agentic_proposal_v3",
            req.industry_type.strip().lower(),
            req.country.strip().upper(),
            req.domain.strip().lower(),
            req.scenario_type.strip().lower(),
            req.type_of_data,
            req.use_case.strip().lower(),
            req.entity_key.strip().lower(),
            " ".join(req.business_scenario.split()).strip().lower(),
        )

    def propose(self, req: ScenarioProposeRequest) -> ScenarioImportResponse:
        prompt = req.business_scenario.strip()
        industry_key = match_industry_key(req.industry_type)
        if industry_key != "telecom":
            raise ValueError(
                "scenario/propose currently supports telecom industry aliases: "
                "Telecom, Telecommunication, or Telecommunications (case-insensitive)"
            )

        agent_prompt = (
            f"Industry: {req.industry_type}\n"
            f"Business domain: {req.domain}\n"
            f"Use case: {req.use_case}\n"
            f"Scenario type: {req.scenario_type}\n"
            f"Data type: {req.type_of_data}\n"
            f"Country: {req.country}\n"
            f"Entity key: {req.entity_key}\n"
            f"Business scenario: {prompt}\n\n"
            "Variable-design requirement: propose a fresh semantic variable set with NO artificial count target or maximum. "
            "Use all relevant concepts from the complete approved telecom standards registry context, across all registered source URLs. "
            "Do not copy reference CSV variable names. Include every variable genuinely needed to represent the business scenario; "
            "return fewer or more as justified by the scenario."
        )

        cid = ensure_conversation(req.scenario_id)
        append_message(cid, "user", agent_prompt)
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
            intent.entity_key = req.entity_key
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

        unresolved_questions = self.compiler.approval_questions(intent, schema)
        variables, field_order = self._schema_to_variables(schema)
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
        }
        save_draft(draft_id, draft)
        append_message(cid, "assistant", json.dumps({"intent": intent.model_dump(), "action": "schema_proposed"}, sort_keys=True))
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
        blocking_unresolved = [
            item for item in schema.unresolved_items
            if not str(item).lower().startswith("unknown concept ")
            and not str(item).lower().startswith("unknown requested concept ")
        ]
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
