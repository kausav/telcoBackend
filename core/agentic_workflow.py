"""Agentic telecom scenario proposal and HITL validation.

Public API design: scenario/propose creates a normal draft; scenario/confirm is the
HITL approval/edit action; scenario/generate executes only confirmed scenarios.
"""
from __future__ import annotations

from typing import Any
import json

from core.agentic_models import ScenarioImportResponse, ScenarioProposeRequest, ScenarioSchema
from core.conversation_store import append_message, ensure_conversation, get_messages
from core.dynamic_scenarios import new_draft_id, save_draft
from core.telecom_registry import TelecomRegistry, get_registry
from agents.intent_agent import PydanticAIIntentAgent
from agents.schema_compiler import SchemaCompiler
from config.industry_profiles import match_industry_key


class AgenticSchemaWorkflow:
    """Build a standards-backed draft and validate HITL edits without an extra API."""

    def __init__(self, api_key: str | None = None, registry: TelecomRegistry | None = None):
        self.registry = registry or get_registry()
        self.intent_agent = PydanticAIIntentAgent(api_key=api_key, registry=self.registry)
        self.compiler = SchemaCompiler(self.registry)

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

    def propose(self, req: ScenarioProposeRequest) -> ScenarioImportResponse:
        prompt = req.business_scenario.strip()
        industry_key = match_industry_key(req.industry_type)
        if industry_key != "telecom":
            raise ValueError(
                "scenario/propose currently supports industryType values 'Telecommunications' or 'Telecom'"
            )

        # industryType selects the telecom model family; domain selects the business-domain
        # slice. The model still returns intent only; the registry/compiler remains the
        # semantic authority for entities, attributes and relationships.
        agent_prompt = (
            f"Industry: {req.industry_type}\n"
            f"Business domain: {req.domain}\n"
            f"Use case: {req.use_case}\n"
            f"Scenario type: {req.scenario_type}\n"
            f"Data type: {req.type_of_data}\n"
            f"Country: {req.country}\n"
            f"Entity key: {req.entity_key}\n"
            f"Business scenario: {prompt}"
        )

        cid = ensure_conversation(req.scenario_id)
        history = get_messages(cid, limit=30)
        append_message(cid, "user", agent_prompt)

        intent = self.intent_agent.run(
            agent_prompt,
            history,
            country=req.country,
            industry_type=industry_key,
            domain_query=req.domain,
        )
        # Backend-owned request values are authoritative; the LLM cannot change them.
        intent.industry_type = industry_key
        intent.domain = req.domain
        intent.subdomain = req.use_case.strip().lower() if req.use_case.strip().lower() in {
            "prepaid", "postpaid", "charging", "usage", "customer", "network"
        } else "unknown"
        if req.country:
            intent.country = req.country
        schema = self.compiler.compile(
            intent,
            min_variables=20,
            domain_query=req.domain,
            entity_key=req.entity_key,
        )

        unresolved_questions = self.compiler.approval_questions(intent, schema)
        variables, field_order = self._schema_to_variables(schema)
        if len(variables) < 20:
            raise ValueError(f"Agentic proposal must contain at least 20 variables; compiler produced {len(variables)}")
        type_of_data = self._infer_type_of_data(req.type_of_data, schema)
        entity_key = self._entity_key(req.entity_key, field_order, type_of_data)

        draft_id = new_draft_id()
        label = req.scenario_id
        description = req.business_scenario
        draft = {
            "label": label,
            "journey": req.domain,
            "description": description,
            "variables": variables,
            "field_order": field_order,
            "domain": req.domain,
            "business_scenario": req.business_scenario,
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
        if schema.unresolved_items:
            raise ValueError(
                "Cannot confirm an agentic draft with unresolved concepts: "
                + "; ".join(schema.unresolved_items)
            )
        fields_by_name = {field.name: field for field in schema.fields}

        for name in delete:
            if name not in fields_by_name:
                raise ValueError(f"HITL cannot delete unknown agentic field '{name}'")
            if name == "account_id" and draft.get("entity_key") == "subscriber_id":
                raise ValueError("HITL cannot delete required prepaid account_id for subscriber-centric agentic scenarios")
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
