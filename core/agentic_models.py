"""Typed contracts for the agentic telecom scenario flow.

The LLM is allowed to produce only ScenarioIntent. Executable schema objects are
compiled from the runtime standards registry.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ScenarioIntent(BaseModel):
    """LLM output: intent only, never executable schema semantics."""

    model_config = ConfigDict(extra="forbid")

    domain: Literal["telecom"] = "telecom"
    subdomain: Literal["prepaid", "postpaid", "charging", "usage", "customer", "network", "unknown"] = "prepaid"
    requested_entities: list[str] = Field(default_factory=list, max_length=20)
    requested_relationships: list[str] = Field(default_factory=list, max_length=30)
    country: str | None = None
    currency: str | None = None
    record_count: int | None = Field(default=None, ge=1, le=5_000_000)
    time_window_days: int | None = Field(default=None, ge=1, le=3650)
    notes: list[str] = Field(default_factory=list, max_length=20)
    ambiguities: list[str] = Field(default_factory=list, max_length=20)


class ResolvedConcept(BaseModel):
    canonical_id: str
    name: str
    source_model: str
    source_references: list[str] = Field(default_factory=list)
    selected_attributes: list[str] = Field(default_factory=list)


class SchemaRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relation: str
    cardinality: str
    required: bool = False
    source_references: list[str] = Field(default_factory=list)


class GeneratedSchemaField(BaseModel):
    name: str
    dtype: str
    description: str = ""
    gen: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    nullable: bool = False
    required: bool = False
    formula: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class ScenarioSchema(BaseModel):
    domain: str = "telecom"
    subdomain: str = "prepaid"
    applicable_standards: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[ResolvedConcept] = Field(default_factory=list)
    relationships: list[SchemaRelationship] = Field(default_factory=list)
    fields: list[GeneratedSchemaField] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScenarioProposeRequest(BaseModel):
    """JSON body equivalent of the former scenario-import metadata, without the file.

    businessScenario is the natural-language request used by the intent agent.
    conversationId is optional so follow-up requests can reuse prior chat context.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    scenario_id: str = Field(alias="scenarioId", min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=100)
    type_of_data: Literal["transactional", "aggregational"] | None = Field(default=None, alias="typeOfData")
    industry_type: str = Field(default="generic", alias="industryType", max_length=100)
    country: str | None = Field(default=None, max_length=20)
    business_scenario: str = Field(default="", alias="businessScenario", max_length=12000)
    business_response: str | None = Field(default=None, alias="businessResponse", max_length=5000)
    expected_outcome: str | None = Field(default=None, alias="expectedOutcome", max_length=5000)
    scenario_type: str = Field(default="agentic", alias="scenarioType", max_length=100)
    use_case: str | None = Field(default=None, alias="useCase", max_length=200)
    label: str = Field(default="", max_length=200)
    entity_key: str | None = Field(default=None, alias="entityKey", max_length=200)

    conversation_id: str | None = Field(default=None, alias="conversationId", max_length=200)


class ScenarioImportResponse(BaseModel):
    """Compatibility response contract used by the scenario proposal workflow."""
    success: bool = True
    draft_id: str
    scenario_id: str
    label: str
    journey: str
    description: str
    variables: list[dict]
    field_order: list[str]
    typeOfData: Literal["transactional", "aggregational"]
    entityKey: str | None = None
