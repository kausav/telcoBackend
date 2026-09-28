from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class WorkflowState(BaseModel):
    """Shared state object passed through every agent in the pipeline."""

    # ── Inputs ────────────────────────────────────────────────────────────
    scenario: str       # e.g. "LB-01"
    count: int          # number of users/entities requested for transactional scenarios
    industry: str = "generic"  # Request context only; standards come from MongoDB source documents.
    country: str | None = None  # Request context only; not an LLM standards source.
    type_of_data: Literal["transactional", "aggregational"] = "aggregational"
    batch_size: int = 50
    records_per_user: int = 10

    # Complete confirmed scenario context.  The generator pipeline receives the
    # Full context persisted by the propose/confirm flow.
    # use case and business intent, rather than only industry/country/type.
    domain: str = ""
    business_scenario: str = ""
    business_response: str | None = None
    expected_outcome: str | None = None
    scenario_type: str | None = None
    use_case: str | None = None
    entity_key: str | None = None
    scenario_context: dict[str, Any] = Field(default_factory=dict)

    # ── Agent outputs (populated as pipeline runs) ────────────────────────
    rules: dict[str, Any] = Field(default_factory=dict)
    raw_records: list[dict[str, Any]] = Field(default_factory=list)
    final_records: list[dict[str, Any]] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    record_errors: list[dict[str, Any]] = Field(default_factory=list)
    field_order: list[str] = Field(default_factory=list)