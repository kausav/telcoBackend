from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class WorkflowState(BaseModel):
    """Shared state object passed through every agent in the pipeline."""

    # ── Inputs ────────────────────────────────────────────────────────────
    scenario: str       # e.g. "LB-01"
    count: int          # total records requested
    industry: str = "generic"  # e.g. "Telecom", "Banking", "Retail" — drives industry conventions (see config/industry_profiles.py)
    country: str | None = None  # None means GLOBAL/non-country-specific conventions
    type_of_data: Literal["transactional", "aggregational"] = "aggregational"
    batch_size: int = 50

    # Complete confirmed scenario context.  The generator pipeline receives the
    # full /scenario/propose context through the confirmed scenario, including
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
    field_order: list[str] = Field(default_factory=list)
    transactional_event_counts: dict[str, dict[str, int]] = Field(default_factory=dict)