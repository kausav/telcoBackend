from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

from config.industry_profiles import DEFAULT_COUNTRY


class WorkflowState(BaseModel):
    """Shared state object passed through every agent in the pipeline."""

    # ── Inputs ────────────────────────────────────────────────────────────
    scenario: str       # e.g. "LB-01"
    count: int          # total records requested
    industry: str = "generic"  # e.g. "Telecom", "Banking", "Retail" — drives industry conventions (see config/industry_profiles.py)
    country: str = DEFAULT_COUNTRY  # ISO 3166-1 alpha-2; drives country conventions (see config/industry_profiles.py)
    type_of_data: Literal["transactional", "aggregational"] = "aggregational"
    batch_size: int = 50

    # ── Agent outputs (populated as pipeline runs) ────────────────────────
    rules: dict[str, Any] = Field(default_factory=dict)
    raw_records: list[dict[str, Any]] = Field(default_factory=list)
    final_records: list[dict[str, Any]] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    field_order: list[str] = Field(default_factory=list)