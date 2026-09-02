"""
Optional built-in scenario catalog.

The current API creates scenarios dynamically through /scenario/propose and
/scenario/confirm. The catalog remains available as a compatibility hook for
preconfigured scenarios without being required by the dynamic pipeline.
"""

SCENARIOS: dict[str, dict] = {}

SCENARIO_LABELS = {k: v["label"] for k, v in SCENARIOS.items()}