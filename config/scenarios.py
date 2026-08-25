"""
Scenario catalog. Each scenario maps to a subset/override of VARIABLES.
Static LB-01/02/03 entries removed — all scenarios are now created dynamically
via /scenario/propose + /scenario/confirm (see core/dynamic_scenarios.py).
"""

SCENARIOS: dict[str, dict] = {}

SCENARIO_LABELS = {k: v["label"] for k, v in SCENARIOS.items()}