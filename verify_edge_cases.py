"""Dynamic, schema-driven verification for the edge-case engine.

This verifier intentionally contains NO hard-coded business variables, values,
industries, or edge-case examples. It discovers usable fields from the active
scenario/configuration, generates a real base record, builds an edge condition
from that schema, applies the generic edge solver, and validates the actual
result. Run it from the project virtualenv.
"""
import ast
import random
from pathlib import Path

root = Path(__file__).parent
for path in root.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    ast.parse(path.read_text(encoding="utf-8"))
print("Python AST parse: PASS")

from config.variables import VARIABLES
from agents.generator_agent import (
    _condition_branches,
    _condition_compatible_with_schema,
    _edge_candidate,
    _generate_selected_record,
    _safe_edge_condition,
    _edge_case_count,
)

active = [v for v in VARIABLES if isinstance(v, dict) and v.get("name") and not v.get("formula")]
by_name = {v["name"]: v for v in active}

# Dynamically choose fields from the current schema. No field name is assumed.
numeric = []
categorical = []
boolean = []
for v in active:
    dtype = str(v.get("dtype", "")).lower()
    params = v.get("params") if isinstance(v.get("params"), dict) else {}
    choices = params.get("choices") if isinstance(params.get("choices"), list) else []
    if dtype in {"int", "float", "number", "numeric"}:
        lo = params.get("min", params.get("lo"))
        hi = params.get("max", params.get("hi"))
        if lo is not None and hi is not None:
            try:
                if float(hi) > float(lo):
                    numeric.append(v)
            except (TypeError, ValueError):
                pass
    elif dtype in {"boolean", "bool"}:
        boolean.append(v)
    elif choices:
        categorical.append(v)

assert active, "No active non-formula variables found in configuration"

# Prefer a numeric + categorical/boolean pair when available, but fall back to
# whatever the active schema actually provides.
num = random.choice(numeric) if numeric else None
cat = random.choice(categorical) if categorical else None
bool_var = random.choice(boolean) if boolean else None

selected = set()
assignments = {}
condition_parts = []
edge_vars = []

if num:
    params = num.get("params") or {}
    lo = float(params.get("min", params.get("lo")))
    hi = float(params.get("max", params.get("hi")))
    value = lo + (hi - lo) * 0.75
    if str(num.get("dtype", "")).lower() == "int":
        value = int(round(value))
    condition_parts.append(f"{num['name']} >= {value!r}")
    assignments[num["name"]] = value
    selected.add(num["name"])
    edge_vars.append({"name": num["name"], "gen": "constant", "params": {"value": value}})

if cat:
    choices = list((cat.get("params") or {}).get("choices") or [])
    if choices:
        value = random.choice(choices)
        condition_parts.append(f"{cat['name']} == {value!r}")
        assignments[cat["name"]] = value
        selected.add(cat["name"])
        edge_vars.append({"name": cat["name"], "gen": "constant", "params": {"value": value}})

if not condition_parts and bool_var:
    condition_parts.append(f"{bool_var['name']} == True")
    assignments[bool_var["name"]] = True
    selected.add(bool_var["name"])
    edge_vars.append({"name": bool_var["name"], "gen": "constant", "params": {"value": True}})

if not condition_parts:
    raise AssertionError("Active schema has no dynamically testable numeric/categorical/boolean field")

condition = " and ".join(condition_parts)
assert _condition_branches(condition), f"Could not derive branches for dynamic condition: {condition}"
assert _condition_compatible_with_schema(condition, VARIABLES), condition

# Generate a real record using the active generators, then transform it into an
# edge candidate. The values are intentionally generated at runtime.
base = _generate_selected_record(VARIABLES, selected, profile=None, rules=None)
group = {
    "edge_case_name": "Dynamic Verification Case",
    "condition": condition,
    "variables": edge_vars,
}
candidate = _edge_candidate(
    base, group, VARIABLES, None, {}, selected, assignments=assignments
)
assert _safe_edge_condition(condition, candidate), (condition, candidate)

print("Dynamic schema fields:", ", ".join(selected))
print("Dynamic condition:", condition)
print("Generated values:", {name: candidate.get(name) for name in selected})
print("Dynamic edge-condition validation: PASS")

assert _edge_case_count(100, 0.02) == 2
assert _edge_case_count(100, 0.04) == 4
assert _edge_case_count(100, 0.0) == 0
print("Percentage invariants: PASS")
