"""
Agent 4 — QA & Validation Agent

The QA layer has two parts:
1) deterministic validation/repair for every generated record (schema, types,
   ranges, categorical values, formulas, dependencies, country conventions,
   entity/transaction consistency and transactional event sequencing); and
2) optional Gemini semantic auditing for business rules that cannot be safely
   reduced to deterministic checks.

The deterministic layer is deliberately the source of truth for arithmetic and
structural invariants so bad generated data is not silently accepted just
because an LLM audit is disabled.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import uuid
from datetime import date, datetime, timezone
from typing import Any

from config.variables import VARIABLES
from core.dynamic_scenarios import resolve_entity_key, resolve_events, resolve_variables, resolve_scenario_context
from core.llm_client import GeminiClient
from core.state import WorkflowState
from config.industry_profiles import get_profile, payment_methods_for_profile, country_allowed_value

logger = logging.getLogger(__name__)

_NUMERIC_DTYPES = {"int", "float"}
_CHUNK = 50

_SYSTEM = """
You are the QA Validation Agent for a synthetic data generation pipeline covering
multiple industries and countries.

Validate the supplied records against the scenario's business rules, field
constraints, country/industry conventions, cross-field rules, calculations,
and transactional event semantics.

Do not invent fields that are intentionally absent from sparse transactional
events. Do not rewrite valid records unnecessarily.

Return JSON:
  - "valid_records": [ ... ]
  - "dropped_records": [ ... ]
  - "fixes_applied": int
  - "issues_found": int
"""


def _default_for_dtype(dtype: str):
    if dtype in _NUMERIC_DTYPES:
        return 0
    if dtype == "boolean":
        return False
    return ""


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fill_missing(rec: dict, field_order: list, dtype_map: dict) -> tuple[dict, int]:
    """Preserve the existing API behavior: fill missing/null fields with a safe default."""
    filled = 0
    for name in field_order:
        if rec.get(name) is None:
            rec[name] = _default_for_dtype(dtype_map.get(name, "string"))
            filled += 1
    return rec, filled


def _safe_formula(expr: str, rec: dict):
    """Evaluate the same small arithmetic expression language used by the generator."""
    allowed_funcs = {"round": round, "min": min, "max": max, "abs": abs}
    names = {k: v for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    try:
        tree = ast.parse(expr, mode="eval")
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
            ast.Name, ast.Call, ast.Load, ast.Tuple, ast.List,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in names and node.id not in allowed_funcs:
                return None
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs):
                return None
        return eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {**names, **allowed_funcs})
    except Exception:
        return None


def _formula_dependencies(expr: str) -> set[str]:
    """Return field names referenced by a formula expression."""
    try:
        tree = ast.parse(expr, mode="eval")
        allowed_funcs = {"round", "min", "max", "abs"}
        return {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in allowed_funcs
        }
    except Exception:
        return set()


def _normalize(v: Any) -> str:
    return str(v).strip().lower().replace(" ", "_").replace("-", "_")


def _safe_edge_condition(expression: str, rec: dict) -> bool:
    """Safely evaluate a machine-checkable edge-case condition against one record."""
    if not expression:
        return True
    try:
        tree = ast.parse(expression, mode="eval")
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Compare,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
            ast.Is, ast.IsNot, ast.Name, ast.Constant, ast.List, ast.Tuple,
            ast.UnaryOp, ast.USub, ast.UAdd, ast.Load,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
            if isinstance(node, ast.Name) and node.id not in rec:
                return False
        return bool(eval(compile(tree, "<edge-condition>", "eval"), {"__builtins__": {}}, dict(rec)))
    except Exception:
        return False


def _country_repair(name: str, value: Any, profile: dict):
    if value is None:
        return value, None
    lname = name.lower()
    if country_allowed_value(name, value, profile):
        return value, None
    if "payment" in lname or "pay_method" in lname:
        choices = payment_methods_for_profile(profile)
        if choices:
            return choices[0], f"{name} corrected for country {profile.get('country_code')}"
    if "currency" in lname:
        return profile.get("currency", value), f"{name} corrected for country {profile.get('country_code')}"
    return value, None


def _collect_formula_specs(variables: list[dict], rules: dict | None) -> list[tuple[str, str]]:
    """Collect authoritative formulas, preferring explicit variable definitions."""
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for var in variables:
        field = str(var.get("name", ""))
        expr = var.get("formula")
        if field and expr and field not in seen:
            specs.append((field, str(expr)))
            seen.add(field)
    for item in (rules or {}).get("formula_rules", []) or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field", ""))
        expr = item.get("expression")
        if field and expr and field not in seen:
            specs.append((field, str(expr)))
            seen.add(field)
    return specs


def _validate_record(
    rec: dict,
    variables: list[dict],
    field_order: list[str],
    profile: dict,
    transactional: bool,
    edge_case_variables: list[dict] | None = None,
    event_fields: set[str] | None = None,
    rules: dict | None = None,
) -> tuple[dict, list[str]]:
    """Validate and repair one record using the actual variable definitions."""
    rec = dict(rec)
    issues: list[str] = []
    by_name = {v["name"]: v for v in variables}
    active_fields = set(event_fields or field_order) if transactional else set(field_order)

    # 1. Country-sensitive values and declared categorical choices.
    for name, value in list(rec.items()):
        if name.startswith("__") or value is None:
            continue
        repaired, issue = _country_repair(name, value, profile)
        if issue:
            rec[name] = repaired
            issues.append(issue)

    for var in variables:
        name = var["name"]
        if name not in rec or rec[name] is None:
            # Sparse transactional rows are intentional. Missing event-local fields
            # are not errors and will be handled by the existing fill behavior later.
            continue
        value = rec[name]
        dtype = var.get("dtype", "string")
        params = var.get("params") or {}

        # 2. Type checks / safe coercion.
        try:
            if dtype == "int" and not isinstance(value, bool):
                if not isinstance(value, int):
                    rec[name] = int(float(value))
                    issues.append(f"{name} coerced to int")
            elif dtype == "float" and not isinstance(value, bool):
                if not isinstance(value, (int, float)):
                    rec[name] = float(value)
                    issues.append(f"{name} coerced to float")
            elif dtype == "boolean":
                if not isinstance(value, bool):
                    if str(value).strip().lower() in {"true", "1", "yes"}:
                        rec[name] = True
                    elif str(value).strip().lower() in {"false", "0", "no"}:
                        rec[name] = False
                    else:
                        raise ValueError("invalid boolean")
                    issues.append(f"{name} coerced to boolean")
            elif dtype == "datetime" and _parse_dt(value) is None:
                rec[name] = datetime.now(timezone.utc).isoformat()
                issues.append(f"{name} repaired as datetime")
            elif dtype == "date":
                try:
                    date.fromisoformat(str(value)[:10])
                except Exception:
                    rec[name] = date.today().isoformat()
                    issues.append(f"{name} repaired as date")
        except Exception:
            # Do not drop the entire record for a single malformed value; regenerate
            # from the declared generator when possible, otherwise use a safe default.
            rec[name] = _default_for_dtype(dtype)
            issues.append(f"{name} repaired from invalid type")
            value = rec[name]

        # 3. Categorical values must stay within the declared scenario vocabulary.
        if dtype in {"categorical", "string"} and params.get("choices") and rec.get(name) is not None:
            choices = list(params.get("choices", []))
            if choices and _normalize(rec[name]) not in {_normalize(c) for c in choices}:
                rec[name] = choices[0]
                issues.append(f"{name} corrected to declared choice")

        # 4. Generic numeric ranges from the variable definition.
        if isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool):
            val = float(rec[name])
            lo = params.get("min")
            hi = params.get("max")
            if lo is not None and val < float(lo):
                rec[name] = int(lo) if dtype == "int" else float(lo)
                issues.append(f"{name} raised to minimum")
            if hi is not None and val > float(hi):
                rec[name] = int(hi) if dtype == "int" else float(hi)
                issues.append(f"{name} lowered to maximum")

        # Percentage-like fields should remain 0..100 when explicitly named as percentages.
        if isinstance(rec.get(name), (int, float)) and not isinstance(rec.get(name), bool) and (
            "pct" in name.lower() or "percent" in name.lower() or "percentage" in name.lower()
        ):
            old = rec[name]
            rec[name] = max(0, min(100, old))
            if old != rec[name]:
                issues.append(f"{name} clamped to percentage range")

    # 5. Formula invariants: recompute every active scenario formula.
    # Formulas are the source of truth across ALL industries; there are no
    # telecom-specific arithmetic assumptions here. RulesAgent formulas are
    # also considered when a business rule adds a formula not present on the
    # variable definition itself.
    formula_specs = _collect_formula_specs(variables, rules)
    for field, expr in formula_specs:
        if field not in active_fields:
            continue
        deps = _formula_dependencies(expr)
        missing = [dep for dep in deps if rec.get(dep) is None]
        if missing:
            # Do not invent a numeric default for a formula when its inputs are
            # absent. For sparse transactional events this is intentional; when
            # the formula field is explicitly part of the event, the record is
            # left for the generator/schema to repair rather than fabricating a value.
            issues.append(f"{field} formula could not be evaluated; missing dependencies: {missing}")
            continue
        expected = _safe_formula(expr, rec)
        if expected is None:
            issues.append(f"{field} formula could not be evaluated")
            continue
        actual = rec.get(field)
        mismatch = False
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            mismatch = not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=0.01)
        else:
            mismatch = str(actual) != str(expected)
        if mismatch:
            rec[field] = expected
            issues.append(f"{field} corrected from formula")

    # 6. Do not hard-code industry-specific arithmetic here.
    # All calculations must come from scenario-declared formulas so the same
    # validator works across every industry.

    # 7. Common timestamp relationships when fields exist.
    event_ts = _parse_dt(rec.get("event_timestamp"))
    dispatch_ts = _parse_dt(rec.get("notification_dispatch_ts"))
    response_ts = _parse_dt(rec.get("customer_response_ts"))
    if event_ts and dispatch_ts and dispatch_ts < event_ts:
        rec["notification_dispatch_ts"] = event_ts.isoformat()
        dispatch_ts = event_ts
        issues.append("notification_dispatch_ts corrected to be >= event_timestamp")
    if dispatch_ts and response_ts and response_ts < dispatch_ts:
        rec["customer_response_ts"] = dispatch_ts.isoformat()
        issues.append("customer_response_ts corrected to be >= notification_dispatch_ts")

    # 8. Country phone-number convention. The profile supplies the expected prefix;
    # reject/repair only fields that clearly represent a phone number.
    phone_names = [n for n in rec if "msisdn" in n.lower() or "phone" in n.lower() or "mobile" in n.lower()]
    expected_cc = str(profile.get("phone_country_code", ""))
    for name in phone_names:
        value = str(rec.get(name, ""))
        if expected_cc and value and not value.startswith(expected_cc):
            # Preserve the subscriber's local digits where possible, but enforce the country code.
            digits = "".join(ch for ch in value if ch.isdigit())
            if expected_cc == "+91" and len(digits) >= 10:
                local = digits[-10:]
                if local[0] in "6789":
                    rec[name] = expected_cc + local
                else:
                    rec[name] = expected_cc + "9" + local[-9:]
            elif expected_cc == "+44" and len(digits) >= 10:
                local = digits[-10:]
                rec[name] = expected_cc + (local if local.startswith("7") else "7" + local[-9:])
            elif expected_cc == "+971":
                rec[name] = expected_cc + "50" + digits[-7:]
            elif expected_cc == "+1":
                rec[name] = expected_cc + digits[-10:].zfill(10)
            else:
                rec[name] = expected_cc + digits[-10:]
            issues.append(f"{name} corrected for country phone convention")

    # 9. Keep output limited to declared variables plus transactional metadata.
    metadata = {"journey_id", "transaction_id", "event_type", "event_sequence", "event_occurrence", "event_timestamp", "isEdgeCaseData"}
    edge_field_names = {
        str(item.get("name")) for item in (edge_case_variables or [])
        if isinstance(item, dict) and item.get("name")
    }
    if transactional:
        allowed = set(field_order) | edge_field_names | metadata
    else:
        allowed = set(field_order) | edge_field_names
    rec = {k: v for k, v in rec.items() if k in allowed}

    return rec, issues


class QAAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        records = state.raw_records
        logger.info("[QAAgent] Validating %d records", len(records))
        edge_groups = {}
        for item in state.edge_case_variables:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            group = str(item.get("edge_case_name") or "Scenario Edge Case")
            edge_groups.setdefault(group, {"condition": str(item.get("condition") or ""), "variables": []})
            edge_groups[group]["variables"].append(item)

        dyn = resolve_variables(state.scenario)
        variables, _ = dyn if dyn is not None else (VARIABLES, [])
        dtype_map = {v["name"]: v.get("dtype", "string") for v in variables}
        profile = get_profile(state.industry, state.country)
        transactional = state.type_of_data == "transactional"
        event_defs = {str(e.get("event_type")): e for e in resolve_events(state.scenario)} if transactional else {}
        formula_specs = _collect_formula_specs(variables, state.rules)

        algo_fixed = 0
        checked: list[dict] = []
        seen_transactions: set[str] = set()
        seen_entities: set[str] = set()
        journey_state: dict[str, tuple[int, datetime | None, str | None]] = {}
        journey_edge_context: dict[str, dict[str, Any]] = {}

        for rec in records:
            pre_event_def = event_defs.get(str(rec.get("event_type"))) if transactional else None
            pre_event_fields = set(pre_event_def.get("fields", []) or []) if pre_event_def else set()
            rec, issues = _validate_record(
                rec, variables, state.field_order, profile, transactional,
                event_fields=pre_event_fields,
                rules=state.rules,
                edge_case_variables=state.edge_case_variables,
            )

            # Edge-case labels are trusted only after the configured condition is checked.
            # Aggregational rows are self-contained. Transactional rows are sparse, so their
            # final edge status is validated later against the complete journey context.
            if not transactional and rec.get("isEdgeCaseData") is True:
                matched = any(_safe_edge_condition(group.get("condition", ""), rec) for group in edge_groups.values())
                if not matched:
                    rec["isEdgeCaseData"] = False
                    issues.append("isEdgeCaseData reset because no edge-case condition was satisfied")

            if transactional:
                # Metadata is structural and should never be delegated to the LLM.
                if not rec.get("journey_id"):
                    rec["journey_id"] = f"JRN-{uuid.uuid4().hex[:12].upper()}"
                    issues.append("journey_id generated")
                if not rec.get("transaction_id") or str(rec.get("transaction_id")) in seen_transactions:
                    rec["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"
                    issues.append("transaction_id generated/renewed for uniqueness")
                seen_transactions.add(str(rec["transaction_id"]))

                event_type = str(rec.get("event_type") or "BUSINESS_EVENT")
                rec["event_type"] = event_type
                seq = int(rec.get("event_sequence") or 1)
                if event_defs:
                    definition = event_defs.get(event_type)
                    if definition:
                        expected_seq = int(definition.get("sequence", seq))
                        if seq != expected_seq:
                            rec["event_sequence"] = expected_seq
                            seq = expected_seq
                            issues.append("event_sequence corrected from scenario definition")
                rec["event_sequence"] = max(1, seq)

                ts = _parse_dt(rec.get("event_timestamp"))
                if ts is None:
                    ts = datetime.now(timezone.utc)
                    rec["event_timestamp"] = ts.isoformat()
                    issues.append("event_timestamp generated")

                journey = str(rec["journey_id"])
                previous = journey_state.get(journey)
                if previous:
                    prev_seq, prev_ts, prev_entity = previous
                    if rec["event_sequence"] < prev_seq:
                        rec["event_sequence"] = prev_seq
                        issues.append("event_sequence corrected to preserve journey ordering")
                    if prev_ts and ts < prev_ts:
                        rec["event_timestamp"] = prev_ts.isoformat()
                        ts = prev_ts
                        issues.append("event_timestamp corrected to preserve journey ordering")
                    entity_name = resolve_entity_key(state.scenario)
                    if entity_name and entity_name in rec and prev_entity and str(rec[entity_name]) != prev_entity:
                        rec[entity_name] = prev_entity
                        issues.append("entity key corrected for journey consistency")
                entity_key = resolve_entity_key(state.scenario)
                entity_value = str(rec.get(entity_key)) if entity_key and rec.get(entity_key) is not None else None
                journey_state[journey] = (rec["event_sequence"], ts, entity_value)
                if entity_value:
                    seen_entities.add(entity_value)

            # Fill missing values only after validation. Formula fields are
            # calculated first whenever their dependencies are available; never
            # replace a calculable formula with an arbitrary dtype default.
            active_fill_fields = set(event_defs.get(str(rec.get("event_type")), {}).get("fields", []) or []) if transactional else set(state.field_order)
            for formula_field, formula_expr in formula_specs:
                if formula_field not in active_fill_fields or rec.get(formula_field) is not None:
                    continue
                deps = _formula_dependencies(formula_expr)
                if deps and all(rec.get(dep) is not None for dep in deps):
                    calculated = _safe_formula(formula_expr, rec)
                    if calculated is not None:
                        rec[formula_field] = calculated
                        issues.append(f"{formula_field} calculated from formula")

            # For transactional data, events are intentionally sparse: fill only
            # fields declared by this event, never every variable in the global schema.
            if transactional:
                event_def = event_defs.get(str(rec.get("event_type")))
                fill_order = list(event_def.get("fields", []) or []) if event_def else []
            else:
                fill_order = state.field_order
            formula_field_names = {field for field, _ in formula_specs}
            safe_fill_order = [f for f in fill_order if f not in formula_field_names or rec.get(f) is not None]
            rec, filled = _fill_missing(rec, safe_fill_order, dtype_map)
            if filled:
                issues.append(f"{filled} event field(s) defaulted")
            algo_fixed += len(issues)
            checked.append(rec)
            if transactional:
                journey_id_for_edge = str(rec.get("journey_id") or "")
                if journey_id_for_edge:
                    ctx = journey_edge_context.setdefault(journey_id_for_edge, {})
                    for key, value in rec.items():
                        if key not in {"journey_id", "transaction_id", "event_type", "event_sequence",
                                       "event_occurrence", "event_timestamp", "isEdgeCaseData"} and value is not None:
                            ctx[key] = value

        # Finalize transactional edge-case labels at RECORD grain. A transactional
        # condition may reference fields from another event, so evaluate each already-
        # selected record against the complete journey context. Never promote a normal
        # record to an edge case merely because another record in the journey matches.
        if transactional and checked:
            for rec in checked:
                if rec.get("isEdgeCaseData") is not True:
                    rec["isEdgeCaseData"] = False
                    continue
                journey_id = str(rec.get("journey_id") or "")
                ctx = dict(journey_edge_context.get(journey_id, {}))
                for key, value in rec.items():
                    if key not in {"journey_id", "transaction_id", "event_type", "event_sequence",
                                   "event_occurrence", "event_timestamp", "isEdgeCaseData"} and value is not None:
                        ctx[key] = value
                matched = any(
                    _safe_edge_condition(group.get("condition", ""), ctx)
                    for group in edge_groups.values()
                )
                rec["isEdgeCaseData"] = bool(matched)

        # Optional LLM semantic audit. Deterministic validation above always runs.
        rules_text = "\n".join(f"- {r}" for r in state.rules.get("business_rules", []))
        cross_text = "\n".join(f"- {r}" for r in state.rules.get("cross_field_rules", []))
        system_prompt = _SYSTEM.format(
            rules=rules_text or "Apply the supplied variable definitions and deterministic checks.",
            cross_field_rules=cross_text or "Validate mathematical, temporal, event and business relationships.",
        )

        valid_all: list[dict] = checked
        dropped_all: list[dict] = []
        llm_fixes = 0
        llm_issues = 0
        qa_mode = os.getenv("QA_LLM_MODE", "off").strip().lower()

        if qa_mode == "full":
            valid_all = []
            for i in range(0, len(checked), _CHUNK):
                chunk = checked[i:i + _CHUNK]
                try:
                    result = self._llm.generate_json(
                        system_prompt,
                        f"Scenario: {state.scenario}\n"
                        f"Industry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                        f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                        f"Business response: {state.business_response or ''}\nExpected outcome: {state.expected_outcome or ''}\n"
                        f"Use case: {state.use_case or ''}\nScenario type: {state.scenario_type or ''}\n"
                        f"Records to validate:\n{json.dumps(chunk, default=str)}",
                        temperature=0.1,
                    )
                    validated = result.get("valid_records", chunk)
                    allowed_output = set(state.field_order) | {
                        str(item.get("name")) for item in state.edge_case_variables
                        if isinstance(item, dict) and item.get("name")
                    } | {
                        "journey_id", "transaction_id", "event_type", "event_sequence",
                        "event_occurrence", "event_timestamp", "isEdgeCaseData"
                    }
                    validated = [{k: r[k] for k in allowed_output if k in r} for r in validated]
                    valid_all.extend(validated)
                    dropped_all.extend(result.get("dropped_records", []))
                    llm_fixes += int(result.get("fixes_applied", 0))
                    llm_issues += int(result.get("issues_found", 0))
                except Exception as exc:
                    logger.warning("[QAAgent] Chunk %d error: %s — deterministic validation retained", i, exc)
                    state.errors.append(f"QA chunk {i} error: {exc}")
                    valid_all.extend(chunk)
        elif qa_mode != "off" and checked:
            sample_size = max(1, int(os.getenv("QA_LLM_SAMPLE_SIZE", "20")))
            try:
                result = self._llm.generate_json(
                    system_prompt,
                    f"Scenario: {state.scenario}\n"
                    f"Industry: {state.industry}\nCountry: {state.country or 'GLOBAL'}\n"
                    f"Domain: {state.domain}\nBusiness scenario: {state.business_scenario}\n"
                    f"Use case: {state.use_case or ''}\n"
                    f"This is a QA audit sample only. Do not rewrite the dataset.\nRecords:\n{json.dumps(checked[:sample_size], default=str)}",
                    temperature=0.1,
                )
                llm_fixes = int(result.get("fixes_applied", 0))
                llm_issues = int(result.get("issues_found", 0))
            except Exception as exc:
                logger.warning("[QAAgent] Sample audit error: %s — deterministic validation retained", exc)
                state.errors.append(f"QA sample error: {exc}")

        state.final_records = valid_all
        state.validation_report = {
            "total_input": len(records),
            "total_valid": len(valid_all),
            "total_dropped": len(dropped_all),
            "recovered": 0,
            "algo_fixes": algo_fixed,
            "llm_fixes": llm_fixes,
            "llm_issues": llm_issues,
            "deterministic_checks": [
                "schema_and_type",
                "declared_ranges_and_choices",
                "formula_and_arithmetic",
                "timestamp_relationships",
                "country_conventions",
                "entity_and_transaction_consistency",
                "transactional_event_sequence",
            ],
        }
        logger.info(
            "[QAAgent] Done. valid=%d dropped=%d algo_fixes=%d llm_fixes=%d",
            len(valid_all), len(dropped_all), algo_fixed, llm_fixes,
        )
        return state
