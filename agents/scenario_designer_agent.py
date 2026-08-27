"""
Agent 0 — Scenario Designer Agent
Given a plain-English business scenario description, asks Gemini to invent a
new scenario (LB-04, LB-05, ...) and its full variable catalog, expressed in
the same generator-type vocabulary that agents/generator_agent.py already
knows how to execute — so the result can be run through the existing
4-agent pipeline with no new generator code.
"""
from __future__ import annotations
import logging

from config.industry_profiles import get_profile, match_industry_key
from core.llm_client import GeminiClient
from core.dynamic_scenarios import get_feedback_history

logger = logging.getLogger(__name__)

MIN_VARIABLES = 15
MAX_VARIABLES = 35
# If the LLM returns fewer than this, we retry asking it to expand toward MAX_VARIABLES.
_EXPAND_TARGET = 35

# These 2 fields must appear in every scenario (static or dynamic) for cross-scenario
# customer profiling/analytics. Injected automatically if the LLM omits them.
_MANDATORY_VARIABLES: list[dict] = [
    {
        "name": "customer_lifetime_value",
        "dtype": "float",
        "description": "Right-skewed lifetime value of the customer in USD.",
        "gen": "lognormal",
        "params": {"mu": 5.5, "sigma": 0.8, "min": 25.00, "max": 2500.00},
        "depends_on": [],
        "nullable": False,
    },
    {
        "name": "loyalty_tier_customer",
        "dtype": "categorical",
        "description": "Customer's loyalty program tier.",
        "gen": "weighted_choice",
        "params": {"choices": ["Bronze", "Silver", "Gold", "Platinum"], "weights": [0.40, 0.30, 0.20, 0.10]},
        "depends_on": [],
        "nullable": False,
    },
]

# Canonical field CONCEPTS. Each concept resolves to a different canonical name per
# industry (e.g. "identifier" is subscriber_id for telecom but customer_id elsewhere),
# so field naming stays consistent WITHIN an industry without forcing telecom vocabulary
# onto unrelated industries like banking or retail.
_CONCEPT_CANONICAL: dict[str, dict[str, str]] = {
    "identifier":     {"telecom": "subscriber_id", "generic": "customer_id"},
    "phone":          {"telecom": "subscriber_msisdn", "generic": "phone_number"},
    "account":        {"generic": "account_id"},
    "segment":        {"telecom": "subscriber_segment", "generic": "customer_segment"},
    "product_code":   {"telecom": "rate_plan_code", "generic": "product_code"},
    "currency":       {"telecom": "wallet_currency", "generic": "currency_code"},
    "clv":            {"generic": "customer_lifetime_value"},
    "loyalty":        {"generic": "loyalty_tier_customer"},
    "event_timestamp":{"generic": "event_timestamp"},
    "balance_before": {"generic": "balance_before"},
    "balance_after":  {"generic": "balance_after"},
    "amount":         {"telecom": "recharge_amount", "generic": "transaction_amount"},
}
_CONCEPT_DESCRIPTIONS: dict[str, str] = {
    "identifier":      "unique identifier for the person/account holder",
    "phone":           "the person's phone number (E.164)",
    "account":         "billing/account identifier tied to the person",
    "segment":         "customer segment/persona category",
    "product_code":    "the person's selected plan/product/tariff code",
    "currency":        "currency code for monetary fields",
    "clv":             "person's lifetime value",
    "loyalty":         "person's loyalty program tier",
    "event_timestamp": "timestamp the triggering event occurred",
    "balance_before":  "account balance before the event/transaction",
    "balance_after":   "account balance after the event/transaction",
    "amount":          "amount of the primary monetary transaction in this scenario (top-up/recharge/purchase/payment)",
}
# Common synonyms Gemini tends to invent for each concept, regardless of industry.
_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "identifier":      ["customer_id", "customerid", "subscriber_id", "client_id", "account_holder_id"],
    "phone":           ["phone_number", "mobile_number", "msisdn", "customer_msisdn", "subscriber_msisdn", "contact_number"],
    "account":         ["account_number", "acct_id", "account_id"],
    "segment":         ["customer_segment", "user_segment", "subscriber_segment", "client_segment"],
    "product_code":    ["plan_code", "tariff_plan", "rate_plan_code", "product_id", "subscription_plan"],
    "currency":        ["currency", "wallet_currency", "currency_code"],
    "clv":             ["clv", "customer_ltv", "customer_lifetime_value"],
    "loyalty":         ["loyalty_tier", "customer_loyalty_tier", "loyalty_tier_customer"],
    "event_timestamp": ["timestamp", "event_time", "trigger_timestamp", "event_timestamp"],
    "balance_before":  ["balance_prior", "previous_balance", "current_balance", "balance_before"],
    "balance_after":   ["new_balance", "balance_post", "post_recharge_balance", "balance_after"],
    "amount":          ["topup_amount", "recharge_value", "recharge_amount", "transaction_amount", "payment_amount", "purchase_amount"],
}
_REF_PARAM_KEYS = ("source_field", "base_field", "hi_field", "add_seconds_field")

# Identifier fields must always look the same shape (PREFIX- + 8-digit int) across every
# scenario — the LLM sometimes proposes prefixed_uuid instead of prefixed_int, which
# breaks consistency for anything downstream that assumes a fixed identifier shape.
_IDENTIFIER_GEN = "prefixed_int"
_IDENTIFIER_DIGITS = 8
_IDENTIFIER_PREFIX_BY_NAME: dict[str, str] = {"subscriber_id": "SUB-", "customer_id": "CUS-"}


def _canonical_name(concept: str, industry_key: str) -> str:
    table = _CONCEPT_CANONICAL[concept]
    return table.get(industry_key, table["generic"])


def _build_base_variables(industry_key: str, profile: dict) -> list[dict]:
    """Fields that are IDENTICAL in shape across every single scenario for a given
    industry (identifier, phone, account, event timestamp) plus the 3 mandatory
    cross-scenario analytics fields. These never need the LLM to invent them, so we
    build them directly in Python and only ask the LLM for scenario-SPECIFIC variables
    — this is what was previously causing the same 5-6 variables to be regenerated
    (and billed/latency-charged) on every single /scenario/propose call."""
    identifier_name = _canonical_name("identifier", industry_key)
    phone_name = _canonical_name("phone", industry_key)
    account_name = _canonical_name("account", industry_key)
    prefix = _IDENTIFIER_PREFIX_BY_NAME.get(identifier_name, "ID-")

    base = [
        {
            "name": identifier_name,
            "dtype": "string",
            "description": "Unique identifier for the person/account holder.",
            "gen": _IDENTIFIER_GEN,
            "params": {"prefix": prefix, "digits": _IDENTIFIER_DIGITS},
            "depends_on": [],
            "nullable": False,
        },
        {
            "name": phone_name,
            "dtype": "string",
            "description": "The person's phone number (E.164).",
            "gen": "e164_phone",
            "params": {"country_codes": [profile["phone_country_code"]]},
            "depends_on": [],
            "nullable": False,
        },
        {
            "name": account_name,
            "dtype": "string",
            "description": "Billing/account identifier tied to the person.",
            "gen": "id_mirror",
            "params": {"prefix": "ACC-", "source_field": identifier_name, "source_prefix": prefix},
            "depends_on": [identifier_name],
            "nullable": False,
        },
        {
            "name": "event_timestamp",
            "dtype": "datetime",
            "description": "Timestamp the triggering event occurred.",
            "gen": "recent_datetime",
            "params": {"days_back": 30},
            "depends_on": [],
            "nullable": False,
        },
    ]
    return base + [dict(m) for m in _MANDATORY_VARIABLES]


def _build_alias_map(industry_key: str) -> dict[str, str]:
    """Synonym -> canonical name mapping for this industry (e.g. telecom canonicalizes
    to subscriber_id, banking/retail/etc. canonicalize to customer_id)."""
    alias_map: dict[str, str] = {}
    for concept, synonyms in _CONCEPT_SYNONYMS.items():
        canonical = _canonical_name(concept, industry_key)
        for syn in synonyms:
            if syn != canonical:
                alias_map[syn] = canonical
    return alias_map


def _build_glossary_text(industry_key: str) -> str:
    lines = [
        "Canonical field names for this industry — if a variable represents one of these "
        "concepts, you MUST reuse the EXACT name below instead of inventing a synonym:"
    ]
    for concept, description in _CONCEPT_DESCRIPTIONS.items():
        canonical = _canonical_name(concept, industry_key)
        note = " (ALWAYS gen=\"prefixed_int\", never prefixed_uuid — enforced for consistency)" if concept == "identifier" else ""
        lines.append(f"- {canonical:<26} — {description}{note}")
    return "\n".join(lines) + "\n"

# Keep this in sync with the dispatch table in agents/generator_agent.py.
_GENERATOR_DOCS = """
Allowed "gen" types and their required "params" (use ONLY these — nothing else):
- prefixed_int      params: {prefix: str, digits: int}
- id_mirror         params: {prefix: str, source_field: str, source_prefix: str}  (needs depends_on: [source_field])
- e164_phone        params: {country_codes: [str]}
- constant          params: {value: any}
- weighted_choice   params: {choices: [any], weights: [float]}  (weights sum to 1.0)
- uniform           params: {min: float, max: float}
- lognormal         params: {mu: float, sigma: float, min: float, max: float}
- lognormal_int     params: {mu: float, sigma: float, min: int, max: int}
- beta              params: {alpha: float, beta: float}   (produces 0..1)
- segment_range     params: {<segment_value>: {min: float, max: float}, ...}  (keyed by a categorical field's values)
- uniform_bounded   params: {hi_field: str, lo: float}   (bounded by another numeric field's value)
- recent_datetime   params: {days_back: int}   (ISO timestamp within last N days)
- ts_offset         params: {base_field: str, min_sec: int, max_sec: int}   (needs depends_on: [base_field])
- ts_add_field      params: {base_field: str, add_seconds_field: str}   (needs depends_on: [base_field, add_seconds_field])
- date_offset       params: {base_field: str, days: int}   (needs depends_on: [base_field])
- prefixed_uuid     params: {prefix: str}
- tx_id             params: {prefix: str}   (uses an existing event_timestamp field if present)
- formula           top-level key "formula": "<python expression using other numeric field names>" (no params needed)

Rules:
1. List variables in dependency order — a variable using depends_on/base_field/source_field/hi_field
   must come AFTER the field it depends on.
2. Every variable object must have: name, dtype ("string"|"float"|"int"|"categorical"|"datetime"|"bool"),
   description (plain English, what this field means and why), gen, params, depends_on (list, may be empty),
   nullable (bool).
3. Cover identifier fields, the primary business-event/decision fields relevant to the
   business scenario, and any timestamps/financial fields implied by the scenario description.
4. Produce AS MANY high-quality, relevant, non-redundant variables as you can — aim for the maximum
   allowed (""" + str(MAX_VARIABLES) + """), and never fewer than """ + str(MIN_VARIABLES) + """. Prefer
   completeness: include identity, segmentation, financial, behavioral, timestamp, channel, and
   outcome/decision fields.
5. ALWAYS include these 2 variables (verbatim names, anywhere in the list):
   customer_lifetime_value, loyalty_tier_customer.
"""

# If a variable represents one of these concepts, reuse the exact name — keeps field
# naming consistent across every scenario generated for the same industry/domain.
# NOTE: this is now built dynamically per-industry inside propose() via
# _build_glossary_text() / _build_alias_map() — see _CONCEPT_CANONICAL above.

_SYSTEM = """
You are the Scenario Designer Agent for a synthetic data generation platform.
Given an industry, a domain, and a business scenario description, invent a complete
synthetic-data scenario definition.

The request will specify a TARGET INDUSTRY and COUNTRY with real-world conventions for
that industry in that country (regulator, currency, market character, typical
product/plan types, phone number format, identity/KYC rules). Every variable you invent
— product/plan names/types, currency-denominated fields, phone number params,
identity/KYC fields, transaction denominations — MUST reflect that industry and
country's actual real-world standards, not a generic or telecom/US-default assumption.
For example, a prepaid-dominant telecom market with mandated biometric KYC looks very
different from a bank account market regulated by a central bank, which looks different
again from an e-commerce retail market — product types, identity fields and
transaction/recharge denominations should follow the target industry and country, not a
one-size-fits-all telecom template.

""" + _GENERATOR_DOCS + """
Return a JSON object with exactly these keys:
  - "data_type": "transactional" or "aggregational"
  - "entity_key": str — for transactional output, the primary business entity variable used to group events; MUST exactly match one variable name. For aggregational output return null
  - "events": [ {"event_type": str, "sequence": int, "fields": [str]} ] — REQUIRED for transactional output; ordered business events; choose the number appropriate to the scenario. For aggregational output return []
  - "label": str               — short human-readable scenario title
  - "journey": str              — the domain/journey name
  - "description": str          — 1-3 sentence description combining businessScenario, businessResponse, expectedOutcome
  - "variables": [ {name, dtype, description, gen, params, depends_on, nullable} ]
  - "field_order": [str]        — variable names in the exact order they should be generated/output

For transactional output, choose entity_key as the variable that identifies the primary business entity whose events should be grouped together. For example subscriber_id, customer_id, account_id, merchant_id, shipment_id, patient_id, etc. Do not hard-code an industry; choose from the variables you actually define.

For transactional output, events are the business events that create separate rows.
Each event must have a unique event_type, increasing sequence starting at 1, and a fields
list containing only variables that are meaningful on that event. Always include relevant
identity/common fields through the generator; event fields should describe what happens
at that step. Do not invent event types unrelated to the supplied business scenario.
For transactional output, each event may occur multiple times for the same entity. You MAY
include min_occurrences and max_occurrences on an event; if omitted, the backend defaults to
1..10 occurrences per entity so the generated transactional response can contain multiple
records per event. The response layer will still return only the latest 10 records while
reporting the full event totalCount.
"""


class ScenarioDesignerAgent:
    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    def propose(
        self,
        industry_type: str,
        domain: str,
        business_scenario: str,
        business_response: str | None = None,
        expected_outcome: str | None = None,
        scenario_id: str | None = None,
        scenario_type: str | None = None,
        country: str | None = None,
        use_case: str | None = None,
        type_of_data: str = "aggregational",
    ) -> dict:
        history = get_feedback_history(domain, business_scenario)
        feedback_block = (
            "\n".join(f"- {f}" for f in history)
            if history else "(none yet)"
        )
        profile = get_profile(industry_type, country)
        industry_key = match_industry_key(industry_type)
        glossary_text = _build_glossary_text(industry_key)
        base_variables = _build_base_variables(industry_key, profile)
        base_names = {v["name"] for v in base_variables}
        preinjected_note = (
            f"These {len(base_variables)} fields are auto-injected and ALREADY generated — "
            f"do NOT include them in your \"variables\" list, just reference their exact names "
            f"in depends_on if a scenario-specific field needs one of them: {sorted(base_names)}\n"
            f"Return ONLY the ADDITIONAL scenario-specific variables needed — aim for "
            f"{max(1, MIN_VARIABLES - len(base_variables))} to {MAX_VARIABLES - len(base_variables)} of them.\n"
        )

        prompt = (
            f"Scenario ID: {scenario_id or '(assign automatically)'}\n"
            f"Scenario type/outcome: {scenario_type or '(not provided — infer a sensible one)'}\n"
            f"Industry: {industry_type}\n"
            f"Domain: {domain}\n"
            f"Use case: {use_case or '(not provided — infer a sensible one from the business scenario)'}\n"
            f"Output data type: {type_of_data}\n"
            f"Business scenario: {business_scenario}\n"
            f"Business response: {business_response or '(not provided — infer a sensible one)'}\n"
            f"Expected outcome: {expected_outcome or '(not provided — infer a sensible one)'}\n\n"
            f"Target country: {profile['country_name']} ({country or 'not specified'})\n"
            f"Regulator: {profile['regulator']}\n"
            f"Currency: {profile['currency']}\n"
            f"Phone country code: {profile['phone_country_code']}; format: {profile['phone_format']}\n"
            f"Market character: {profile['market_character']}\n"
            f"Typical product/plan types in this industry+country: {profile['product_types']}\n"
            f"Identity/KYC notes: {profile['identity_notes']}\n"
            f"Typical transaction denominations in {profile['currency']}: {profile['typical_denominations']}\n\n"
            f"{glossary_text}\n"
            f"{preinjected_note}\n"
            f"Feedback from prior attempts at this same domain/scenario (learn from these, "
            f"avoid repeating past mistakes):\n{feedback_block}\n\n"
            "Design the scenario now, following this industry and country's real-world standards."
        )

        logger.info("[ScenarioDesigner] Proposing scenario for domain=%s industry=%s country=%s", domain, industry_type, country)
        result = self._llm.generate_json(_SYSTEM, prompt, temperature=0.3)
        result["data_type"] = type_of_data
        llm_variables = [v for v in result.get("variables", []) if isinstance(v, dict) and v.get("name")]
        # Defensive: drop any base field the LLM ignored the instruction and re-emitted anyway.
        llm_variables = [v for v in llm_variables if v["name"] not in base_names]
        if not llm_variables and not base_variables:
            raise ValueError("Gemini response contained no usable variables")
        llm_variables = self._normalize_field_names(llm_variables, industry_key)
        combined = base_variables + llm_variables
        result["variables"] = self._enforce_variable_bounds(combined, prompt, base_names, industry_key)
        result["field_order"] = [v["name"] for v in result["variables"]]
        # Resolve entity_key only AFTER base variables are injected. The primary entity
        # identifier is normally one of those auto-injected fields (e.g. subscriber_id),
        # so validating against Gemini's scenario-specific variables alone is incorrect.
        result["entity_key"] = self._normalize_entity_key(
            result.get("entity_key"), result["variables"], industry_key, type_of_data
        )
        result["events"] = self._normalize_events(result.get("events", []), result["variables"], type_of_data, industry_key)
        return result


    def _normalize_entity_key(self, entity_key: object, variables: list[dict], industry_key: str, type_of_data: str) -> str | None:
        if type_of_data != "transactional":
            return None
        valid_names = [str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")]
        requested = str(entity_key or "").strip()
        if requested in valid_names:
            return requested
        # Safe deterministic fallback to the canonical primary identifier for the industry.
        fallback = _canonical_name("identifier", industry_key)
        if fallback in valid_names:
            return fallback
        # Last resort: first variable. This should be unreachable because base variables
        # always inject an identifier.
        return valid_names[0] if valid_names else None

    def _normalize_events(self, events: list, variables: list[dict], type_of_data: str, industry_key: str) -> list[dict]:
        """Normalize the LLM event definitions and keep event field references valid."""
        if type_of_data != "transactional":
            return []
        alias_map = _build_alias_map(industry_key)
        valid_names = {v["name"] for v in variables}
        normalized = []
        for index, event in enumerate(events if isinstance(events, list) else [], start=1):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", "")).strip().upper().replace(" ", "_")
            if not event_type:
                continue
            fields = []
            for name in event.get("fields", []) if isinstance(event.get("fields", []), list) else []:
                canonical = alias_map.get(str(name).strip().lower(), str(name).strip())
                if canonical in valid_names and canonical not in fields:
                    fields.append(canonical)
            min_occurrences = max(1, int(event.get("min_occurrences", 1)))
            max_occurrences = max(min_occurrences, min(1000, int(event.get("max_occurrences", 10))))
            normalized.append({
                "event_type": event_type,
                "sequence": index,
                "fields": fields,
                "min_occurrences": min_occurrences,
                "max_occurrences": max_occurrences,
            })
        if not normalized:
            normalized = [{
                "event_type": "BUSINESS_EVENT",
                "sequence": 1,
                "fields": [],
                "min_occurrences": 1,
                "max_occurrences": 10,
            }]
        return normalized[:8]

    def _normalize_field_names(self, variables: list[dict], industry_key: str = "generic") -> list[dict]:
        """Collapse known synonyms (customer_id, phone_number, ...) to the canonical name
        so the same concept is always named the same way across every scenario."""
        alias_map = _build_alias_map(industry_key)
        rename_map = {
            v["name"]: alias_map[v["name"].strip().lower()]
            for v in variables
            if v["name"].strip().lower() in alias_map and alias_map[v["name"].strip().lower()] != v["name"]
        }
        if not rename_map:
            return variables

        seen: set[str] = set()
        result: list[dict] = []
        for v in variables:
            new_name = rename_map.get(v["name"], v["name"])
            if new_name in seen:
                continue  # drop duplicate created by the rename (e.g. both customer_id & subscriber_id present)
            seen.add(new_name)
            v = dict(v)
            v["name"] = new_name
            params = v.get("params")
            if isinstance(params, dict):
                for key in _REF_PARAM_KEYS:
                    if key in params and params[key] in rename_map:
                        params[key] = rename_map[params[key]]
            v["depends_on"] = [rename_map.get(d, d) for d in v.get("depends_on", [])]
            result.append(v)
        return result

    def _enforce_variable_bounds(
        self, variables: list[dict], prompt: str, protected_names: set[str], industry_key: str = "generic"
    ) -> list[dict]:
        """Trim/expand the LLM-supplied portion only — protected_names (base + mandatory
        fields, pre-injected in propose()) are never removed and never re-requested."""
        if len(variables) > MAX_VARIABLES:
            logger.warning("[ScenarioDesigner] %d variables produced, truncating to %d",
                           len(variables), MAX_VARIABLES)
            overflow = len(variables) - MAX_VARIABLES
            removed = 0
            kept_reversed = []
            for v in reversed(variables):
                if removed < overflow and v["name"] not in protected_names:
                    removed += 1
                    continue
                kept_reversed.append(v)
            variables = list(reversed(kept_reversed))

        if len(variables) < _EXPAND_TARGET:
            logger.info("[ScenarioDesigner] Only %d variables produced, requesting more (target ~%d)",
                        len(variables), _EXPAND_TARGET)
            scenario_specific = [v for v in variables if v["name"] not in protected_names]
            expand_prompt = (
                prompt + "\n\n"
                f"Your previous attempt produced only {len(scenario_specific)} scenario-specific variables:\n"
                f"{[v['name'] for v in scenario_specific]}\n"
                f"Add as many NEW, distinct, relevant variables as possible so the scenario-specific TOTAL "
                f"reaches close to {MAX_VARIABLES - len(protected_names)}. Do NOT include the auto-injected "
                f"fields again. Return the FULL combined scenario-specific variables list (previous ones + "
                "new ones) under the \"variables\" key, in dependency order."
            )
            try:
                retry = self._llm.generate_json(_SYSTEM, expand_prompt, temperature=0.3)
                retried_vars = [v for v in retry.get("variables", []) if isinstance(v, dict) and v.get("name")]
                retried_vars = [v for v in retried_vars if v["name"] not in protected_names]
                retried_vars = self._normalize_field_names(retried_vars, industry_key)
                if len(retried_vars) > len(scenario_specific):
                    protected = [v for v in variables if v["name"] in protected_names]
                    # protected is always small (base + mandatory fields, ~7) relative to
                    # MAX_VARIABLES (35), so this slice never truncates a protected field.
                    variables = (protected + retried_vars)[:MAX_VARIABLES]
            except Exception as exc:
                logger.warning("[ScenarioDesigner] Expansion retry failed: %s", exc)

        return variables
