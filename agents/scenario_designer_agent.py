"""
Agent 0 — Scenario Designer Agent
Given a plain-English business scenario description, asks Gemini to invent a
new scenario (LB-04, LB-05, ...) and its full variable catalog, expressed in
the same generator-type vocabulary that agents/data_generation_agent.py already
knows how to execute — so the result can be run through the existing
4-agent pipeline with no new generator code.
"""
from __future__ import annotations
import logging
import ast
import math

from config.industry_profiles import get_profile, match_industry_key
from core.llm_client import GeminiClient
from core.dynamic_scenarios import get_feedback_history
from core.runtime_cache import get_schema, set_schema

logger = logging.getLogger(__name__)

MIN_VARIABLES = 15
MAX_VARIABLES = 35
# Proposals target the maximum catalog size. Telecom can reach this target locally
# from its reusable schema catalog, avoiding an extra LLM round trip.
_EXPAND_TARGET = MAX_VARIABLES

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

# ── Telecom-only client overrides ───────────────────────────────────────────────
# Every branch below is gated on industry_key == "telecom" at the call site, so no
# other industry's proposed scenarios are affected by any of these constants.
_TELECOM_CLV_CHOICES = ["low", "medium", "high"]
_TELECOM_CLV_WEIGHTS = [0.34, 0.43, 0.23]
_TELECOM_SEGMENT_CHOICES = [
    "Occasional Rechargers", "Habitual Rechargers", "Valued Customers",
    "Heavy Data Users", "Frequent Data Exhausters", "Low Data Users",
]
_TELECOM_SEGMENT_WEIGHTS = [0.22, 0.18, 0.15, 0.20, 0.15, 0.10]
_TELECOM_SERVICE_PROVIDER_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
_TELECOM_TRIGGER_CAUSE_SYNONYMS = ["trigger_type", "triggertype", "trigger", "event_trigger_type", "trigger_reason"]
_TELECOM_TRIGGER_CAUSE_CHOICES = ["LOW_BALANCE_THRESHOLD", "VALIDITY_EXPIRY_WARNING", "ZERO_BALANCE"]
_TELECOM_TRIGGER_CAUSE_WEIGHTS = [0.5, 0.3, 0.2]
_TELECOM_RECHARGE_STATUS_SYNONYMS = ["recharge_status", "topup_status", "recharge_result", "recharge_outcome", "topup_result"]
_TELECOM_RECHARGE_STATUS_CHOICES = ["SUCCESS", "FAILED", "PENDING", "GATEWAY_TIMEOUT"]
_TELECOM_RECHARGE_STATUS_WEIGHTS = [0.70, 0.15, 0.10, 0.05]
# loyalty_tier_customer is intentionally omitted for telecom — redundant with customer_lifetime_value.
_TELECOM_DROPPED_MANDATORY_NAMES = {"loyalty_tier_customer"}
# Replaces a raw churn_risk_score with the inputs needed to derive one downstream.
_TELECOM_CHURN_HELPER_VARIABLES: list[dict] = [
    {
        "name": "historical_low_balance_frequency",
        "dtype": "int",
        "description": "Number of times the subscriber hit a low-balance trigger in the trailing 90 days.",
        "gen": "lognormal_int",
        "params": {"mu": 1.2, "sigma": 0.6, "min": 0, "max": 30},
        "depends_on": [],
        "nullable": False,
    },
    {
        "name": "offer_acceptance_rate_historical",
        "dtype": "float",
        "description": "Historical fraction of recharge/top-up offers accepted by the subscriber.",
        "gen": "uniform",
        "params": {"min": 0.0, "max": 1.0},
        "depends_on": [],
        "nullable": False,
    },
    {
        "name": "preferred_topup_channel",
        "dtype": "categorical",
        "description": "Channel the subscriber most frequently uses to top up/recharge.",
        "gen": "weighted_choice",
        "params": {"choices": ["MY_ACCOUNT_APP", "RETAILER_POS", "WEB_PORTAL", "USSD"], "weights": [0.45, 0.30, 0.15, 0.10]},
        "depends_on": [],
        "nullable": False,
    },
]
_TELECOM_CHURN_NAME_MARKER = "churn"
_KYC_NAME_MARKER = "kyc"
_UPI_FIELD_NAME_MARKERS = ("payment", "pay_method", "channel")
_UPI_VALUE_MARKER = "upi"
_UPI_MASK_LABEL = "MASKED"

# Telecom-only generated-data vocabulary contract: UPI must not appear in Telecom
# generated output. Any Telecom UPI-related payment terminology is represented as DIGITAL_WALLET.
_UPI_CANONICAL_OUTPUT = "DIGITAL_WALLET"

# Telecom-wide mandatory client contract. These fields are injected for EVERY
# Telecom proposal, independent of scenario id. Scenario-specific requirements
# below are layered on top only for the requested LB-06/LB-07 journeys.
_TELECOM_CORE_MANDATORY = []



# Reusable Telecom proposal catalog. These definitions are immutable templates used
# only to fill the requested 35-field catalog when Gemini returns fewer fields.
# Keeping them in the process cache avoids a second Gemini round-trip while preserving
# the 35-variable target. Scenario-specific fields from Gemini always take precedence.
_TELECOM_CACHED_PROPOSAL_CATALOG = [
    {"name":"subscriber_segment","dtype":"categorical","description":"Subscriber usage/behavior segment.","gen":"weighted_choice","params":{"choices":list(_TELECOM_SEGMENT_CHOICES),"weights":list(_TELECOM_SEGMENT_WEIGHTS)},"depends_on":[],"nullable":False},
    {"name":"service_provider","dtype":"categorical","description":"Real-world telecom operator/service-provider brand appropriate to the target country.","gen":"weighted_choice","params":{"choices":["Vodafone","Orange","Telefónica","Deutsche Telekom"],"weights":[0.40,0.30,0.20,0.10]},"depends_on":[],"nullable":False},
    {"name":"rate_plan_code","dtype":"categorical","description":"Synthetic rate plan associated with the service provider.","gen":"weighted_choice","params":{"choices":["PREPAID_DAILY","PREPAID_MONTHLY","POSTPAID_INDIVIDUAL","POSTPAID_FAMILY","DATA_ONLY"],"weights":[0.25,0.25,0.20,0.20,0.10]},"depends_on":["service_provider"],"nullable":False},
    {"name":"trigger_cause","dtype":"categorical","description":"Cause that triggered the telecom journey.","gen":"weighted_choice","params":{"choices":list(_TELECOM_TRIGGER_CAUSE_CHOICES),"weights":list(_TELECOM_TRIGGER_CAUSE_WEIGHTS)},"depends_on":[],"nullable":False},
    {"name":"recharge_status","dtype":"categorical","description":"Recharge/top-up outcome.","gen":"weighted_choice","params":{"choices":list(_TELECOM_RECHARGE_STATUS_CHOICES),"weights":list(_TELECOM_RECHARGE_STATUS_WEIGHTS)},"depends_on":[],"nullable":False},
    {"name":"balance_before","dtype":"float","description":"Subscriber balance before a relevant transaction.","gen":"uniform","params":{"min":0.0,"max":1000.0},"depends_on":[],"nullable":False},
    {"name":"recharge_amount","dtype":"float","description":"Recharge/top-up amount.","gen":"uniform","params":{"min":10.0,"max":500.0},"depends_on":[],"nullable":False},
    {"name":"payment_method","dtype":"categorical","description":"Synthetic payment method.","gen":"weighted_choice","params":{"choices":["DEBIT_CARD","CREDIT_CARD","DIGITAL_WALLET","NETBANKING"],"weights":[0.35,0.30,0.20,0.15]},"depends_on":[],"nullable":False},
    {"name":"failure_reason","dtype":"categorical","description":"Recharge/payment failure reason.","gen":"weighted_choice","params":{"choices":["INSUFFICIENT_FUNDS","GATEWAY_TIMEOUT","NETWORK_FAILURE","BANK_DECLINE","AUTHENTICATION_FAILURE","PAYMENT_INSTRUMENT_UNAVAILABLE"],"weights":[0.30,0.15,0.15,0.15,0.10,0.15]},"depends_on":[],"nullable":False},
    {"name":"recovery_action","dtype":"categorical","description":"Recovery action after a failed recharge.","gen":"weighted_choice","params":{"choices":["RETRY_SAME_METHOD","USE_ALTERNATE_PAYMENT_METHOD","REAUTHENTICATE","CONTACT_SUPPORT"],"weights":[0.35,0.35,0.15,0.15]},"depends_on":[],"nullable":False},
    {"name":"recovery_status","dtype":"categorical","description":"Recovery outcome.","gen":"weighted_choice","params":{"choices":["RECOVERY_PENDING","RECOVERED_SAME_METHOD","RECOVERED_ALTERNATE_METHOD","ESCALATED","ABANDONED"],"weights":[0.10,0.35,0.35,0.10,0.10]},"depends_on":[],"nullable":False},
    {"name":"recovery_channel","dtype":"categorical","description":"Recovery guidance/action channel.","gen":"weighted_choice","params":{"choices":["APP","SMS","WEB","USSD","CONTACT_CENTER"],"weights":[0.35,0.20,0.15,0.15,0.15]},"depends_on":[],"nullable":False},
    {"name":"transaction_status","dtype":"categorical","description":"Transaction processing status.","gen":"weighted_choice","params":{"choices":["SUCCESS","FAILED","PENDING"],"weights":[0.75,0.15,0.10]},"depends_on":[],"nullable":False},
    {"name":"settlement_status","dtype":"categorical","description":"Payment settlement status.","gen":"weighted_choice","params":{"choices":["SETTLED","PENDING","REVERSED"],"weights":[0.85,0.10,0.05]},"depends_on":[],"nullable":False},
    {"name":"retry_count","dtype":"int","description":"Number of recharge retries in the journey.","gen":"uniform","params":{"min":0,"max":4},"depends_on":[],"nullable":False},
    {"name":"recovery_timestamp","dtype":"datetime","description":"Timestamp of a recovery action.","gen":"ts_offset","params":{"source_field":"event_timestamp","min_seconds":60,"max_seconds":3600},"depends_on":["event_timestamp"],"nullable":False},
    {"name":"final_journey_status","dtype":"categorical","description":"Final telecom journey outcome.","gen":"weighted_choice","params":{"choices":["RECOVERED","UNRESOLVED_FAILURE","ABANDONED","ESCALATED"],"weights":[0.60,0.20,0.10,0.10]},"depends_on":[],"nullable":False},
    {"name":"expected_balance","dtype":"float","description":"Expected balance after a successful top-up.","gen":"formula","params":{},"depends_on":["balance_before","recharge_amount"],"nullable":False,"formula":"balance_before + recharge_amount"},
    {"name":"observed_balance","dtype":"float","description":"Observed account balance during an exception.","gen":"uniform","params":{"min":0.0,"max":5000.0},"depends_on":[],"nullable":False},
    {"name":"balance_update_status","dtype":"categorical","description":"Whether the balance update completed.","gen":"weighted_choice","params":{"choices":["UPDATED","FAILED","DELAYED"],"weights":[0.75,0.15,0.10]},"depends_on":[],"nullable":False},
    {"name":"balance_variance","dtype":"float","description":"Difference between expected and observed balance.","gen":"formula","params":{},"depends_on":["expected_balance","observed_balance"],"nullable":False,"formula":"expected_balance - observed_balance"},
    {"name":"exception_detected_flag","dtype":"boolean","description":"Whether a balance/payment exception was detected.","gen":"weighted_choice","params":{"choices":[True,False],"weights":[0.15,0.85]},"depends_on":[],"nullable":False},
    {"name":"verification_status","dtype":"categorical","description":"Transaction verification result.","gen":"weighted_choice","params":{"choices":["REQUESTED","VERIFIED","FAILED","PENDING"],"weights":[0.10,0.75,0.05,0.10]},"depends_on":[],"nullable":False},
    {"name":"verification_attempt_count","dtype":"int","description":"Number of verification attempts.","gen":"uniform","params":{"min":1,"max":3},"depends_on":[],"nullable":False},
    {"name":"reconciliation_status","dtype":"categorical","description":"Balance reconciliation outcome.","gen":"weighted_choice","params":{"choices":["RESOLVED","PENDING","MANUAL_REVIEW"],"weights":[0.75,0.10,0.15]},"depends_on":[],"nullable":False},
    {"name":"final_status","dtype":"categorical","description":"Final transaction/customer status.","gen":"weighted_choice","params":{"choices":["RESOLVED","PENDING","MANUALLY_RESOLVED","NOT_RESOLVED"],"weights":[0.75,0.10,0.10,0.05]},"depends_on":[],"nullable":False},
    {"name":"final_balance","dtype":"float","description":"Final reconciled subscriber balance.","gen":"uniform","params":{"min":1.0,"max":5000.0},"depends_on":[],"nullable":False},
    {"name":"resolution_type","dtype":"categorical","description":"How an exception was resolved.","gen":"weighted_choice","params":{"choices":["AUTO_RECONCILIATION","MANUAL_REVIEW","BALANCE_CORRECTION","STATUS_CLARIFICATION"],"weights":[0.45,0.20,0.25,0.10]},"depends_on":[],"nullable":False},
    {"name":"parent_transaction_id","dtype":"string","description":"Parent failed transaction identifier for a retry.","gen":"constant","params":{"value":"PARENT_TXN"},"depends_on":[],"nullable":False},
]

_LB06_VARIABLES = [
    {"name":"balance_before","dtype":"float","description":"Subscriber balance immediately before the recharge/recovery transaction.","gen":"uniform","params":{"min":0.0,"max":100.0},"depends_on":[],"nullable":False},
    {"name":"recharge_amount","dtype":"float","description":"Recharge/top-up amount.","gen":"uniform","params":{"min":10.0,"max":500.0},"depends_on":[],"nullable":False},
    {"name":"payment_method","dtype":"categorical","description":"Synthetic payment method used for recharge/retry.","gen":"weighted_choice","params":{"choices":["DEBIT_CARD","CREDIT_CARD","DIGITAL_WALLET","NETBANKING"],"weights":[0.35,0.30,0.20,0.15]},"depends_on":[],"nullable":False},
    {"name":"failure_reason","dtype":"categorical","description":"Reason for failed recharge attempt.","gen":"weighted_choice","params":{"choices":["INSUFFICIENT_FUNDS","GATEWAY_TIMEOUT","NETWORK_FAILURE","BANK_DECLINE","AUTHENTICATION_FAILURE","PAYMENT_INSTRUMENT_UNAVAILABLE"],"weights":[0.30,0.15,0.15,0.15,0.10,0.15]},"depends_on":[],"nullable":False},
    {"name":"recovery_action","dtype":"categorical","description":"Recovery action actually taken after a failed recharge.","gen":"weighted_choice","params":{"choices":["RETRY_SAME_METHOD","USE_ALTERNATE_PAYMENT_METHOD","REAUTHENTICATE","CONTACT_SUPPORT"],"weights":[0.35,0.35,0.15,0.15]},"depends_on":[],"nullable":False},
    {"name":"recovery_status","dtype":"categorical","description":"Outcome of the recovery action.","gen":"weighted_choice","params":{"choices":["RECOVERY_PENDING","RECOVERED_SAME_METHOD","RECOVERED_ALTERNATE_METHOD","ESCALATED","ABANDONED"],"weights":[0.10,0.35,0.35,0.10,0.10]},"depends_on":[],"nullable":False},
    {"name":"recovery_channel","dtype":"categorical","description":"Channel used for recovery guidance/action.","gen":"weighted_choice","params":{"choices":["APP","SMS","WEB","USSD","CONTACT_CENTER"],"weights":[0.35,0.20,0.15,0.15,0.15]},"depends_on":[],"nullable":False},
    {"name":"recovery_timestamp","dtype":"datetime","description":"Timestamp at which recovery action occurred.","gen":"ts_offset","params":{"source_field":"event_timestamp","min_seconds":60,"max_seconds":3600},"depends_on":["event_timestamp"],"nullable":False},
    {"name":"parent_transaction_id","dtype":"string","description":"Transaction id of the failed recharge that this retry/recovery belongs to.","gen":"constant","params":{"value":"PARENT_TXN"},"depends_on":[],"nullable":False},
    {"name":"retry_count","dtype":"int","description":"Number of recharge retry attempts in this journey.","gen":"uniform","params":{"min":0,"max":4},"depends_on":[],"nullable":False},
    {"name":"final_journey_status","dtype":"categorical","description":"Final outcome of the recovery journey.","gen":"weighted_choice","params":{"choices":["RECOVERED","UNRESOLVED_FAILURE","ABANDONED","ESCALATED"],"weights":[0.60,0.20,0.10,0.10]},"depends_on":[],"nullable":False},
]
_LB06_EVENTS = [
    ("LOW_BALANCE_DETECTED",1,["subscriber_id","account_id","subscriber_msisdn","balance_after","trigger_cause"]),
    ("TOPUP_ATTEMPT",2,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","recharge_amount","recharge_status","failure_reason","payment_method"]),
    ("TOPUP_FAILURE",3,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","recharge_amount","recharge_status","failure_reason","payment_method"]),
    ("RECOVERY_GUIDANCE",4,["subscriber_id","account_id","subscriber_msisdn","balance_after","recovery_channel","recovery_action"]),
    ("TOPUP_RETRY",5,["subscriber_id","account_id","subscriber_msisdn","balance_after","recharge_amount","recharge_status","parent_transaction_id","retry_count"]),
    ("TOPUP_SUCCESS",6,["subscriber_id","account_id","subscriber_msisdn","balance_after","recharge_amount","recharge_status","recovery_status","recovery_timestamp","parent_transaction_id","final_journey_status"]),
]

_LB07_VARIABLES = [
    {"name":"balance_before","dtype":"float","description":"Subscriber balance before the top-up.","gen":"uniform","params":{"min":0.0,"max":100.0},"depends_on":[],"nullable":False},
    {"name":"recharge_amount","dtype":"float","description":"Top-up amount.","gen":"uniform","params":{"min":10.0,"max":500.0},"depends_on":[],"nullable":False},
    {"name":"transaction_status","dtype":"categorical","description":"Payment transaction status.","gen":"constant","params":{"value":"SUCCESS"},"depends_on":[],"nullable":False},
    {"name":"expected_balance","dtype":"float","description":"Expected balance after successful top-up.","gen":"uniform","params":{"min":1.0,"max":5000.0},"depends_on":[],"nullable":False},
    {"name":"observed_balance","dtype":"float","description":"Observed balance immediately after the successful payment before reconciliation.","gen":"uniform","params":{"min":0.0,"max":4999.0},"depends_on":[],"nullable":False},
    {"name":"balance_update_status","dtype":"categorical","description":"Whether the balance update succeeded at first attempt.","gen":"weighted_choice","params":{"choices":["UPDATED","FAILED","DELAYED"],"weights":[0.10,0.75,0.15]},"depends_on":[],"nullable":False},
    {"name":"balance_variance","dtype":"float","description":"Observed balance minus expected balance during the discrepancy.","gen":"uniform","params":{"min":-500.0,"max":0.0},"depends_on":[],"nullable":False},
    {"name":"exception_reason","dtype":"categorical","description":"Reason for successful payment with missing/delayed balance update.","gen":"weighted_choice","params":{"choices":["LEDGER_SYNCHRONIZATION_DELAY","CORE_SYSTEM_UPDATE_DELAY","CACHE_REFRESH_ISSUE","EVENT_PROCESSING_LAG","RECONCILIATION_MISMATCH","BACKEND_TIMEOUT_AFTER_SUCCESSFUL_PROCESSING"],"weights":[0.20,0.20,0.15,0.15,0.15,0.15]},"depends_on":[],"nullable":False},
    {"name":"exception_detected_flag","dtype":"boolean","description":"Whether a balance discrepancy was detected.","gen":"constant","params":{"value":True},"depends_on":[],"nullable":False},
    {"name":"verification_status","dtype":"categorical","description":"Verification status after discrepancy detection.","gen":"weighted_choice","params":{"choices":["PENDING","VERIFIED","FAILED"],"weights":[0.10,0.80,0.10]},"depends_on":[],"nullable":False},
    {"name":"verification_attempt_count","dtype":"int","description":"Number of verification attempts.","gen":"uniform","params":{"min":1,"max":3},"depends_on":[],"nullable":False},
    {"name":"settlement_status","dtype":"categorical","description":"Payment settlement status.","gen":"weighted_choice","params":{"choices":["SETTLED","PENDING","REVERSED"],"weights":[0.85,0.10,0.05]},"depends_on":[],"nullable":False},
    {"name":"reconciliation_status","dtype":"categorical","description":"Balance reconciliation outcome.","gen":"weighted_choice","params":{"choices":["RESOLVED","PENDING","MANUAL_REVIEW"],"weights":[0.75,0.10,0.15]},"depends_on":[],"nullable":False},
    {"name":"final_status","dtype":"categorical","description":"Final status communicated after reconciliation.","gen":"weighted_choice","params":{"choices":["RESOLVED","PENDING","MANUALLY_RESOLVED","NOT_RESOLVED"],"weights":[0.75,0.10,0.10,0.05]},"depends_on":[],"nullable":False},
    {"name":"final_balance","dtype":"float","description":"Final reconciled balance after correction or status clarification.","gen":"uniform","params":{"min":1.0,"max":5000.0},"depends_on":[],"nullable":False},
    {"name":"resolution_type","dtype":"categorical","description":"How the discrepancy was resolved.","gen":"weighted_choice","params":{"choices":["AUTO_RECONCILIATION","MANUAL_REVIEW","BALANCE_CORRECTION","STATUS_CLARIFICATION"],"weights":[0.45,0.20,0.25,0.10]},"depends_on":[],"nullable":False},
]
_LB07_EVENTS = [
    ("TOPUP_INITIATED",1,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","recharge_amount","transaction_status"]),
    ("PAYMENT_AUTHORIZED",2,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","recharge_amount","transaction_status"]),
    ("PAYMENT_SETTLED",3,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","recharge_amount","transaction_status","settlement_status"]),
    ("BALANCE_UPDATE_FAILED",4,["subscriber_id","account_id","subscriber_msisdn","balance_before","balance_after","expected_balance","observed_balance","balance_update_status","balance_variance","exception_reason","transaction_status"]),
    ("DISCREPANCY_DETECTED",5,["subscriber_id","account_id","subscriber_msisdn","balance_after","expected_balance","observed_balance","balance_variance","exception_detected_flag"]),
    ("VERIFICATION_REQUESTED",6,["subscriber_id","account_id","subscriber_msisdn","balance_after","verification_status","verification_attempt_count"]),
    ("TRANSACTION_LOOKUP",7,["subscriber_id","account_id","subscriber_msisdn","balance_after","settlement_status","verification_status"]),
    ("RECONCILIATION_STARTED",8,["subscriber_id","account_id","subscriber_msisdn","balance_after","reconciliation_status","resolution_type"]),
    ("STATUS_CONFIRMED",9,["subscriber_id","account_id","subscriber_msisdn","balance_after","verification_status","settlement_status","reconciliation_status","final_status"]),
    ("CUSTOMER_NOTIFIED",10,["subscriber_id","account_id","subscriber_msisdn","balance_after","final_status"]),
    ("BALANCE_CORRECTED",11,["subscriber_id","account_id","subscriber_msisdn","balance_after","expected_balance","final_balance","resolution_type"]),
    ("CASE_RESOLVED",12,["subscriber_id","account_id","subscriber_msisdn","balance_after","final_balance","final_status","resolution_type"]),
]

def _telecom_requirement_contract(scenario_id: str, type_of_data: str) -> tuple[list[dict], list[dict]]:
    """Return deterministic client-contract additions for the two audited Telecom journeys."""
    sid = str(scenario_id or "").strip().upper()
    if type_of_data != "transactional":
        return [], []
    if sid == "LB-06":
        defs, events = _LB06_VARIABLES, _LB06_EVENTS
    elif sid == "LB-07":
        defs, events = _LB07_VARIABLES, _LB07_EVENTS
    else:
        return [], []
    return [dict(v) for v in defs], [
        {"event_type": et, "sequence": seq, "fields": list(fields), "description": "Client-required scenario lifecycle event.", "min_occurrences": (1 if et not in {"TOPUP_ATTEMPT", "TOPUP_FAILURE", "TOPUP_RETRY"} else 1), "max_occurrences": (3 if et in {"TOPUP_ATTEMPT", "TOPUP_FAILURE", "TOPUP_RETRY"} else 1)}
        for et, seq, fields in events
    ]


def _telecom_mandatory_variables(profile: dict | None = None) -> list[dict]:
    """Telecom client override of the generic mandatory fields: fixed CLV tiers,
    fixed subscriber segments, a synthetic service_provider, and churn-helper fields
    instead of a raw churn score. loyalty_tier_customer is dropped — redundant with
    customer_lifetime_value. Only used when industry_key == "telecom"."""
    clv = dict(_MANDATORY_VARIABLES[0])
    clv.update({
        "dtype": "categorical",
        "description": "Customer lifetime value tier.",
        "gen": "weighted_choice",
        "params": {"choices": list(_TELECOM_CLV_CHOICES), "weights": list(_TELECOM_CLV_WEIGHTS)},
    })
    segment = {
        "name": "subscriber_segment",
        "dtype": "categorical",
        "description": "Subscriber usage/behavior segment.",
        "gen": "weighted_choice",
        "params": {"choices": list(_TELECOM_SEGMENT_CHOICES), "weights": list(_TELECOM_SEGMENT_WEIGHTS)},
        "depends_on": [],
        "nullable": False,
    }
    service_provider = {
        "name": "service_provider",
        "dtype": "categorical",
        "description": "Real-world telecom operator/service-provider brand appropriate to the target country.",
        "gen": "weighted_choice",
        "params": {
            "choices": list((profile or {}).get("service_providers") or ["Vodafone", "Orange", "Telefónica", "Deutsche Telekom"]),
            "weights": ([1.0] * len((profile or {}).get("service_providers"))) if (profile or {}).get("service_providers") else [0.40, 0.30, 0.20, 0.10],
        },
        "depends_on": [],
        "nullable": False,
    }
    return [clv, segment, service_provider] + [dict(v) for v in _TELECOM_CHURN_HELPER_VARIABLES]

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
    if industry_key == "telecom":
        core = [dict(v) for v in _TELECOM_CORE_MANDATORY if v["name"] not in {x["name"] for x in base}]
        for item in core:
            if item.get("name") == "subscriber_msisdn":
                item["params"] = {"country_codes": [profile["phone_country_code"]]}
        return base + _telecom_mandatory_variables(profile) + core
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
    if industry_key == "telecom":
        # Telecom-only rename: any trigger/cause-like field becomes "trigger_cause".
        # Gated here so no other industry's field naming is affected.
        for syn in _TELECOM_TRIGGER_CAUSE_SYNONYMS:
            alias_map.setdefault(syn, "trigger_cause")
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

# Keep this in sync with the dispatch table in agents/data_generation_agent.py.
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
5. customer_lifetime_value is auto-injected by the backend. Do not create or duplicate it in the variables list.
   loyalty_tier_customer is a generic-only legacy field and MUST NOT be created for Telecom.
"""

# If a variable represents one of these concepts, reuse the exact name — keeps field
# naming consistent across every scenario generated for the same industry/domain.
# NOTE: this is now built dynamically per-industry inside propose() via
# _build_glossary_text() / _build_alias_map() — see _CONCEPT_CANONICAL above.

_SYSTEM = """
You are the Scenario Designer Agent for a synthetic data generation platform.

REAL-WORLD DATA GROUNDING CONTRACT — NON-NEGOTIABLE:
- Every generated value must be semantically appropriate to the TARGET INDUSTRY, DOMAIN, USE CASE, and COUNTRY. Never use a value merely because it is syntactically valid.
- Never use placeholder/fake template values such as Provider_A, Provider_B, Company_A, Product_A, Service_A, Gateway_A, Test_Company, Synthetic_Provider, or similar when the field represents a real-world organization, operator, product, plan, regulator, location, payment method, or other industry entity.
- If a field represents a real-world company/operator/provider, use a real company/operator/provider appropriate to the target industry and country. Prefer the supplied industry profile vocabulary when available.
- If a field represents an industry-specific product, plan, service, instrument, channel, status, reason, role, or event, use terminology that actually exists in that industry; do not invent telecom-like values for non-telecom industries.
- Synthetic identifiers (IDs, UUIDs, account numbers, masked values) may be fabricated, but their FORMAT and semantic role must match the target industry.
- Do not add filler variables just to reach MAX_VARIABLES. Every variable must have a clear business purpose in the supplied scenario.
- Before returning the schema, perform a semantic relevance audit: for EACH variable, confirm that its description, generator choices, ranges, dependencies, and event usage are appropriate for the target industry/country. Remove or replace anything that fails this audit.
- For categorical fields, choices are an authoritative domain. Do not generate values outside those choices downstream.
- DATA RELEVANCE GATE: before accepting ANY variable, ask: “Would a domain expert in this exact industry, country, and scenario recognize this field and its values as normal business data?” If no, remove/replace it.
- REAL ENTITY GATE: for fields such as service_provider, operator, bank, insurer, hospital, merchant, retailer, carrier, manufacturer, regulator, or similar, use entities that actually belong to the TARGET INDUSTRY and TARGET COUNTRY. Never use a telecom operator as a generic company/provider for another industry.
- NO GENERIC FALLBACK: when the profile has no entity vocabulary for a field, do not copy vocabulary from another industry. Prefer scenario-supported domain terminology or omit the field rather than introducing an unrelated entity.
- VALUE-LEVEL AUDIT: audit every categorical choice individually, not only the field name. A plausible-looking value is invalid if it belongs to another industry, country, product category, or business process.
- DEPENDENCY AUDIT: after the value-level audit, verify that related fields describe the same business state. A failed/declined/cancelled state must not coexist with a success-only outcome unless the scenario explicitly represents a later recovery state.
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
  - "events": [ {"event_type": str, "sequence": int, "fields": [str], "description": str} ] — REQUIRED for transactional output; ordered business events; choose the number appropriate to the scenario; "description" is a short plain-English explanation of what happens at that event. For aggregational output return []
  - "label": str               — short human-readable scenario title
  - "journey": str              — the domain/journey name
  - "description": str          — 1-3 sentence description combining businessScenario, businessResponse, expectedOutcome
  - "variables": [ {name, dtype, description, gen, params, depends_on, nullable} ]
  - "field_order": [str]        — variable names in the exact order they should be generated/output
  - "edge_case_variables": [ {edge_case_name, edge_case_description, condition, name, dtype, description, gen, params, depends_on, nullable} ] — edge-case overrides/conditions; return [] when no useful edge cases can be defined

Edge cases: identify 2-4 high-value, realistic edge cases for the supplied scenario/use case. Return them as edge_case_variables. Each item MUST include edge_case_name, edge_case_description, name, and a machine-checkable condition expression using record field names (for example `balance_before == 0 and session_status == "TERMINATING"`). Reuse variables from the normal catalog when possible. Edge-case-only variables are allowed when they are necessary to make the edge case explicit, but they MUST include a complete executable dtype/gen/params definition. The condition must describe what makes the record an edge case and MUST be satisfiable within the declared variable constraints. Do not create impossible combinations.

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
        base_cache_key = ("proposal_base", industry_key, str(profile.get("country_code") or country or profile.get("country_name") or "default"))
        # Base fields are immutable schema templates; cache them because they are reused
        # on every proposal. A miss simply builds them once.
        base_variables = get_schema(base_cache_key)
        if base_variables is None:
            base_variables = _build_base_variables(industry_key, profile)
            set_schema(base_cache_key, base_variables)
        base_variables = [dict(v) for v in base_variables]
        base_names = {v["name"] for v in base_variables}
        preinjected_note = (
            f"These {len(base_variables)} fields are auto-injected and ALREADY generated — "
            f"do NOT include them in your \"variables\" list, just reference their exact names "
            f"in depends_on if a scenario-specific field needs one of them: {sorted(base_names)}\n"
            f"Return scenario-specific variables only. Prefer highly relevant variables over filler. "
            f"The backend will deterministically complete the final catalog to {MAX_VARIABLES} fields "
            f"from the reusable industry schema catalog when necessary.\n"
        )

        telecom_note = ""
        if industry_key == "telecom":
            telecom_note = (
                "Telecom-specific requirements for this scenario:\n"
                "- Do NOT invent any KYC/identity-verification field (no field name containing 'kyc').\n"
                "- Any plan/rate-plan/data-plan field (e.g. rate_plan_code) MUST depend_on service_provider "
                "and describe plans/products realistic for the actual operator brands supplied by the target-country profile, not placeholder names.\n"
                "- Do NOT invent a churn_risk_score/churn_propensity_score field; churn is derived downstream "
                "from the already auto-injected historical_low_balance_frequency, offer_acceptance_rate_historical "
                "and preferred_topup_channel fields.\n"
                "- customer_lifetime_value MUST be categorical with choices exactly low, medium, high.\n"
                f"- recharge_status (including aliases such as topup_status/recharge_result) MUST be named exactly 'recharge_status' "
                f"and MUST be categorical with choices exactly {_TELECOM_RECHARGE_STATUS_CHOICES}.\n"
                "- Do NOT create loyalty_tier_customer; it is not part of the Telecom schema.\n"
                "- If this scenario needs a trigger/cause field, name it exactly 'trigger_cause' with categorical "
                f"choices exactly {_TELECOM_TRIGGER_CAUSE_CHOICES}.\n"
                "- service_provider MUST use real operator/company brands from the target-country profile; NEVER use Provider_A, Provider_B, Company_A, or other placeholders. "
                "- Choose variables based on the requested journey and business outcome; do not generate a "
                "fixed scenario template. For any missing non-mandatory fields, the backend may select from "
                "its reusable Telecom schema catalog after this response.\n"
                "- Return schema definitions and, for transactional output, the required event definitions. "
                "Never return static customer/transaction records or pre-filled sample data.\n"
            )

        if industry_key == "telecom" and str(scenario_id or "").strip().upper() == "LB-06":
            telecom_note += "\nLB-06 client acceptance requirements: model Low Balance -> Top-up Attempt -> Failure -> Recovery Guidance -> Retry/Alternative -> Successful Recovery. Include an actual successful TOPUP_SUCCESS or equivalent event, balance_after movement, explicit parent-child linkage via parent_transaction_id, retry_count reconciliation, final_journey_status, multiple realistic failure reasons, abandonment, support escalation, payment-method switching, delayed recovery, and consecutive retry failures followed by eventual success. Do not claim recovery without a successful event and balance movement. Keep failure_reason semantically consistent with recharge_status. Use chronologically consistent timestamps.\n"
        elif industry_key == "telecom" and str(scenario_id or "").strip().upper() == "LB-07":
            telecom_note += "\nLB-07 client acceptance requirements: model Low Balance -> Top-Up Initiated -> Payment Authorized -> Payment Settled -> Balance Update Failed -> Discrepancy Detected -> Verification Requested -> Transaction Lookup -> Reconciliation Started -> Status Confirmed -> Customer Notified -> Balance Corrected -> Case Resolved. Explicitly represent Transaction Success=TRUE with Balance Updated=FALSE, expected/observed balance, variance, exception reason, verification, settlement, reconciliation, final status and final balance. Do not equate GATEWAY_TIMEOUT with successful payment unless subsequent settlement verification establishes success.\n"

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
            f"Country-appropriate payment methods: {profile.get('payment_methods', [])}\n"
            f"Industry-appropriate service providers/operators (when applicable): {profile.get('service_providers', [])}\n"
            f"Phone country code: {profile['phone_country_code']}; format: {profile['phone_format']}\n"
            f"Market character: {profile['market_character']}\n"
            f"Typical product/plan types in this industry+country: {profile['product_types']}\n"
            f"Identity/KYC notes: {profile['identity_notes']}\n"
            f"Typical transaction denominations in {profile['currency']}: {profile['typical_denominations']}\n\n"
            f"{glossary_text}\n"
            f"{telecom_note}"
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
        if industry_key == "telecom":
            llm_variables = self._apply_telecom_overrides(llm_variables, profile)
        combined = base_variables + llm_variables
        result["variables"] = self._enforce_variable_bounds(combined, prompt, base_names, industry_key, profile)
        if industry_key == "telecom":
            result["variables"] = self._apply_telecom_overrides(result["variables"], profile)
        # Deterministic client contract for the two audited Telecom scenarios.
        # The LLM remains responsible for additional scenario detail, but these
        # required fields/events cannot be omitted or renamed.
        if industry_key == "telecom":
            required_vars, required_events = _telecom_requirement_contract(scenario_id, type_of_data)
            existing_names = {str(v.get("name")) for v in result["variables"]}
            for required in required_vars:
                if required["name"] not in existing_names:
                    result["variables"].append(required)
                    existing_names.add(required["name"])
            if required_events:
                result["events"] = required_events
            # Required client fields are protected; trim only non-required fields if the
            # combined proposal exceeds the hard 35-variable ceiling.
            if len(result["variables"]) > MAX_VARIABLES:
                protected_required = {v["name"] for v in required_vars}
                kept = []
                for v in result["variables"]:
                    if len(kept) < MAX_VARIABLES or v["name"] in protected_required:
                        kept.append(v)
                # If protected fields pushed the list over the ceiling, retain all protected
                # fields and the earliest non-protected fields until the ceiling is reached.
                if len(kept) > MAX_VARIABLES:
                    protected_items = [v for v in kept if v["name"] in protected_required]
                    other_items = [v for v in kept if v["name"] not in protected_required]
                    result["variables"] = protected_items + other_items[:max(0, MAX_VARIABLES-len(protected_items))]
                else:
                    result["variables"] = kept
            # Required LB-06/LB-07 fields are added after the first bounds pass.
            # Fill any remaining Telecom slots from cached schema metadata so the final
            # proposal still targets the requested maximum of 35 fields.
            if len(result["variables"]) < MAX_VARIABLES:
                existing_names = {v["name"] for v in result["variables"]}
                catalog_key = ("telecom_proposal_catalog", str(profile.get("country_code") or "GLOBAL"))
                catalog = get_schema(catalog_key)
                if catalog is None:
                    catalog = [dict(v) for v in _TELECOM_CACHED_PROPOSAL_CATALOG]
                    set_schema(catalog_key, catalog)
                providers = list(profile.get("service_providers") or ["Vodafone", "Orange", "Telefónica", "Deutsche Telekom"])
                payment_methods = list(profile.get("payment_methods") or ["credit_card", "debit_card", "digital_wallet", "bank_transfer"])
                for candidate in catalog:
                    if len(result["variables"]) >= MAX_VARIABLES:
                        break
                    candidate = dict(candidate)
                    if candidate.get("name") == "service_provider":
                        candidate["params"] = {"choices": providers, "weights": [1.0] * len(providers)}
                    elif candidate.get("name") == "payment_method":
                        candidate["params"] = {"choices": payment_methods, "weights": [1.0] * len(payment_methods)}
                    if candidate["name"] == "payment_gateway":
                        continue
                    if candidate["name"] not in existing_names:
                        result["variables"].append(dict(candidate))
                        existing_names.add(candidate["name"])
        result["field_order"] = [v["name"] for v in result["variables"]]
        # Resolve entity_key only AFTER base variables are injected. The primary entity
        # identifier is normally one of those auto-injected fields (e.g. subscriber_id),
        # so validating against Gemini's scenario-specific variables alone is incorrect.
        result["entity_key"] = self._normalize_entity_key(
            result.get("entity_key"), result["variables"], industry_key, type_of_data
        )
        result["events"] = self._normalize_events(result.get("events", []), result["variables"], type_of_data, industry_key)
        edge_vars = self._normalize_edge_case_variables(
            result.get("edge_case_variables", []), result["variables"], industry_key, profile=profile
        )

        # If the LLM omitted usable edge cases (or returned invalid conditions), build
        # deterministic boundary cases from the actual scenario variables. This is a
        # safety net, not an industry-specific hard-code: it works for numeric, boolean
        # and categorical variables in Telecom, Banking, Retail, Healthcare, etc.
        if not edge_vars:
            edge_vars = self._fallback_edge_cases(result["variables"])

        # Promote any valid edge-only fields into the normal schema so generation and QA
        # know their dtype/generator and can place them in event records.
        known = {v.get("name") for v in result["variables"]}
        for ev in edge_vars:
            name = ev.get("name")
            if name and name not in known and ev.get("gen") in {
                "constant", "uniform", "lognormal", "lognormal_int", "beta",
                "weighted_choice", "segment_range", "recent_datetime", "prefixed_int",
                "prefixed_uuid", "e164_phone", "id_mirror", "formula"
            }:
                result["variables"].append({
                    k: ev[k] for k in ("name", "dtype", "description", "gen", "params", "depends_on", "nullable", "formula")
                    if k in ev
                })
                known.add(name)
        result["field_order"] = [v["name"] for v in result["variables"]]
        result["edge_case_variables"] = edge_vars
        return result


    def _normalize_edge_case_variables(
        self,
        edge_vars: list,
        variables: list[dict],
        industry_key: str,
        profile: dict | None = None,
    ) -> list[dict]:
        """Normalize and preflight edge cases before they reach /scenario/confirm.

        The LLM can produce syntactically valid but operationally impossible edge
        conditions (for example a percentage threshold plus a zero balance when the
        effective generator cannot materialize both values).  Such definitions are
        removed here, while valid definitions are retained.  The check deliberately
        reuses the same generic deterministic edge solver as data generation, so
        proposal-time validation and generation-time behavior stay aligned.
        """
        alias_map = _build_alias_map(industry_key)
        valid_names = {str(v.get("name")) for v in variables if isinstance(v, dict) and v.get("name")}
        out = []
        for item in edge_vars if isinstance(edge_vars, list) else []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            v = dict(item)
            raw_name = str(v.get("name")).strip()
            v["name"] = alias_map.get(raw_name.lower(), raw_name)
            v["edge_case_name"] = str(v.get("edge_case_name") or "Scenario Edge Case").strip()
            v["edge_case_description"] = str(v.get("edge_case_description") or "").strip()
            v["condition"] = str(v.get("condition") or "").strip()
            if v.get("event_type"):
                v["event_type"] = str(v.get("event_type")).strip().upper().replace(" ", "_")
            if not v["condition"]:
                continue

            # Validate expression syntax first. Names may refer to another edge-only
            # item; collect all edge names from the LLM response before rejecting refs.
            edge_names = {
                str(x.get("name")).strip() for x in edge_vars
                if isinstance(x, dict) and x.get("name")
            }
            if not self._valid_edge_condition(v["condition"], valid_names | edge_names):
                continue

            base = next((x for x in variables if x.get("name") == v["name"]), None)
            if base:
                merged = dict(base)
                merged.update({k: val for k, val in v.items() if val not in (None, "")})
                v = merged
            else:
                # Edge-only fields are allowed when the item carries a complete,
                # executable generator definition.
                if v.get("dtype") not in {"string", "float", "int", "categorical", "datetime", "boolean", "bool"}:
                    continue
                if v.get("gen") not in {
                    "constant", "uniform", "lognormal", "lognormal_int", "beta", "weighted_choice",
                    "segment_range", "recent_datetime", "prefixed_int", "prefixed_uuid", "e164_phone",
                    "id_mirror", "formula"
                }:
                    continue
            out.append(v)

        # Materialize deterministic overrides for simple condition assignments.
        # This is the key guard against LLM-generated conditions such as
        # ``data_depletion_pct >= 100 and balance_before == 0`` when the ordinary
        # generators cannot naturally emit those values.  The edge condition itself
        # remains the source of truth; these constant overrides merely make the
        # requested state executable and are applied only to edge records.
        if out:
            try:
                from agents.data_generation_agent import _condition_branches

                by_name = {str(v.get("name")): v for v in variables if v.get("name")}
                formula_names = {str(v.get("name")) for v in variables if v.get("formula")}
                groups: dict[str, dict] = {}
                for item in out:
                    group_name = str(item.get("edge_case_name") or "Scenario Edge Case").strip()
                    group = groups.setdefault(group_name, {
                        "condition": str(item.get("condition") or "").strip(),
                        "variables": [],
                    })
                    group["variables"].append(item)
                    if item.get("condition") and not group.get("condition"):
                        group["condition"] = str(item["condition"]).strip()

                for group in groups.values():
                    branches = _condition_branches(group.get("condition", ""), variables) or []
                    if not branches:
                        continue
                    # The first satisfiable branch is sufficient for OR expressions.
                    assignments = branches[0]
                    existing = {str(v.get("name")): v for v in group["variables"] if v.get("name")}
                    for name, value in assignments.items():
                        if str(name).startswith("__") or name in formula_names or name not in by_name:
                            continue
                        base = by_name[name]
                        target = existing.get(name)
                        if target is None:
                            target = dict(base)
                            target["edge_case_name"] = next(
                                k for k, g in groups.items() if g is group
                            )
                            target["edge_case_description"] = ""
                            target["condition"] = group.get("condition", "")
                            group["variables"].append(target)
                            existing[name] = target
                        target["gen"] = "constant"
                        target["params"] = {"value": value}
                        target["dtype"] = base.get("dtype", target.get("dtype", "string"))

                out = [item for group in groups.values() for item in group["variables"]]
            except Exception as exc:
                logger.warning("[ScenarioDesigner] Edge-condition override normalization skipped: %s", exc)

        # Preflight complete edge groups using the same deterministic solver used by
        # generation.  This is intentionally best-effort: if a group cannot be
        # constructed from the declared schema, omit that group instead of allowing
        # an unusable edge definition to reach /scenario/confirm and later break
        # /scenario/generate.
        if out:
            try:
                from agents.data_generation_agent import (
                    _condition_branches,
                    _edge_case_candidate_for_aggregation,
                    _generate_record,
                    _safe_edge_condition,
                )

                groups: dict[str, dict] = {}
                for item in out:
                    group_name = str(item.get("edge_case_name") or "Scenario Edge Case").strip()
                    group = groups.setdefault(group_name, {
                        "condition": str(item.get("condition") or "").strip(),
                        "variables": [],
                    })
                    group["variables"].append(item)
                    if item.get("condition") and not group.get("condition"):
                        group["condition"] = str(item["condition"]).strip()

                valid_groups: set[str] = set()
                for group_name, group in groups.items():
                    condition = group.get("condition", "")
                    branches = _condition_branches(condition, variables) or []
                    if not branches:
                        continue
                    constructible = False
                    for _ in range(5):
                        try:
                            base = _generate_record(variables, profile=profile, rules=None)
                        except Exception:
                            base = {}
                        for assignments in branches:
                            try:
                                candidate = _edge_case_candidate_for_aggregation(
                                    dict(base), group, variables, profile, None, assignments=assignments
                                )
                                if _safe_edge_condition(condition, candidate):
                                    constructible = True
                                    break
                            except Exception:
                                continue
                        if constructible:
                            break
                    if constructible:
                        valid_groups.add(group_name)

                out = [
                    item for item in out
                    if str(item.get("edge_case_name") or "Scenario Edge Case").strip() in valid_groups
                ]
            except Exception as exc:
                # Never make proposal fail solely because the optional preflight
                # helper is unavailable. Syntax/schema validation above remains the
                # hard gate.
                logger.warning("[ScenarioDesigner] Edge-case constructability preflight skipped: %s", exc)

        return out[:50]

    @staticmethod
    def _valid_edge_condition(expression: str, valid_names: set[str]) -> bool:
        try:
            tree = ast.parse(expression, mode="eval")
            allowed = (
                ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not,
                ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Name, ast.Constant,
                ast.List, ast.Tuple, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load,
            )
            if any(not isinstance(n, allowed) for n in ast.walk(tree)):
                return False
            refs = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            return refs.issubset(valid_names)
        except Exception:
            return False

    @staticmethod
    def _fallback_edge_cases(variables: list[dict]) -> list[dict]:
        """Create valid, scenario-derived boundary edge cases when the LLM fails."""
        cases: list[dict] = []
        for var in variables:
            if len(cases) >= 4:
                break
            name = str(var.get("name", ""))
            if not name:
                continue
            dtype = str(var.get("dtype", "string"))
            params = var.get("params") if isinstance(var.get("params"), dict) else {}
            override = None
            condition = ""
            if dtype in {"float", "int"}:
                lo, hi = params.get("min"), params.get("max")
                if hi is not None and isinstance(hi, (int, float)) and math.isfinite(float(hi)):
                    override = hi
                    condition = f"{name} == {repr(hi)}"
                elif lo is not None and isinstance(lo, (int, float)) and math.isfinite(float(lo)):
                    override = lo
                    condition = f"{name} == {repr(lo)}"
            elif dtype in {"boolean", "bool"}:
                override = True
                condition = f"{name} == True"
            elif dtype in {"categorical", "string"}:
                choices = params.get("choices")
                if isinstance(choices, list) and choices:
                    override = choices[0]
                    condition = f"{name} == {repr(choices[0])}"
            if override is None or not condition:
                continue
            edge = dict(var)
            edge.update({
                "edge_case_name": f"Boundary {name}",
                "edge_case_description": f"{name} is at a defined scenario boundary/value.",
                "condition": condition,
                "gen": "constant",
                "params": {"value": override},
            })
            cases.append(edge)
        return cases

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
                "description": str(event.get("description") or "").strip(),
                "min_occurrences": min_occurrences,
                "max_occurrences": max_occurrences,
            })
        # A transactional proposal must always have at least one executable event.
        # Gemini can occasionally omit events or return event fields that do not map to
        # the confirmed variable catalog. Previously that left a weak BUSINESS_EVENT
        # definition (or an empty event list after normalization), which could result
        # in no useful transactional output for ordinary scenarios. Build a generic
        # executable event from the actual scenario variables instead of requiring an
        # LB-06/LB-07 special case.
        if not normalized:
            preferred = [
                v["name"] for v in variables
                if v.get("name") not in {"event_timestamp"}
            ]
            normalized = [{
                "event_type": "BUSINESS_EVENT",
                "sequence": 1,
                "fields": preferred,
                "description": "Primary business event for the requested transactional scenario.",
                "min_occurrences": 1,
                "max_occurrences": 10,
            }]

        # Keep the full client-defined lifecycle when supplied (LB-07 currently has
        # 12 events). The previous 8-event cap silently removed valid events. A bounded
        # ceiling prevents malformed LLM output from creating unbounded event lists.
        return normalized[:32]

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

    @staticmethod
    def _apply_telecom_overrides(variables: list[dict], profile: dict | None = None) -> list[dict]:
        """Telecom-only client requirements: drop KYC fields and raw churn scores,
        mask UPI in payment/channel choices, and force trigger_cause's fixed enum
        when that field is present. Only ever called when industry_key == "telecom";
        every other industry's variables pass through this file unmodified."""
        out: list[dict] = []
        for v in variables:
            name = str(v.get("name", ""))
            lname = name.lower()
            if _KYC_NAME_MARKER in lname:
                continue
            if _TELECOM_CHURN_NAME_MARKER in lname:
                continue
            if lname in _TELECOM_DROPPED_MANDATORY_NAMES:
                continue

            # Telecom client contract: canonicalize all recharge-status synonyms and
            # force the exact four allowed values regardless of what Gemini proposed.
            if lname in {x.lower() for x in _TELECOM_RECHARGE_STATUS_SYNONYMS}:
                v = dict(v)
                v["name"] = "recharge_status"
                v["dtype"] = "categorical"
                v["gen"] = "weighted_choice"
                v["params"] = {
                    "choices": list(_TELECOM_RECHARGE_STATUS_CHOICES),
                    "weights": list(_TELECOM_RECHARGE_STATUS_WEIGHTS),
                }
                name = "recharge_status"
                lname = name
            else:
                v = dict(v)

            # Telecom provider is a real-world operator, not a synthetic placeholder.
            # The country profile is the authoritative vocabulary for this field.
            if name == "service_provider":
                providers = list((profile or {}).get("service_providers") or ["Vodafone", "Orange", "Telefónica", "Deutsche Telekom"])
                weights = [1.0] * len(providers)
                v["dtype"] = "categorical"
                v["gen"] = "weighted_choice"
                v["description"] = "Real-world telecom operator/service-provider brand appropriate to the target country."
                v["params"] = {"choices": providers, "weights": weights}
            elif name == "payment_method":
                # Telecom client vocabulary is explicit and must be identical across
                # Telecom proposals. UPI is intentionally represented as DIGITAL_WALLET.
                v["dtype"] = "categorical"
                v["gen"] = "weighted_choice"
                v["params"] = {
                    "choices": ["DEBIT_CARD", "CREDIT_CARD", "DIGITAL_WALLET", "NETBANKING"],
                    "weights": [0.35, 0.30, 0.20, 0.15],
                }

            if name == "customer_lifetime_value":
                v["dtype"] = "categorical"
                v["gen"] = "weighted_choice"
                v["params"] = {
                    "choices": list(_TELECOM_CLV_CHOICES),
                    "weights": list(_TELECOM_CLV_WEIGHTS),
                }
            elif name == "trigger_cause":
                v["dtype"] = "categorical"
                v["gen"] = "weighted_choice"
                v["params"] = {
                    "choices": list(_TELECOM_TRIGGER_CAUSE_CHOICES),
                    "weights": list(_TELECOM_TRIGGER_CAUSE_WEIGHTS),
                }
            elif any(marker in lname for marker in _UPI_FIELD_NAME_MARKERS):
                params = v.get("params")
                if isinstance(params, dict) and isinstance(params.get("choices"), list):
                    params = dict(params)
                    params["choices"] = [
                        _UPI_CANONICAL_OUTPUT if _UPI_VALUE_MARKER in str(c).lower() else c
                        for c in params["choices"]
                    ]
                    v["params"] = params
            out.append(v)

        # Deduplicate fields after canonicalization (e.g. recharge_status + topup_status)
        # and repair dependencies after telecom-only fields have been removed.
        deduped: list[dict] = []
        seen_names: set[str] = set()
        for v in out:
            field_name = str(v.get("name", ""))
            if not field_name or field_name in seen_names:
                continue
            seen_names.add(field_name)
            deduped.append(v)
        valid_names = {v["name"] for v in deduped}
        for v in deduped:
            v["depends_on"] = [d for d in v.get("depends_on", []) if d in valid_names]
        return deduped

    def _enforce_variable_bounds(
        self, variables: list[dict], prompt: str, protected_names: set[str], industry_key: str = "generic", profile: dict | None = None
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

        if len(variables) < _EXPAND_TARGET and industry_key == "telecom":
            # Telecom deliberately does not make a second Gemini call just to reach 35.
            # Gemini chooses the scenario-specific fields; Python fills only the remaining
            # slots from reusable schema metadata. No customer/transaction records are cached.
            existing = {v["name"] for v in variables}
            catalog_key = ("telecom_proposal_catalog", str(profile.get("country_code") or "GLOBAL"))
            catalog = get_schema(catalog_key)
            if catalog is None:
                catalog = [dict(v) for v in _TELECOM_CACHED_PROPOSAL_CATALOG]
                set_schema(catalog_key, catalog)
            # Always re-ground reusable catalog values to the current country profile.
            providers = list(profile.get("service_providers") or ["Vodafone", "Orange", "Telefónica", "Deutsche Telekom"]) if profile else ["Vodafone", "Orange", "Telefónica", "Deutsche Telekom"]
            payment_methods = list(profile.get("payment_methods") or ["credit_card", "debit_card", "digital_wallet", "bank_transfer"]) if profile else ["credit_card", "debit_card", "digital_wallet", "bank_transfer"]
            for candidate in catalog:
                if candidate.get("name") == "service_provider":
                    candidate["params"] = {"choices": providers, "weights": [1.0] * len(providers)}
                elif candidate.get("name") == "payment_method":
                    candidate["params"] = {"choices": payment_methods, "weights": [1.0] * len(payment_methods)}
                elif candidate.get("name") == "payment_gateway":
                    continue
            for candidate in catalog:
                if len(variables) >= MAX_VARIABLES:
                    break
                if candidate["name"] == "payment_gateway":
                    continue
                if candidate["name"] not in existing:
                    variables.append(dict(candidate))
                    existing.add(candidate["name"])
            return variables

        if len(variables) < _EXPAND_TARGET:
            logger.info("[ScenarioDesigner] Only %d variables produced; requesting one bounded expansion (target %d)",
                        len(variables), _EXPAND_TARGET)
            scenario_specific = [v for v in variables if v["name"] not in protected_names]
            expand_prompt = (
                prompt + "\n\n"
                f"Your previous attempt produced {len(scenario_specific)} scenario-specific variables. "
                f"Add NEW, distinct, relevant variables so the scenario-specific total approaches "
                f"{MAX_VARIABLES - len(protected_names)}. Do not repeat protected fields. "
                "Return the full scenario-specific variables list under the variables key, in dependency order."
            )
            try:
                retry = self._llm.generate_json(_SYSTEM, expand_prompt, temperature=0.3)
                retried_vars = [v for v in retry.get("variables", []) if isinstance(v, dict) and v.get("name")]
                retried_vars = [v for v in retried_vars if v["name"] not in protected_names]
                retried_vars = self._normalize_field_names(retried_vars, industry_key)
                if len(retried_vars) > len(scenario_specific):
                    protected = [v for v in variables if v["name"] in protected_names]
                    variables = (protected + retried_vars)[:MAX_VARIABLES]
            except Exception as exc:
                logger.warning("[ScenarioDesigner] Expansion retry failed: %s", exc)

        return variables

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