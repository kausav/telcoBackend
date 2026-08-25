"""
Full variable catalog built from "Telco SDG 18_8_26.xlsx" — Sheet1.
Every variable carries its generation_type, params, and inter-field dependencies.
The generator resolves dependencies in the order variables appear here.
"""
from __future__ import annotations

VARIABLES: list[dict] = [
    # ── Identity & Account ────────────────────────────────────────────────
    {
        "name": "subscriber_id",
        "dtype": "string",
        "gen": "prefixed_int",
        "params": {"prefix": "SUB-", "digits": 7},
        "unique": True,
    },
    {
        "name": "subscriber_msisdn",
        "dtype": "string",
        "gen": "e164_phone",
        "params": {"country_codes": ["+1"]},
    },
    {
        "name": "account_id",
        "dtype": "string",
        # numeric part mirrors subscriber_id so ACC-PRE-XXXXXXX == SUB-XXXXXXX
        "gen": "id_mirror",
        "params": {"prefix": "ACC-PRE-", "source_field": "subscriber_id", "source_prefix": "SUB-"},
        "depends_on": ["subscriber_id"],
    },

    # ── Segmentation & Plan ───────────────────────────────────────────────
    {
        "name": "subscriber_segment",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices":  ["Habitual Rechargers", "Occasional Rechargers", "Valued Customers"],
            "weights":  [0.55, 0.30, 0.15],
        },
    },
    {
        "name": "rate_plan_code",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices":  ["PRE_FLEX_TALK_30D", "PRE_UNLIMITED_28D", "PRE_LONG_VAL_84D",
                         "PRE_DATA_BOOSTER_DAILY", "PRE_ANNUAL_365D"],
            "weights":  [0.35, 0.30, 0.15, 0.10, 0.10],
        },
    },
    {
        "name": "wallet_currency",
        "dtype": "string",
        "gen": "constant",
        "params": {"value": "USD"},
    },

    # ── Financial Profile ─────────────────────────────────────────────────
    {
        "name": "customer_lifetime_value",
        "dtype": "float",
        # right-skewed log-normal, clipped to [25, 2500]
        "gen": "lognormal",
        "params": {"mu": 5.5, "sigma": 0.8, "min": 25.00, "max": 2500.00},
    },
    {
        "name": "churn_propensity_score",
        "dtype": "float",
        # Beta(α=2, β=5) for normal LB baseline — skewed toward low churn
        "gen": "beta",
        "params": {"alpha": 2, "beta": 5},
    },
    {
        "name": "loyalty_tier_customer",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices":  ["Bronze", "Silver", "Gold", "Platinum"],
            "weights":  [0.40, 0.30, 0.20, 0.10],
        },
    },
    {
        "name": "avg_daily_spend_trailing_30d",
        "dtype": "float",
        "gen": "uniform",
        "params": {"min": 0.50, "max": 50.00},
    },

    # ── Threshold Calculation (segment-driven) ────────────────────────────
    {
        "name": "dynamic_lb_threshold_pct",
        "dtype": "float",
        # percentage: Habitual 100–150%, Occasional 150–200%, Valued 200–300%
        "gen": "segment_range",
        "params": {
            "Habitual Rechargers":   {"min": 100.00, "max": 150.00},
            "Occasional Rechargers": {"min": 150.00, "max": 200.00},
            "Valued Customers":      {"min": 200.00, "max": 300.00},
        },
        "depends_on": ["subscriber_segment"],
    },
    {
        "name": "low_balance_threshold_amt",
        "dtype": "float",
        # avg_daily_spend × (threshold_pct / 100), clipped to [0.50, 15.00]
        "gen": "formula",
        "formula": "round(min(15.0, max(0.5, avg_daily_spend_trailing_30d * (dynamic_lb_threshold_pct / 100))), 2)",
        "depends_on": ["avg_daily_spend_trailing_30d", "dynamic_lb_threshold_pct"],
    },
    {
        "name": "balance_before",
        "dtype": "float",
        # strictly: 0.00 ≤ balance_before ≤ low_balance_threshold_amt
        "gen": "uniform_bounded",
        "params": {"lo": 0.00, "hi_field": "low_balance_threshold_amt"},
        "depends_on": ["low_balance_threshold_amt"],
    },

    # ── Recharge History ──────────────────────────────────────────────────
    {
        "name": "days_since_last_recharge",
        "dtype": "int",
        # min 7, modal peaks at 28/30/84 matching plan validity cycles
        "gen": "weighted_choice",
        "params": {
            "choices": [7, 10, 14, 21, 28, 30, 60, 84, 90],
            "weights": [0.08, 0.05, 0.07, 0.07, 0.28, 0.23, 0.07, 0.12, 0.03],
        },
    },

    # ── Trigger Event ─────────────────────────────────────────────────────
    {
        "name": "trigger_event_id",
        "dtype": "string",
        "gen": "prefixed_uuid",
        "params": {"prefix": "trig-"},
    },
    {
        "name": "event_timestamp",
        "dtype": "datetime",
        # T0 — the anchor timestamp, everything else is relative to this
        "gen": "recent_datetime",
        "params": {"days_back": 90},
    },
    {
        "name": "trigger_type",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["LOW_BALANCE_THRESHOLD_BREACH", "VALIDITY_EXPIRY_WARNING", "ZERO_BALANCE"],
            "weights": [0.70, 0.20, 0.10],
        },
    },

    # ── NBA Offer ─────────────────────────────────────────────────────────
    {
        "name": "nba_offer_id",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["TOPUP_BUNDLE_PROMO_20", "TOPUP_BUNDLE_PROMO_10", "TOPUP_VALUE_PACK_50",
                        "TOPUP_SACHET_BOOSTER_05", "TOPUP_RECHARGE_ANNUAL_200"],
            "weights": [0.40, 0.30, 0.15, 0.10, 0.05],
        },
    },
    {
        "name": "recommended_topup_amt",
        "dtype": "float",
        "gen": "weighted_choice",
        "params": {
            "choices": [5.00, 10.00, 20.00, 50.00],
            "weights": [0.15, 0.30, 0.40, 0.15],
        },
    },
    {
        "name": "incentive_bonus_credit",
        "dtype": "float",
        "gen": "uniform",
        "params": {"min": 0.00, "max": 5.00},
    },

    # ── Outreach ──────────────────────────────────────────────────────────
    {
        "name": "outreach_channel",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["APP_PUSH", "SMS", "RCS", "WHATSAPP"],
            "weights": [0.40, 0.30, 0.15, 0.15],
        },
    },
    {
        "name": "notification_dispatch_ts",
        "dtype": "datetime",
        # T1 = T0 + 1s–10s
        "gen": "ts_offset",
        "params": {"base_field": "event_timestamp", "min_sec": 1, "max_sec": 10},
        "depends_on": ["event_timestamp"],
    },
    {
        "name": "channel_delivery_status",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["DELIVERED", "READ", "SENT", "FAILED"],
            "weights": [0.60, 0.36, 0.03, 0.01],
        },
    },

    # ── Customer Response ─────────────────────────────────────────────────
    {
        "name": "customer_action",
        "dtype": "categorical",
        # LB-01 Normal: customer always accepts
        "gen": "constant",
        "params": {"value": "ACCEPTED"},
    },
    {
        "name": "response_time_seconds",
        "dtype": "int",
        # log-normal(μ=5.5, σ=1.2), peak 2–10 mins, clipped [10, 86400]
        "gen": "lognormal_int",
        "params": {"mu": 5.5, "sigma": 1.2, "min": 10, "max": 86400},
    },
    {
        "name": "customer_response_ts",
        "dtype": "datetime",
        # T2 = T1 + response_time_seconds
        "gen": "ts_add_field",
        "params": {"base_field": "notification_dispatch_ts", "add_seconds_field": "response_time_seconds"},
        "depends_on": ["notification_dispatch_ts", "response_time_seconds"],
    },

    # ── Transaction ───────────────────────────────────────────────────────
    {
        "name": "recharge_channel",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["MY_ACCOUNT_APP", "RCS_LINK", "USSD", "WEB_PORTAL"],
            "weights": [0.50, 0.30, 0.10, 0.10],
        },
    },
    {
        "name": "payment_method",
        "dtype": "categorical",
        "gen": "weighted_choice",
        "params": {
            "choices": ["DIRECT_PAY", "CREDIT_CARD", "DEBIT_CARD", "DIGITAL_WALLET"],
            "weights": [0.40, 0.25, 0.20, 0.15],
        },
    },
    {
        "name": "payment_gateway_tx_id",
        "dtype": "string",
        "gen": "tx_id",
        "params": {"prefix": "PG-TXN-"},
    },
    {
        "name": "recharge_amount",
        "dtype": "float",
        # equals recommended_topup_amt for LB-01 Normal (successful acceptance)
        "gen": "formula",
        "formula": "recommended_topup_amt",
        "depends_on": ["recommended_topup_amt"],
    },
    {
        "name": "transaction_status",
        "dtype": "categorical",
        # LB-01 Normal: always SUCCESS
        "gen": "constant",
        "params": {"value": "SUCCESS"},
    },

    # ── Post-Transaction Balance ───────────────────────────────────────────
    {
        "name": "balance_after",
        "dtype": "float",
        # balance_before + recharge_amount, rounded to 2dp
        "gen": "formula",
        "formula": "round(balance_before + recharge_amount, 2)",
        "depends_on": ["balance_before", "recharge_amount"],
    },
    {
        "name": "balance_expiry_date_new",
        "dtype": "date",
        # event_timestamp + 30 calendar days
        "gen": "date_offset",
        "params": {"base_field": "event_timestamp", "days": 30},
        "depends_on": ["event_timestamp"],
    },

    # ── Terminal State ────────────────────────────────────────────────────
    {
        "name": "journey_state_final",
        "dtype": "categorical",
        "gen": "constant",
        "params": {"value": "COMPLETED_SUCCESS"},
    },
    {
        "name": "ocs_ledger_sync_status",
        "dtype": "categorical",
        "gen": "constant",
        "params": {"value": "SYNCHRONIZED"},
    },
]

# Ordered field names for output columns
FIELD_ORDER = [v["name"] for v in VARIABLES]