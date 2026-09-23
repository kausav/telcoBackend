"""Generic scenario semantics shared by proposal-generation guardrails and QA.

No scenario IDs are referenced here. Behavior is derived from the request context
and the semantic meaning of generated fields.
"""
from __future__ import annotations

import re


def classify_outcome_mode(
    scenario_type: str = "",
    expected_outcome: str = "",
    business_response: str = "",
    business_scenario: str = "",
) -> str:
    """Classify the requested outcome without using scenario IDs or templates."""
    def norm(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower()).replace("_", " ").replace("-", " ")

    mode = norm(scenario_type)
    exact = {
        "suppression": "suppression",
        "suppressed": "suppression",
        "suppress": "suppression",
        "hold": "suppression",
        "customer declines": "decline_or_no_response",
        "decline": "decline_or_no_response",
        "declines": "decline_or_no_response",
        "customer declines / no response": "decline_or_no_response",
        "no response": "decline_or_no_response",
        "failure": "negative",
        "failed": "negative",
        "error": "negative",
        "exception": "negative",
        "concurrent": "concurrent",
        "concurrency": "concurrent",
        "priority": "concurrent",
        "cross journey": "concurrent",
        "no clear priority": "concurrent",
        "normal": "positive",
        "standard": "positive",
        "happy path": "positive",
        "success": "positive",
        "successful": "positive",
        "positive": "positive",
        "nominal": "positive",
    }
    if mode in exact:
        # Explicit outcome content can override ordinary "normal" wording, but not
        # explicit adverse/suppression/decline/concurrency scenario modes.
        if exact[mode] in {"suppression", "decline_or_no_response", "negative", "concurrent"}:
            return exact[mode]
        mode = exact[mode]

    positive_terms = ("successful", "success", "completed", "approved", "accepted", "delivered", "fulfilled", "retained")
    negative_terms = ("failure", "failed", "unsuccessful", "rejected", "declined", "denied", "blocked", "error", "cancelled", "abandoned")
    suppression_terms = ("suppression", "suppressed", "do not contact", "opt out", "hold")
    decline_terms = ("customer decline", "declined", "no response", "no-response", "customer reject", "rejected")
    concurrent_terms = ("concurrent", "priority", "competing", "no clear priority", "cross-journey", "cross journey")

    # Compound/verbose scenario types (for example "Recharge Failure" or
    # "Customer Declines - No Response") must still resolve to the intended mode.
    # Exact scenario IDs are never required; semantic keywords in scenarioType are enough.
    if any(term in mode for term in suppression_terms):
        return "suppression"
    if any(term in mode for term in decline_terms):
        return "decline_or_no_response"
    if any(term in mode for term in concurrent_terms):
        return "concurrent"
    if any(term in mode for term in negative_terms):
        return "negative"
    if any(term in mode for term in positive_terms):
        return "positive"

    outcome_text = " ".join(norm(x) for x in (expected_outcome, business_response)).strip()
    scenario_text = " ".join(norm(x) for x in (business_scenario,)).strip()
    if any(term in outcome_text for term in suppression_terms):
        return "suppression"
    if any(term in outcome_text for term in decline_terms):
        return "decline_or_no_response"
    if any(term in outcome_text for term in negative_terms):
        return "negative"
    if any(term in outcome_text for term in positive_terms):
        return "positive"
    if any(term in scenario_text for term in concurrent_terms) or any(term in outcome_text for term in concurrent_terms):
        return "concurrent"
    if any(term in scenario_text for term in suppression_terms):
        return "suppression"
    if any(term in scenario_text for term in decline_terms):
        return "decline_or_no_response"
    if any(term in scenario_text for term in negative_terms):
        return "negative"
    if any(term in scenario_text for term in positive_terms):
        return "positive"
    if mode in {"positive", "negative", "suppression", "decline_or_no_response", "concurrent"}:
        return mode
    return "mixed"


def derive_scenario_semantics(state, variables: list[dict]) -> dict:
    """Derive generic, deterministic semantic guardrails from the full scenario context.

    This is intentionally not tied to any one scenario type or field name.  The
    hierarchy is:
      1) explicit confirmed-contract value/formula constraints remain authoritative;
      2) explicit outcome language in expected_outcome/business_response/business_scenario
         overrides generic scenario-type defaults;
      3) scenario_type supplies a fallback intent (positive/negative/mixed);
      4) field meaning comes from its name + description.

    The LLM still supplies richer cross-field rules.  These deterministic directives
    are a safety net so semantic contradictions cannot be introduced merely because
    the field generator is stochastic or the LLM omitted a machine-readable rule.
    """
    def clean(value: object) -> str:
        return str(value or "").strip()

    def norm(value: object) -> str:
        return re.sub(r"\s+", " ", clean(value).lower())

    scenario_type = clean(getattr(state, "scenario_type", ""))
    context = getattr(state, "scenario_context", {}) or {}
    pieces = {
        "expected_outcome": clean(getattr(state, "expected_outcome", "") or context.get("expected_outcome")),
        "business_response": clean(getattr(state, "business_response", "") or context.get("business_response")),
        "business_scenario": clean(getattr(state, "business_scenario", "") or context.get("business_scenario")),
        "use_case": clean(getattr(state, "use_case", "") or context.get("use_case")),
        "domain": clean(getattr(state, "domain", "") or context.get("domain")),
        "scenario_type": scenario_type,
        "description": clean(context.get("description")),
        "journey": clean(context.get("journey")),
        "label": clean(context.get("label")),
    }

    positive_terms = {
        "success", "successful", "succeed", "succeeded", "completed", "completion",
        "approved", "approval", "accepted", "acceptance", "passed", "pass", "delivered",
        "fulfilled", "settled", "authorized", "enabled", "active", "eligible", "retained",
    }
    negative_terms = {
        "failure", "failed", "unsuccessful", "rejected", "rejection", "declined", "decline",
        "error", "errored", "denied", "denial", "blocked", "expired", "cancelled", "canceled",
        "abandoned", "churn", "churned", "ineligible", "unpaid", "unsatisfied",
    }
    mixed_terms = {
        "mixed", "distribution", "rate", "ratio", "probability", "probabilities", "both",
        "varied", "variation", "realistic", "real-world", "realistic mix", "success rate",
        "failure rate", "exception rate", "conversion rate",
    }

    def term_score(text: str, terms: set[str]) -> int:
        lowered = norm(text)
        return sum(len(re.findall(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered)) for term in terms)

    # Outcome statements have higher authority than descriptive/context fields.
    weighted = {
        "expected_outcome": 6,
        "business_response": 5,
        "business_scenario": 4,
        "use_case": 3,
        "domain": 2,
        "scenario_type": 2,
        "description": 1,
        "journey": 1,
        "label": 1,
    }
    pos = neg = mixed = 0
    for key, text in pieces.items():
        w = weighted[key]
        pos += w * term_score(text, positive_terms)
        neg += w * term_score(text, negative_terms)
        mixed += w * term_score(text, mixed_terms)

    outcome_mode = classify_outcome_mode(
        scenario_type=scenario_type,
        expected_outcome=pieces["expected_outcome"],
        business_response=pieces["business_response"],
        business_scenario=pieces["business_scenario"],
    )

    force_true, force_false, preferred_values = [], [], {}
    for var in variables:
        name = clean(var.get("name"))
        dtype = clean(var.get("dtype")).lower()
        params = var.get("params") if isinstance(var.get("params"), dict) else {}
        if not name or dtype not in {"boolean", "bool", "categorical", "string"}:
            continue

        # A literal CSV value is an explicit instruction and cannot be semantically overridden.
        if params.get("value") is not None:
            continue

        field_text = norm(f"{name} {var.get('description', '')}")
        success_like = any(t in field_text for t in positive_terms)
        failure_like = any(t in field_text for t in negative_terms)
        boolean_signal = any(x in field_text for x in (
            "send", "sent", "contact", "notification", "message", "offer",
            "suppress", "suppressed", "eligible", "accepted", "approved",
            "converted", "retained", "declined", "failed", "success", "successfully",
            "delivered", "retry", "resolved", "conflict", "priority", "allowed", "permitted",
        ))
        status_like = "status" in field_text or success_like or failure_like or "outcome" in field_text or boolean_signal
        if not status_like:
            continue

        choices = params.get("choices") if isinstance(params.get("choices"), list) else []
        normalized_choices = {
            re.sub(r"[^a-z0-9]+", "_", str(choice).strip().lower()).strip("_"): choice
            for choice in choices
        }

        if dtype in {"boolean", "bool"}:
            positive_bool = success_like or any(x in field_text for x in ("accepted", "approved", "converted", "retained", "eligible"))
            negative_bool = failure_like or any(x in field_text for x in ("declined", "failed", "rejected", "suppressed", "denied", "blocked", "ineligible", "cancelled", "abandoned"))
            if outcome_mode == "positive" and positive_bool and not negative_bool:
                force_true.append(name)
            elif outcome_mode == "suppression" and any(x in field_text for x in ("send", "sent", "contact", "notification", "message", "offer", "present", "recommend", "eligible")):
                force_false.append(name)
            elif outcome_mode in {"negative", "decline_or_no_response"} and (positive_bool or negative_bool):
                force_false.append(name)
            elif outcome_mode == "concurrent" and any(x in field_text for x in ("conflict", "competing", "clear priority", "selected", "resolved")):
                force_true.append(name)
        elif dtype in {"categorical", "string"} and choices:
            if outcome_mode == "positive":
                wanted = ["completed", "success", "successful", "complete", "approved", "accepted", "passed", "delivered", "fulfilled", "settled", "authorized", "active", "eligible", "retained"]
            elif outcome_mode == "negative":
                wanted = ["failed", "failure", "unsuccessful", "rejected", "declined", "error", "denied", "blocked", "expired", "cancelled", "canceled", "abandoned", "churned", "ineligible"]
            elif outcome_mode == "decline_or_no_response":
                wanted = ["declined", "decline", "no_response", "no-response", "no response", "rejected", "abandoned", "ignored"]
            elif outcome_mode == "suppression":
                wanted = ["suppressed", "suppression", "not_sent", "not-sent", "held", "hold", "ineligible", "skipped", "disabled"]
            elif outcome_mode == "concurrent":
                wanted = ["no_clear_priority", "no-clear-priority", "multiple_eligible_no_priority", "conflict", "pending_priority", "priority_journey_selected", "highest_priority", "competing", "concurrent", "pending"]
            else:
                wanted = []
            for candidate in wanted:
                normalized = candidate.replace("-", "_").replace(" ", "_")
                if normalized in normalized_choices:
                    preferred_values[name] = [normalized_choices[normalized]]
                    break

    return {
        "mode": scenario_type or "unspecified",
        "outcome_mode": outcome_mode,
        "force_true_fields": force_true,
        "force_false_fields": force_false,
        "preferred_values": preferred_values,
        "context_used": {k: v for k, v in pieces.items() if v},
    }




def temporal_role(var: dict) -> str:
    text = f"{var.get('name', '')} {var.get('description', '')}".lower()
    if any(k in text for k in ("decision", "response", "reply", "decline", "acceptance")):
        return "response"
    if any(k in text for k in ("completion", "completed", "finished", "settled", "processed", "fulfilled")):
        return "completion"
    if any(k in text for k in ("presented", "displayed", "shown", "offered")):
        return "presentation"
    if any(k in text for k in ("sent", "dispatch", "dispatched", "notification")):
        return "dispatch"
    if any(k in text for k in ("created", "creation", "initiated", "start", "started", "opened", "request", "requested")):
        return "start"
    if any(k in text for k in ("end", "ended", "closed", "closure", "expired", "expiry")):
        return "end"
    return "generic"


def temporal_delay_limit_seconds(child: dict, parent: dict) -> int | None:
    text = f"{child.get('name','')} {child.get('description','')}".lower()
    match = re.search(r"within\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)", text)
    if match:
        amount = int(match.group(1))
        return amount * {"second":1,"seconds":1,"minute":60,"minutes":60,"hour":3600,"hours":3600,"day":86400,"days":86400}[match.group(2)]
    if "same day" in text or "same-day" in text:
        return 86400
    child_role = temporal_role(child)
    parent_role = temporal_role(parent)
    if child_role == "response" and parent_role == "presentation":
        return 7 * 86400
    if child_role == "completion" and parent_role in {"presentation", "dispatch", "response", "start"}:
        return 30 * 86400
    if child_role == "dispatch" and parent_role in {"start", "presentation"}:
        return 7 * 86400
    if child_role == "end" and parent_role != "generic":
        return 90 * 86400
    return 90 * 86400
