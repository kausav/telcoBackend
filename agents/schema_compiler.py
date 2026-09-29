"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

import logging
from typing import Any

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship, VariableIdea
import math
import re
from difflib import SequenceMatcher
from core.scenario_semantics import classify_outcome_mode
from core.json_domain_policy import is_json_grounded_domain, is_low_balance_domain, source_manifest
from core.industry_source_store import catalog_for_request, normalize_lookup_key, select_json_source_catalog, canonical_variable_semantic_key, semantic_exclusion_aliases
from core.variable_quality import VariableQualityEngine
from core.temporal_contract import is_supported_temporal_rule
from config.runtime import SCHEMA_MAX_VARIABLES, SCHEMA_MIN_VARIABLE_SCORE


logger = logging.getLogger(__name__)


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1]


def _is_entity_catalog_row(spec: dict[str, Any]) -> bool:
    text = f"{spec.get('model','')} {spec.get('path','')}".lower()
    return any(token in text for token in ("customer", "account", "party", "subscriber", "member", "patient", "policyholder", "user", "profile")) and not any(
        token in text for token in ("transaction", "event", "order", "payment", "claim", "encounter", "visit")
    )



class SchemaCompiler:
    """Compile executable schemas strictly from MongoDB source catalogs or persisted variables."""

    UNSUPPORTED_NESTED_DTYPES = {"object", "array"}
    def __init__(self):
        """Create a compiler whose only industry-standard input is MongoDB."""

    @staticmethod
    def _normalize_variable_key(name: str) -> str:
        """Return a stable full-length identity key for an executable variable name.

        The source catalog can contain deeply nested standards fields whose flattened names are
        longer than the 120-character LLM-facing ``VariableIdea.name`` contract. Identity keys must
        never truncate those names because truncation can merge distinct source fields and silently
        drop variables during deterministic compilation.
        """
        text = str(name or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = "scenario_attribute"
        if text[0].isdigit():
            text = f"feature_{text}"
        return text

    @classmethod
    def _normalize_variable_name(cls, name: str) -> str:
        """Return the legacy 120-character-safe name used for LLM-facing semantic variables.

        Source-backed variables retain their original full names and use ``_normalize_variable_key``
        for identity/deduplication. This 120-character view remains only for semantic/non-source
        naming paths so existing provider and API contracts are not widened unnecessarily.
        """
        return cls._normalize_variable_key(name)[:120]

    @staticmethod
    def _idea_tokens(idea: dict[str, object]) -> set[str]:
        return set(_tokens(" ".join(str(idea.get(k, "")) for k in ("name", "description", "role", "grain"))))

    def _fresh_name(
        self,
        requested: str,
        description: str,
        used: set[str],
        entity_key: str | None = None,
    ) -> str:
        """Preserve semantically meaningful requested names; only disambiguate collisions."""
        base = self._normalize_variable_name(requested)
        if base and base not in used:
            return base

        # Source-backed field names are preserved by the caller; this helper only resolves
        # collisions for non-source semantic names retained for compatibility.
        desc_tokens = [token for token in _tokens(description) if len(token) > 2]
        candidate_bases = []
        if desc_tokens:
            candidate_bases.append(self._normalize_variable_name("_".join(desc_tokens[:4])))
        candidate_bases.append(f"{base}_scenario")

        for candidate in candidate_bases:
            if candidate and candidate not in used:
                return candidate

        index = 2
        while f"{base}_{index}" in used:
            index += 1
        return f"{base}_{index}"

    @staticmethod
    def _description_choices(description: str) -> list[str]:
        """Extract explicit example/allowed values from human-readable descriptions."""
        text = str(description or "").strip()
        if not text:
            return []
        bodies: list[str] = []
        patterns = (
            r"\(([^()]{3,220})\)",
            r"\bsuch as\s+(.+?)(?:\.|$)",
            r"\bvalid values are\s+(.+?)(?:\.|$)",
            r"\bvalues are\s+(.+?)(?:\.|$)",
            r"\be\.g\.?\s*:?\s*(.+?)(?:\.|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                bodies.append(match.group(1).strip())
        for body in bodies:
            parts = [
                re.sub(r"""^[\s"'`]+|[\s"'`]+$""", "", item).strip()
                for item in re.split(r"\s*(?:,|;|\bor\b)\s*", body, flags=re.IGNORECASE)
            ]
            parts = [p for p in parts if p and len(p) <= 80]
            if len(parts) >= 2:
                return list(dict.fromkeys(parts))
        return []

    @classmethod
    def _normalize_semantic_dtype(cls, name: str, description: str, role: str, dtype: str) -> str:
        """Correct obvious LLM type mistakes from the variable's own semantics."""
        n = cls._normalize_variable_name(name)
        text = f"{name} {description}".lower()
        role = str(role or "").lower()
        dtype = str(dtype or "string").lower()

        if n.endswith(("_timestamp", "_time", "_date")) or role == "timing":
            return "date" if n.endswith("_date") and not n.endswith("_datetime") else "datetime"
        if (
            any(token in n for token in ("flag", "is_", "has_", "enabled", "active"))
            or bool(re.match(r"^(?:is|has)[a-z]", n))
        ):
            return "boolean"
        # Unit/currency-unit fields describe denominations, not numeric amounts.
        # Check them before the broad ``_amount`` heuristic so names such as
        # ``topup_amount_currency_unit`` remain strings.
        if n.endswith(("_unit", "_units")) or "currency_unit" in n or "usage_unit" in n:
            return "string"
        if any(token in n for token in ("_amount", "_balance", "_quota", "_score", "_rate", "_percentage", "_percent")):
            return "float"
        if any(token in n for token in ("_count", "_days", "_months", "_hours", "_minutes", "number_of", "num_")):
            return "integer"
        # Preserve explicit boolean declarations before role-based categorical coercion.
        if dtype in {"boolean", "bool"}:
            return "boolean"
        if any(token in n for token in ("_channel", "_method", "_type", "_status", "_state", "_reason", "_category", "_segment", "_capability", "circle")):
            return "categorical"
        if "categorical" in role or role in {"status", "decision", "configuration"}:
            return "categorical"
        return dtype

    def _generic_contract_for_idea(
        self,
        idea: dict[str, object],
        country: str | None,
        scenario_mode: str,
    ) -> tuple[str, str, dict[str, object]] | None:
        """Build a deterministic contract only when the semantic idea is safely executable."""
        name = str(idea.get("name") or "scenario_attribute")
        desc = str(idea.get("description") or "")
        role = str(idea.get("role") or "other").lower()
        dtype = self._normalize_semantic_dtype(
            name, desc, role, str(idea.get("dtype") or "string").lower()
        )
        lower = f"{name} {desc}".lower()
        country_code = str(country or "IN").upper()

        if name.strip().lower() == "msisdn":
            dial_codes = {
                "IN": "+91", "US": "+1", "CA": "+1", "GB": "+44", "AU": "+61",
                "AE": "+971", "SG": "+65", "DE": "+49", "FR": "+33", "IT": "+39",
            }
            dial = dial_codes.get(country_code, country_code if country_code.startswith("+") else "+" + country_code)
            return "e164_phone", "string", {"country_codes": [dial], "country": country_code}

        if name.strip().lower() in {"subscriber_id", "account_id"}:
            prefix = "SUB-" if name.strip().lower() == "subscriber_id" else "ACC-"
            if name.strip().lower() == "account_id":
                return "id_mirror", "string", {
                    "prefix": prefix, "source_field": "subscriber_id", "source_prefix": "SUB-"
                }
            return "prefixed_int", "string", {"prefix": prefix, "digits": 10}

        if role == "identity" or name.lower().endswith(("_id", "_key")) or name.lower() == "id":
            prefix = f"{name[:-3].upper()}-" if name.lower().endswith("_id") else "ID-"
            return "prefixed_int", "string", {"prefix": prefix, "digits": 10}

        if dtype in self.UNSUPPORTED_NESTED_DTYPES:
            return None

        if dtype in {"datetime", "timestamp"} or role == "timing":
            return "recent_datetime", "datetime", {
                "timezone": "Asia/Kolkata" if country_code == "IN" else "UTC",
                "days_back": 365,
            }
        if dtype == "date":
            return "recent_date", "date", {
                "timezone": "Asia/Kolkata" if country_code == "IN" else "UTC",
                "days_back": 365,
            }
        # Preserve an explicit boolean declaration even when the semantic role is
        # ``decision`` or ``status``. A boolean eligibility/flag must not be coerced into
        # an open-ended categorical field merely because its business role is a decision.
        if dtype in {"boolean", "bool"}:
            return "weighted_choice", "boolean", {"choices": [False, True], "weights": [0.5, 0.5]}

        if name.strip().lower() == "recharge_plan_validity_days":
            return "uniform_int", "integer", {"min": 1, "max": 84, "precision": 0}
        if dtype in {"integer", "int"} or role == "metric":
            return "uniform_int", "integer", {"min": 0, "max": 100, "precision": 0}
        if dtype in {"float", "decimal", "number", "numeric"} or role == "measurement":
            return "uniform", "float", {"min": 0.0, "max": 1000.0, "precision": 2}
        if dtype == "categorical" or role in {"status", "decision", "configuration", "categorical"}:
            explicit = self._description_choices(desc)
            if explicit:
                return "weighted_choice", "categorical", {"choices": explicit, "weights": [1.0] * len(explicit)}

            if "network" in lower and "capability" in lower:
                choices = ["2G", "3G", "4G", "5G"]
            elif "channel" in lower:
                choices = ["APP", "SMS", "WEB", "USSD", "WHATSAPP", "IVR", "RETAIL"]
            elif "payment" in lower and ("instrument" in lower or "method" in lower):
                choices = ["UPI", "CREDIT_CARD", "DEBIT_CARD", "WALLET", "CASH", "AUTO_DEBIT"]
            elif "recharge_plan_code" in name.lower() or ("plan" in lower and "code" in lower):
                choices = ["PREPAID_1D", "PREPAID_7D", "PREPAID_14D", "PREPAID_28D", "PREPAID_30D", "PREPAID_56D", "PREPAID_84D"]
            elif "offer" in lower:
                choices = ["EXTRA_DATA", "CASH_BACK", "VALIDITY_BOOSTER", "DISCOUNT_VOUCHER"]
            elif "segment" in lower:
                choices = ["ULTRA_LOW", "MASS", "MID_TIER", "HIGH_VALUE"]
            elif "reason" in lower:
                choices = ["LOW_BALANCE", "DATA_EXHAUSTED", "VALIDITY_EXPIRY", "CUSTOMER_REQUEST"]
            elif role == "status" or any(token in lower for token in (" status", "_status", " state")):
                mode_values = {
                    "positive": ["COMPLETED", "SUCCESS", "APPROVED"],
                    "negative": ["FAILED", "ERROR", "REJECTED"],
                    "suppression": ["SUPPRESSED", "HELD", "SKIPPED"],
                    "decline_or_no_response": ["DECLINED", "NO_RESPONSE", "REJECTED"],
                    "concurrent": ["NO_CLEAR_PRIORITY", "CONFLICT", "PENDING_PRIORITY"],
                    "mixed": ["COMPLETED", "FAILED", "PENDING"],
                }
                choices = mode_values.get(scenario_mode or "mixed", mode_values["mixed"])
            else:
                return None
            return "weighted_choice", "categorical", {"choices": choices, "weights": [1.0] * len(choices)}

        # Do not manufacture a meaningless generic string contract. The field must either
        # be grounded in a concrete registry generator or have a recognized semantic contract.
        return None

    @staticmethod
    def _json_source_contract(
        spec: dict[str, object],
        country: str | None,
    ) -> tuple[str, str, dict]:
        """Create an executable generator contract from a flattened Swagger scalar leaf."""
        dtype = str(spec.get("dtype") or "string").strip().lower()
        fmt = str(spec.get("format") or "").strip().lower()
        enum_values = list(spec.get("enum_values") or [])
        params: dict = {}

        if enum_values:
            return "weighted_choice", "categorical", {"choices": enum_values, "weights": [1.0] * len(enum_values)}

        # Honor JSON Schema/OpenAPI temporal formats before generic string semantics. Swagger often
        # declares dates as ``type=string, format=date-time``; treating those as ordinary semantic
        # strings produces non-dates that later validators cannot reason about chronologically.
        if fmt in {"date-time", "datetime", "timestamp"} or dtype in {"date-time", "datetime", "timestamp"}:
            return "recent_datetime", "datetime", {
                "timezone": "Asia/Kolkata" if str(country or "IN").upper() == "IN" else "UTC",
                "days_back": 365,
            }
        if fmt == "date" or dtype == "date":
            return "recent_date", "date", {
                "timezone": "Asia/Kolkata" if str(country or "IN").upper() == "IN" else "UTC",
                "days_back": 365,
            }

        # Honor standard-defined string formats before generic semantic name heuristics. This is
        # important for non-telecom sources where a field named ``id`` may legally be a UUID/URI.
        if dtype in {"string", "str", "text"}:
            string_params: dict[str, object] = {}
            if fmt:
                string_params["format"] = fmt
            if country:
                string_params["country"] = country
            if spec.get("minLength") is not None:
                string_params["min_length"] = spec.get("minLength")
            if spec.get("maxLength") is not None:
                string_params["max_length"] = spec.get("maxLength")
            if spec.get("pattern") is not None:
                string_params["pattern"] = spec.get("pattern")
            # Every MongoDB-backed JSON string is source-contract-bound. There is no
            # filesystem/telecom fallback for an industry source field.
            string_params["source_contract"] = True
            if fmt in {"uuid", "uuid4"}:
                return "uuid_string", "string", string_params
            if fmt in {"email", "idn-email"}:
                return "email_string", "string", string_params
            if fmt in {"uri", "uri-reference", "url"}:
                return "uri_string", "string", string_params
            if fmt == "ipv4":
                return "ipv4_string", "string", string_params
            if fmt == "ipv6":
                return "ipv6_string", "string", string_params
            return "semantic_string", "string", string_params

        # Some official Swagger string definitions encode a constrained vocabulary only
        # in their descriptions (for example RelatedTopupBalance.role = parent/child).
        # Materialize those explicit source-described choices instead of falling back to
        # generic semantic strings.
        description = str(spec.get("description") or "")
        explicit = SchemaCompiler._description_choices(description)
        path_lower = str(spec.get("path") or "").lower()
        if explicit and (path_lower.endswith(".role") or "valid values" in description.lower()):
            return "weighted_choice", "categorical", {"choices": explicit, "weights": [1.0] * len(explicit)}
        if dtype in {"boolean", "bool"}:
            return "weighted_choice", "boolean", {"choices": [False, True], "weights": [0.5, 0.5]}
        if dtype in {"integer", "int", "bigint", "smallint"}:
            minimum_raw = spec.get("minimum", 0)
            maximum_raw = spec.get("maximum", 100)
            exclusive_min = spec.get("exclusiveMinimum")
            exclusive_max = spec.get("exclusiveMaximum")
            try:
                minimum = float(minimum_raw)
            except (TypeError, ValueError):
                minimum = 0.0
            try:
                maximum = float(maximum_raw)
            except (TypeError, ValueError):
                maximum = max(minimum + 1.0, 100.0)
            if isinstance(exclusive_min, bool):
                if exclusive_min:
                    minimum = math.floor(minimum) + 1
            elif exclusive_min is not None:
                try:
                    minimum = math.floor(float(exclusive_min)) + 1
                except (TypeError, ValueError):
                    pass
            if isinstance(exclusive_max, bool):
                if exclusive_max:
                    maximum = math.ceil(maximum) - 1
            elif exclusive_max is not None:
                try:
                    maximum = math.ceil(float(exclusive_max)) - 1
                except (TypeError, ValueError):
                    pass
            minimum = int(math.ceil(minimum))
            maximum = int(math.floor(maximum))
            if maximum < minimum:
                maximum = minimum
            params = {"min": minimum, "max": maximum, "precision": 0}
            if spec.get("multipleOf") is not None:
                params["multiple_of"] = spec["multipleOf"]
            return "uniform_int", "integer", params
        if dtype in {"number", "float", "double", "decimal", "numeric"}:
            minimum_raw = spec.get("minimum", 0.0)
            maximum_raw = spec.get("maximum", 1000.0)
            exclusive_min = spec.get("exclusiveMinimum")
            exclusive_max = spec.get("exclusiveMaximum")
            try:
                minimum = float(minimum_raw)
            except (TypeError, ValueError):
                minimum = 0.0
            try:
                maximum = float(maximum_raw)
            except (TypeError, ValueError):
                maximum = max(minimum + 1.0, 1000.0)
            if isinstance(exclusive_min, bool):
                if exclusive_min:
                    minimum = math.nextafter(minimum, math.inf)
            elif exclusive_min is not None:
                try:
                    minimum = math.nextafter(float(exclusive_min), math.inf)
                except (TypeError, ValueError):
                    pass
            if isinstance(exclusive_max, bool):
                if exclusive_max:
                    maximum = math.nextafter(maximum, -math.inf)
            elif exclusive_max is not None:
                try:
                    maximum = math.nextafter(float(exclusive_max), -math.inf)
                except (TypeError, ValueError):
                    pass
            if maximum < minimum:
                maximum = minimum
            precision = int(spec.get("precision") or 2) if str(spec.get("precision") or "").isdigit() else 2
            params = {"min": minimum, "max": maximum, "precision": max(0, min(8, precision))}
            if spec.get("multipleOf") is not None:
                params["multiple_of"] = spec["multipleOf"]
            return "uniform", "float", params
        # Identifier/reference semantics are handled by the semantic generator after all
        # primitive/standard formats above have been accounted for.
        return "semantic_string", "string", {}

    @staticmethod
    def _merge_dependencies(
        idea: dict[str, object],
        name_map: dict[str, str],
        grain: str,
        entity_key: str | None,
    ) -> list[str]:
        deps: list[str] = []
        for raw in idea.get("depends_on") or []:
            key = SchemaCompiler._normalize_variable_name(str(raw))
            mapped = name_map.get(key) or name_map.get(str(raw))
            if mapped and mapped not in deps:
                deps.append(mapped)
        if grain in {"transaction", "event", "derived"} and entity_key and entity_key not in deps:
            deps.insert(0, entity_key)
        return deps

    def _build_fresh_fields(
        self,
        intent: ScenarioIntent,
        entities: list[Any],
        *,
        entity_key: str | None,
        country: str | None,
        type_of_data: str | None,
        scenario_mode: str,
        max_variables: int | None = None,
        include_all_registry_scalars: bool = True,
        include_all_json_source_scalars: bool = False,
        include_application_telecom_anchors: bool = True,
        excluded_field_names: list[str] | None = None,
        business_scenario: str | None = None,
        context_text: str | None = None,
        candidate_variables_override: list[dict[str, object]] | None = None,
    ) -> list[GeneratedSchemaField]:
        normalized_type = str(type_of_data or intent.type_of_data or "transactional").strip().lower()
        normalized_industry = str(intent.industry_type or "generic").strip().lower()
        telecom_context = normalized_industry in {"telecom", "telecommunications", "telecommunication"}
        excluded_keys = {
            self._normalize_variable_key(name)
            for name in (excluded_field_names or [])
            if self._normalize_variable_key(name)
        }
        raw_ideas = (
            [dict(idea) for idea in candidate_variables_override]
            if candidate_variables_override is not None
            else [idea.model_dump() for idea in intent.candidate_variables]
        )
        ideas: list[dict[str, object]] = []
        seen_idea_keys: set[str] = set()

        def add_idea(
            idea: dict[str, object],
            *,
            force_name: str | None = None,
            preserve_name: bool = False,
        ) -> None:
            item = dict(idea)
            if force_name:
                item["name"] = force_name
            if preserve_name:
                item["_preserve_name"] = True
            key = (
                self._normalize_variable_key(str(item.get("name") or ""))
                if bool(item.get("_json_source_spec"))
                else self._normalize_variable_name(str(item.get("name") or ""))
            )
            if not key or key in seen_idea_keys:
                return
            seen_idea_keys.add(key)
            ideas.append(item)

        # Keep LLM-proposed scenario variables even when their names are similar to official
        # attributes. The official scalar catalog is already de-duplicated by exact field name,
        # while scenario-derived analytics such as ``balance_remaining_amount`` and
        # ``topup_recharge_amount`` can intentionally coexist with their source-backed fields.
        # Fuzzy suppression previously removed legitimate scenario variables and narrowed the
        # generated schema below the requested business scope.
        for idea in raw_ideas:
            add_idea(idea)
        if entity_key:
            add_idea({
                "name": entity_key,
                "description": f"Stable identifier for the requested {intent.use_case or 'telecom'} entity.",
                "role": "identity",
                "grain": "entity",
                "dtype": "string",
                "depends_on": [],
            }, force_name=entity_key)

        # Subscriber is the canonical telecom subscriber identity for proposal generation.
        # A separate customer_id is redundant in the flat scenario contract unless the caller
        # explicitly requested customer_id as the transactional entity key. Suppress it before
        # quality scoring so it cannot consume the variable budget or be reintroduced by
        # standards expansion.
        has_subscriber_anchor = any(
            self._normalize_variable_key(str(item.get("name") or "")) == "subscriber_id"
            for item in ideas
        )
        explicit_customer_entity_key = self._normalize_variable_name(entity_key or "") == "customer_id"
        if telecom_context and has_subscriber_anchor and not explicit_customer_entity_key:
            ideas = [
                item for item in ideas
                if self._normalize_variable_key(str(item.get("name") or "")) != "customer_id"
            ]

        # For JSON-source compilation, candidate_variables_override has already passed through the
        # deterministic source relevance selector. Running a second quality-based pruning stage would
        # narrow the authoritative source-backed selection and could discard valid fields after the
        # source relevance decision has already been made. Preserve the selected source set and use
        # the quality engine only for per-field scoring and provenance below.
        quality_engine = VariableQualityEngine(
            max_variables=max_variables if max_variables is not None else SCHEMA_MAX_VARIABLES,
            min_score=SCHEMA_MIN_VARIABLE_SCORE,
        )
        selection_context = context_text or " ".join(
            str(value or "")
            for value in (
                intent.industry_type, intent.domain, intent.subdomain, intent.scenario_type,
                intent.type_of_data, intent.use_case, intent.entity_key,
                business_scenario or "",
            )
        )
        if candidate_variables_override is not None and all(
            isinstance(item.get("_json_source_spec"), dict) for item in ideas
        ):
            quality_report = {
                "candidate_count": len(ideas),
                "selected_count": len(ideas),
                "maximum": max_variables if max_variables is not None else SCHEMA_MAX_VARIABLES,
                "minimum_quality_score": SCHEMA_MIN_VARIABLE_SCORE,
                "semantic_duplicates_removed": 0,
                "low_quality_candidates_removed": 0,
                "quality_budget_truncated": 0,
                "dependency_closure_added": 0,
                "dependency_aliases": {},
            }
        else:
            ideas, quality_report = quality_engine.select(
                ideas,
                context_text=selection_context,
                entity_key=entity_key,
            )

        # Defensive post-selection guard for the same redundancy rule. This prevents any
        # future quality-engine dependency closure change from reintroducing customer_id into
        # a subscriber-anchored telecom proposal.
        if telecom_context and has_subscriber_anchor and not explicit_customer_entity_key:
            ideas = [
                item for item in ideas
                if self._normalize_variable_key(str(item.get("name") or "")) != "customer_id"
            ]
        dependency_aliases = quality_report.get("dependency_aliases", {})
        if dependency_aliases:
            for idea in ideas:
                idea["depends_on"] = [
                    dependency_aliases.get(self._normalize_variable_name(str(dep)), str(dep))
                    for dep in (idea.get("depends_on") or [])
                ]

        fields: list[GeneratedSchemaField] = []
        used_names: set[str] = set()
        name_map: dict[str, str] = {}

        # First assign names so later dependency references can be resolved to the fresh names.
        # Source-backed names are kept at full length because deeply nested standards fields may
        # legitimately exceed the 120-character LLM-facing semantic-name limit.
        for idea in ideas:
            current_requested = str(idea.get("name") or "scenario_attribute")
            if bool(idea.get("_json_source_spec")):
                fresh = current_requested
            else:
                fresh = self._fresh_name(
                    current_requested,
                    str(idea.get("description") or ""),
                    used_names,
                    entity_key,
                )
            used_names.add(fresh)
            name_map[self._normalize_variable_key(str(idea.get("name") or ""))] = fresh

        # No static industry/country profile is used here. Source JSON constraints are authoritative.
        datetime_by_name: dict[str, dict] = {}
        for candidate in ideas:
            candidate_name = self._normalize_variable_key(str(candidate.get("name") or ""))
            candidate_spec = candidate.get("_json_source_spec") if isinstance(candidate.get("_json_source_spec"), dict) else None
            if candidate_spec is None:
                continue
            _candidate_gen, candidate_dtype, _candidate_params = self._json_source_contract(candidate_spec, country)
            if str(candidate_dtype).strip().lower() == "datetime":
                datetime_by_name[candidate_name] = {
                    "name": name_map.get(candidate_name, str(candidate.get("name") or "")),
                    "dtype": "datetime",
                    "provenance": {"source_json_model": candidate_spec.get("model")},
                }

        for idea in ideas:
            original_name = str(idea.get("name") or "scenario_attribute")
            fresh = name_map[self._normalize_variable_key(original_name)]
            source_spec = idea.get("_json_source_spec") if isinstance(idea.get("_json_source_spec"), dict) else None
            if source_spec is None:
                raise ValueError(
                    f"Ungrounded executable variable '{original_name}' reached MongoDB JSON compilation"
                )
            runtime_generator, dtype, params = self._json_source_contract(
                source_spec, country
            )
            if not runtime_generator or not dtype or not isinstance(params, dict):
                # Fail closed: every source-backed field must have a concrete deterministic generator.
                raise ValueError(
                    f"Source-backed field '{original_name}' has no executable generator contract"
                )
            # The executable contract is derived exclusively from the selected MongoDB JSON
            # leaf. No static industry profile or LLM-invented fallback is consulted here.

            role = str(idea.get("role") or "other")
            grain = str(idea.get("grain") or ("entity" if original_name == entity_key else "transaction"))
            raw_dependencies = [str(dep) for dep in (idea.get("depends_on") or []) if str(dep).strip()]
            unresolved_dependencies = [
                dep for dep in raw_dependencies
                if self._normalize_variable_key(dep) not in name_map
                and self._normalize_variable_key(dependency_aliases.get(self._normalize_variable_key(dep), dep)) not in name_map
            ]
            if unresolved_dependencies:
                # A selected field with an unrepresentable prerequisite cannot produce a faithful
                # executable contract. Fail confirmation rather than silently weakening the source contract.
                raise ValueError(
                    f"Source-backed field '{original_name}' has unresolved dependencies: {unresolved_dependencies}"
                )
            deps = self._merge_dependencies(idea, name_map, grain, entity_key)
            if str(dtype).strip().lower() == "datetime":
                safe_deps: list[str] = []
                child_meta = {
                    "name": fresh,
                    "dtype": "datetime",
                    "provenance": {"source_json_model": source_spec.get("model")},
                }
                for dep in deps:
                    parent_meta = datetime_by_name.get(self._normalize_variable_key(str(dep)))
                    if parent_meta and not is_supported_temporal_rule(parent_meta, child_meta):
                        logger.warning(
                            "[SchemaCompiler] Removed unsafe cross-resource datetime dependency %s -> %s",
                            dep, original_name,
                        )
                        continue
                    safe_deps.append(dep)
                deps = safe_deps
            if original_name == entity_key:
                grain = "entity"
                deps = []
            required = bool(source_spec.get("required")) or original_name == entity_key
            nullable = False if required else bool(source_spec.get("nullable", not bool(source_spec.get("required"))))
            description = str(idea.get("description") or "").strip() or f"Scenario-specific {role.replace('_', ' ')} attribute for {intent.domain}."
            source_from_json = isinstance(source_spec, dict)
            provenance = {
                "generated_from": "mongodb_json_source",
                "canonical_entity": f"{source_spec.get('source_id','')}__{source_spec.get('model','')}".strip("_"),
                "source_json_id": source_spec.get("source_id") if source_from_json else None,
                "source_json_model": source_spec.get("model") if source_from_json else None,
                "source_json_path": source_spec.get("path") if source_from_json else None,
                "source_json_semantic_key": source_spec.get("semantic_key") if source_from_json else None,
                "source_json_paths": list(source_spec.get("source_paths") or []) if source_from_json else [],
                "source_json_aliases": list(source_spec.get("source_aliases") or []) if source_from_json else [],
                "grain": grain,
                "quality_score": quality_engine.score(idea, selection_context, entity_key).score,
                "quality_reasons": list(quality_engine.score(idea, selection_context, entity_key).reasons),
            }
            fields.append(GeneratedSchemaField(
                name=fresh,
                dtype=dtype,
                description=description,
                gen=runtime_generator,
                params=params,
                depends_on=deps,
                nullable=nullable,
                required=required,
                # Source formulas are not copied unless explicitly represented by the source contract.
                formula=None,
                provenance=provenance,
                scope=grain,
            ))
        return fields

    def _compile_json_source_grounded(
        self,
        intent: ScenarioIntent,
        *,
        industry_type: str,
        domain: str,
        business_scenario: str,
        use_case: str,
        country: str | None,
        type_of_data: str | None,
        entity_key: str | None,
        scenario_type: str | None = None,
        excluded_field_names: list[str] | None = None,
        external_variable_names: set[str] | None = None,
        external_variable_definitions: list[dict[str, Any]] | None = None,
        max_variables: int | None = None,
    ) -> ScenarioSchema:
        """Compile any MongoDB-backed standards domain without assuming telecom semantics."""
        catalog_payload = catalog_for_request(industry_type, domain)
        catalog_rows = [dict(row) for row in (catalog_payload.get("models") or []) if isinstance(row, dict)]
        if not catalog_rows:
            raise ValueError(
                f"No usable active JSON source variables exist for industryType='{industry_type}' and domain='{domain}'."
            )

        normalized_type = str(type_of_data or intent.type_of_data or "transactional").strip().lower()
        business_context = " ".join(
            str(value or "") for value in (
                industry_type, domain, scenario_type or intent.scenario_type, normalized_type,
                use_case, business_scenario, country or "", intent.entity_key or "",
            )
        )
        preferred_names = {
            normalize_lookup_key(idea.name)
            for idea in intent.candidate_variables
            if normalize_lookup_key(idea.name)
        }
        if entity_key and normalize_lookup_key(entity_key):
            preferred_names.add(normalize_lookup_key(entity_key))
        excluded = {
            normalize_lookup_key(name)
            for name in (excluded_field_names or [])
            if normalize_lookup_key(name)
        }
        requested_budget = max(0, int(max_variables if max_variables is not None else SCHEMA_MAX_VARIABLES))
        db_variable_count = sum(1 for item in (external_variable_definitions or []) if isinstance(item, dict) and str(item.get("name") or "").strip())
        variable_budget = max(0, requested_budget - db_variable_count)
        external_semantic_aliases: set[str] = set()
        for external in external_variable_definitions or []:
            if isinstance(external, dict):
                external_semantic_aliases.update(semantic_exclusion_aliases(external))
        selected_rows, selection_report = select_json_source_catalog(
            catalog_rows,
            business_context=business_context,
            preferred_names=preferred_names,
            excluded_names=excluded,
            excluded_semantic_keys=external_semantic_aliases,
            max_fields=variable_budget,
        )
        selected_names = {normalize_lookup_key(row.get("name")) for row in selected_rows}
        external_keys = {
            normalize_lookup_key(name)
            for name in (external_variable_names or set())
            if normalize_lookup_key(name)
        }

        all_identity_names = [
            normalize_lookup_key(row.get("name"))
            for row in catalog_rows
            if normalize_lookup_key(row.get("name")) and self._normalize_variable_key(row.get("name"))
            and str(row.get("name") or "").lower().endswith(("_id", "_key"))
        ]
        resolved_entity_key = None
        if normalized_type == "transactional":
            requested_key = self._normalize_variable_key(entity_key or "")
            if requested_key:
                if requested_key not in {self._normalize_variable_key(row.get("name")) for row in catalog_rows} and requested_key not in external_keys:
                    raise ValueError(
                        f"Requested entity key '{entity_key}' is not present in the active JSON source catalog or MongoDB variables."
                    )
                resolved_entity_key = entity_key.strip()
            else:
                chosen_key = next((name for name in all_identity_names if name in selected_names), None) or (all_identity_names[0] if all_identity_names else None)
                if chosen_key:
                    resolved_entity_key = next(
                        (str(row.get("name")) for row in catalog_rows if normalize_lookup_key(row.get("name")) == chosen_key),
                        chosen_key,
                    )

        selected_source_ideas: list[dict[str, object]] = []
        for spec in selected_rows:
            dtype = str(spec.get("dtype") or "string").strip().lower()
            normalized_dtype = (
                "integer" if dtype in {"integer", "int", "bigint", "smallint"}
                else "float" if dtype in {"number", "float", "double", "decimal", "numeric"}
                else "datetime" if dtype in {"date-time", "datetime", "timestamp"} or str(spec.get("format") or "").lower() == "date-time"
                else "date" if dtype == "date"
                else "categorical" if spec.get("enum_values")
                else "boolean" if dtype in {"boolean", "bool"}
                else "string"
            )
            role = (
                "timing" if normalized_dtype in {"datetime", "date"}
                else "measurement" if normalized_dtype in {"integer", "float"}
                else "identity" if normalize_lookup_key(spec.get("name")).endswith(("_id", "_key"))
                else "status" if spec.get("enum_values") and any(token in f"{spec.get('name','')} {spec.get('path','')}".lower() for token in ("status", "state", "reason"))
                else "categorical" if spec.get("enum_values")
                else "other"
            )
            grain = "entity" if _is_entity_catalog_row(spec) else "transaction"
            selected_source_ideas.append({
                "name": str(spec["name"]),
                "description": str(spec.get("description") or "")[:500],
                "role": role if role in {"identity", "profile", "event", "transaction", "status", "measurement", "metric", "timing", "decision", "configuration", "derived", "other"} else "other",
                "grain": grain,
                "dtype": normalized_dtype,
                "depends_on": [],
                "_json_source_spec": dict(spec),
                "_json_source": True,
                "_preserve_name": True,
                "_source_model": f"{spec.get('source_id','')}__{spec.get('model','')}",
                "_source_required": bool(spec.get("required")),
                "_source_nullable": not bool(spec.get("required")),
            })

        working_intent = intent
        build_entity_key = (
            resolved_entity_key
            if resolved_entity_key and self._normalize_variable_key(resolved_entity_key) in selected_names
            else None
        )
        fields = self._build_fresh_fields(
            working_intent,
            [],
            entity_key=build_entity_key,
            country=country or "GLOBAL",
            type_of_data=normalized_type,
            scenario_mode=str(scenario_type or intent.scenario_type or "mixed"),
            max_variables=max(len(selected_source_ideas), variable_budget),
            include_all_registry_scalars=False,
            include_all_json_source_scalars=False,
            include_application_telecom_anchors=False,
            excluded_field_names=excluded_field_names,
            business_scenario=business_scenario,
            context_text=business_context,
            candidate_variables_override=selected_source_ideas,
        )

        actual_names = {self._normalize_variable_key(field.name) for field in fields}
        missing_source_fields = sorted(selected_names - actual_names)
        if missing_source_fields:
            raise ValueError(
                "JSON-source variables were lost during deterministic compilation: " + ", ".join(missing_source_fields[:25])
            )
        if normalized_type == "transactional" and resolved_entity_key and normalize_lookup_key(resolved_entity_key) not in actual_names and normalize_lookup_key(resolved_entity_key) not in external_keys:
            raise ValueError(f"Transactional entity key '{resolved_entity_key}' could not be represented by the active JSON source fields.")

        model_groups: dict[tuple[str, str], dict[str, Any]] = {}
        for spec in selected_rows:
            key = (str(spec.get("source_id") or ""), str(spec.get("model") or ""))
            model_groups.setdefault(key, spec)
        entities = [
            ResolvedConcept(
                canonical_id=f"{source_id}__{model}".strip("_"),
                name=model or source_id,
                source_model=str(spec.get("standard") or source_id),
                source_references=[str(source_id)],
                selected_attributes=sorted(
                    normalize_lookup_key(row.get("name"))
                    for row in selected_rows
                    if str(row.get("source_id") or "") == source_id and str(row.get("model") or "") == model
                ),
            )
            for (source_id, model), spec in sorted(model_groups.items())
        ]
        hard_constraints = [
            "Executable source variables are restricted to scalar leaves extracted from active MongoDB JSON documents for the exact industryType/domain pair.",
            "The LLM is a selector only; it cannot invent, rename, alias, or derive executable source variable names.",
            "Exact scalar names from the active catalog are preserved through deterministic compilation.",
            "Standard enum values and declared numeric constraints from the source documents are authoritative.",
            "Arrays and complex objects are not emitted as fake flat scalar variables.",
        ]
        if resolved_entity_key:
            hard_constraints.append(f"'{resolved_entity_key}' is the authoritative entity key for transactional grouping.")
        warnings = [
            f"Source catalog selection: {selection_report['selected_count']} of {selection_report['candidate_count']} scalar variables were selected deterministically for this request.",
            "Adding or replacing industry/domain standard JSONs changes the available source catalog without requiring code changes.",
        ]
        source_docs = source_manifest(industry_type, domain)
        return ScenarioSchema(
            domain=intent.domain,
            subdomain="unknown",
            applicable_standards=source_docs,
            entities=entities,
            relationships=[],
            fields=fields,
            hard_constraints=hard_constraints,
            unresolved_items=[],
            warnings=warnings,
        )

    def compile(
        self,
        intent: ScenarioIntent,
        selected_entities: list[str] | None = None,
        max_variables: int | None = None,
        domain_query: str | None = None,
        entity_key: str | None = None,
        industry_type: str | None = None,
        scenario_type: str | None = None,
        type_of_data: str | None = None,
        use_case: str | None = None,
        business_scenario: str | None = None,
        business_response: str | None = None,
        expected_outcome: str | None = None,
        country: str | None = None,
        excluded_field_names: list[str] | None = None,
        external_variable_names: set[str] | None = None,
        external_variable_definitions: list[dict[str, Any]] | None = None,
    ) -> ScenarioSchema:
        """Compile only from active MongoDB industry/domain JSON sources."""
        normalized_industry = (industry_type or intent.industry_type or "").strip()
        domain = (domain_query or intent.domain or "").strip()
        if not normalized_industry or not domain or not is_json_grounded_domain(domain, normalized_industry):
            raise ValueError(
                f"No active JSON source documents are registered for industryType='{normalized_industry}', domain='{domain}'. "
                "Upload at least one industry-standard JSON to MongoDB before compiling this scenario."
            )
        return self._compile_json_source_grounded(
            intent,
            industry_type=normalized_industry,
            domain=domain,
            business_scenario=business_scenario or "",
            use_case=use_case or intent.use_case or "",
            country=country,
            type_of_data=type_of_data or intent.type_of_data,
            entity_key=entity_key,
            scenario_type=scenario_type,
            excluded_field_names=excluded_field_names,
            external_variable_names=external_variable_names,
            external_variable_definitions=external_variable_definitions,
            max_variables=max_variables,
        )

    def approval_questions(self, intent: ScenarioIntent, schema: ScenarioSchema) -> list[str]:
        questions = list(intent.ambiguities)
        if intent.record_count is None:
            questions.append("How many records/entities should be generated?")
        if intent.country is None:
            questions.append("Which country/market should be modeled?")
        if "recharge" in {e.canonical_id for e in schema.entities} and "usage_event" in {e.canonical_id for e in schema.entities}:
            questions.append("Should recharge and usage histories be generated as a causal timeline with balance impact enabled?")
        return questions
