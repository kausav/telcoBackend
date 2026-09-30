"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

import logging
import hashlib
from typing import Any

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship, VariableIdea
import math
import re
from difflib import SequenceMatcher
from core.scenario_semantics import classify_outcome_mode
from core.json_domain_policy import is_json_grounded_domain, is_low_balance_domain
from core.industry_source_store import catalog_for_request, normalize_lookup_key, select_json_source_catalog, canonical_variable_semantic_key, semantic_exclusion_aliases, _catalog_role
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

        JSON-source variables use their full source identity during compilation and are assigned a
        separate human-readable path name in the final source-field naming pass. This method remains
        limited to semantic/non-source naming paths so existing provider/API contracts are unchanged.
        """
        return cls._normalize_variable_key(name)[:120]

    SOURCE_RUNTIME_NAME_MAX_LENGTH = 32
    _GENERIC_SOURCE_LEAF_TOKENS = {
        "id", "key", "name", "label", "type", "status", "state", "reason", "value",
        "amount", "unit", "date", "time", "datetime", "timestamp", "role", "code",
        "description", "href", "url", "duration", "action", "result", "category", "method",
        "channel", "price", "period", "count", "number", "reference", "identifier",
        "visible", "shared", "enabled", "active", "available", "valid",
    }
    _SOURCE_BOOLEAN_PREFIXES = {"is", "has", "can", "should"}

    @staticmethod
    def _source_semantic_segments(field: GeneratedSchemaField) -> list[str]:
        """Return normalized semantic-path segments recorded by the JSON source."""
        provenance = field.provenance or {}
        semantic_key = str(provenance.get("source_json_semantic_key") or "").strip()
        if semantic_key:
            segments: list[str] = []
            for raw_segment in semantic_key.split("."):
                normalized = SchemaCompiler._normalize_variable_key(raw_segment)
                if normalized:
                    segments.append(normalized)
            if segments:
                return segments
        fallback = SchemaCompiler._normalize_variable_key(field.name)
        return [fallback] if fallback else ["scenario_attribute"]

    @classmethod
    def _source_candidate_tokens(cls, segments: list[str]) -> list[str]:
        """Flatten the source path while removing only redundant structural repetition."""
        output: list[str] = []
        for segment in segments:
            tokens = [token for token in str(segment or "").split("_") if token]
            if not tokens:
                continue
            max_overlap = min(len(output), len(tokens) - 1)
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if output[-size:] == tokens[:size]:
                    overlap = size
                    break
            output.extend(tokens[overlap:])

        collapsed: list[str] = []
        for token in output:
            if collapsed and collapsed[-1] == token:
                continue
            collapsed.append(token)

        # Flattened Money/Quantity wrappers frequently end with ``amount.value``. The leaf ``value``
        # does not add a second business concept, so keep the source meaning as ``amount``.
        if len(collapsed) >= 2 and collapsed[-1] == "value" and collapsed[-2] == "amount":
            collapsed.pop()
        if len(collapsed) >= 2 and collapsed[-1] == "amount" and collapsed[-2] == "amount":
            collapsed.pop()
        return collapsed

    @classmethod
    def _source_candidate_forms(cls, segments: list[str]) -> list[str]:
        """Build compact whole-word names from the leaf side of the source semantic path."""
        tokens = cls._source_candidate_tokens(segments)
        if not tokens:
            return []

        leaf_is_generic = tokens[-1] in cls._GENERIC_SOURCE_LEAF_TOKENS
        has_boolean_prefix = any(token in cls._SOURCE_BOOLEAN_PREFIXES for token in tokens[-3:])
        if has_boolean_prefix:
            minimum_tokens = min(3, len(tokens))
        elif tokens[-1] in {"amount", "value", "quantity", "count"} and len(tokens) > 2:
            minimum_tokens = 3
        elif leaf_is_generic and len(tokens) > 2:
            # Generic identifiers/statuses/names are too ambiguous on their own. Keep at least
            # three semantic words for oversized source paths whenever the source provides them.
            minimum_tokens = 3
        elif leaf_is_generic and len(tokens) > 1:
            minimum_tokens = 2
        else:
            minimum_tokens = 1
        forms: list[str] = []
        seen: set[str] = set()

        # Prefer 2-4 semantic words. Larger context is considered only when needed for uniqueness.
        max_window = min(len(tokens), 6)
        preferred_sizes = list(range(minimum_tokens, max_window + 1))
        preferred_sizes.sort(key=lambda size: (0 if 2 <= size <= 4 else 1, size))
        for size in preferred_sizes:
            candidate = cls._normalize_variable_key("_".join(tokens[-size:]))
            if not candidate or candidate in seen:
                continue
            if len(candidate) <= cls.SOURCE_RUNTIME_NAME_MAX_LENGTH:
                forms.append(candidate)
                seen.add(candidate)

        return forms

    @classmethod
    def _source_readable_names(cls, fields: list[GeneratedSchemaField]) -> dict[str, str]:
        """Assign compact, meaningful, collision-safe names derived from the source semantic path."""
        source_fields = [
            field for field in fields
            if str((field.provenance or {}).get("generated_from") or "").strip().lower() == "mongodb_json_source"
        ]
        if not source_fields:
            return {}

        paths: dict[str, list[str]] = {}
        options: dict[str, list[str]] = {}
        original_names: dict[str, str] = {}
        for field in source_fields:
            key = cls._normalize_variable_key(field.name)
            paths[key] = cls._source_semantic_segments(field)
            original_names[key] = field.name
            # Existing runtime names that already fit the contract are preserved exactly.
            # We only synthesize a shorter source-path name when the current name is too long.
            if len(field.name) <= cls.SOURCE_RUNTIME_NAME_MAX_LENGTH:
                options[key] = [field.name]
            else:
                options[key] = cls._source_candidate_forms(paths[key])

        occupied = {
            cls._normalize_variable_key(field.name)
            for field in fields
            if str((field.provenance or {}).get("generated_from") or "").strip().lower() != "mongodb_json_source"
        }
        assigned: dict[str, str] = {}
        used = set(occupied)

        # Preserve existing short names first. Long names are compacted only when necessary, and
        # their candidates are allocated from the most constrained fields outward.
        short_names = sorted(
            (key for key in options if len(original_names[key]) <= cls.SOURCE_RUNTIME_NAME_MAX_LENGTH),
            key=lambda key: (len(original_names[key]), key),
        )
        for key in short_names:
            original = original_names[key]
            normalized_original = cls._normalize_variable_key(original)
            if normalized_original not in used:
                assigned[key] = original
                used.add(normalized_original)
            else:
                # This should be rare because exact-name duplicates are normally resolved earlier.
                # Leave it for the contextual candidate pass rather than silently changing an
                # existing runtime name here.
                options[key] = cls._source_candidate_forms(paths[key])

        pending = sorted(
            (key for key in options if key not in assigned),
            key=lambda key: (len(options[key]) if options[key] else 999, len(options[key][0]) if options[key] else 999, key),
        )
        unresolved: list[str] = []
        for key in pending:
            chosen = next((candidate for candidate in options[key] if candidate not in used), None)
            if chosen:
                assigned[key] = chosen
                used.add(chosen)
            else:
                unresolved.append(key)

        # Extremely rare collision fallback. It never changes source words or truncates characters;
        # a short stable suffix disambiguates two paths whose readable context is otherwise identical.
        for key in unresolved:
            tokens = cls._source_candidate_tokens(paths[key])
            base_tokens = tokens[-4:] if len(tokens) >= 4 else tokens
            base = cls._normalize_variable_key("_".join(base_tokens)) or "scenario_attribute"
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]
            budget = cls.SOURCE_RUNTIME_NAME_MAX_LENGTH - len(digest) - 1
            parts = base.split("_")
            while parts and len("_".join(parts)) > budget:
                parts.pop(0)
            base = "_".join(parts) or "scenario_attribute"
            candidate = f"{base}_{digest}"
            assigned[key] = candidate
            used.add(candidate)

        return assigned

    @classmethod
    def _compact_source_field_names(cls, fields: list[GeneratedSchemaField]) -> list[GeneratedSchemaField]:
        """Apply compact source-derived names while retaining the full original JSON identity."""
        if not fields:
            return fields
        replacements = cls._source_readable_names(fields)
        if not replacements:
            return fields

        def redirect(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return replacements.get(cls._normalize_variable_key(value), value)

        def redirect_formula(formula: Any) -> Any:
            if not isinstance(formula, str) or not formula.strip():
                return formula
            result = formula
            for loser_key, winner in sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0])):
                result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(loser_key)}(?![A-Za-z0-9_])", winner, result)
            return result

        reference_keys = {
            "depends_on_field", "field", "segment_field", "hi_field", "lo_field",
            "base_field", "source_field", "add_seconds_field",
        }
        result: list[GeneratedSchemaField] = []
        for field in fields:
            rewritten = field.model_copy(deep=True)
            old_name = rewritten.name.strip()
            original_key = cls._normalize_variable_key(old_name)
            is_source = str((rewritten.provenance or {}).get("generated_from") or "").strip().lower() == "mongodb_json_source"
            rewritten.name = replacements.get(original_key, old_name)
            rewritten.depends_on = [redirect(dep) for dep in (rewritten.depends_on or [])]
            rewritten.formula = redirect_formula(rewritten.formula)
            params = dict(rewritten.params or {})
            for key in reference_keys:
                if key in params:
                    params[key] = redirect(params[key])
            rewritten.params = params
            if is_source:
                provenance = dict(rewritten.provenance or {})
                provenance.setdefault("source_json_name", old_name)
                provenance.setdefault("source_json_original_name", old_name)
                rewritten.provenance = provenance
            result.append(rewritten)
        return result

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

        # Source-backed fields are assigned their traceable JSON-path names in the final naming pass;
        # this helper resolves collisions for non-source semantic names retained for compatibility.
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
            if string_params.get("pattern"):
                return "pattern_string", "string", string_params
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
        excluded_field_names: list[str] | None = None,
        business_scenario: str | None = None,
        context_text: str | None = None,
        candidate_variables_override: list[dict[str, object]] | None = None,
    ) -> list[GeneratedSchemaField]:
        normalized_type = str(type_of_data or intent.type_of_data or "transactional").strip().lower()
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
        # Source-backed names remain at their canonical source identity while fields are assembled;
        # the final pass below assigns concise, traceable JSON-path names without altering provenance.
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
                "source_json_name": original_name if source_from_json else None,
                "source_json_original_name": original_name if source_from_json else None,
                "source_json_model": source_spec.get("model") if source_from_json else None,
                "source_json_path": source_spec.get("path") if source_from_json else None,
                "source_json_semantic_key": source_spec.get("semantic_key") if source_from_json else None,
                "source_json_paths": list(source_spec.get("source_paths") or []) if source_from_json else [],
                "source_json_aliases": list(source_spec.get("source_aliases") or []) if source_from_json else [],
                "source_owner_model": source_spec.get("source_owner_model") if source_from_json else None,
                "source_owner_kind": source_spec.get("source_owner_kind") if source_from_json else None,
                "source_owner_relation": source_spec.get("source_owner_relation") if source_from_json else None,
                "source_owner_relative_path": source_spec.get("source_owner_relative_path") if source_from_json else None,
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
                useCase=(str(idea.get("useCase") or "").strip() or str(intent.use_case or "").strip() or None),
                provenance=provenance,
                scope=grain,
            ))
        return self._compact_source_field_names(fields)

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
        external_keys = {
            normalize_lookup_key(name)
            for name in (external_variable_names or set())
            if normalize_lookup_key(name)
        }

        # Resolve the entity anchor before source selection so the quality policy can force it even
        # when its field would otherwise score below the analytical quality floor.
        role_identity_names = [
            normalize_lookup_key(row.get("name"))
            for row in catalog_rows
            if normalize_lookup_key(row.get("name"))
            and str(_catalog_role(row)).strip().lower() == "identity"
        ]
        all_identity_names = role_identity_names or [
            normalize_lookup_key(row.get("name"))
            for row in catalog_rows
            if normalize_lookup_key(row.get("name"))
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
                chosen_key = all_identity_names[0] if all_identity_names else None
                if chosen_key:
                    resolved_entity_key = next(
                        (str(row.get("name")) for row in catalog_rows if normalize_lookup_key(row.get("name")) == chosen_key),
                        chosen_key,
                    )
        if resolved_entity_key:
            preferred_names.add(normalize_lookup_key(resolved_entity_key))

        # Source relevance is intentionally recall-oriented. Give the quality policy a larger
        # candidate pool, then let deterministic scoring keep the best source-backed concepts up to
        # the requested budget. This preserves the variable ceiling while preferring useful fields
        # over filling the budget with low-information metadata.
        candidate_pool_budget = min(
            len(catalog_rows),
            max(variable_budget, variable_budget * 3, variable_budget + 50),
        ) if variable_budget else 0
        selected_rows, source_selection_report = select_json_source_catalog(
            catalog_rows,
            business_context=business_context,
            preferred_names=preferred_names,
            preferred_models={str(value) for value in (intent.requested_entities or []) if str(value).strip()},
            excluded_names=excluded,
            excluded_semantic_keys=external_semantic_aliases,
            max_fields=candidate_pool_budget,
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
            source_role = _catalog_role(spec)
            role = source_role if source_role in {
                "identity", "profile", "event", "transaction", "status", "measurement",
                "metric", "timing", "decision", "configuration", "derived", "categorical", "other",
            } else "other"
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
                "_force_include": bool(spec.get("required")) or (
                    normalized_type == "transactional"
                    and resolved_entity_key
                    and self._normalize_variable_key(str(spec.get("name") or "")) == self._normalize_variable_key(resolved_entity_key)
                ),
            })

        quality_engine = VariableQualityEngine(
            max_variables=variable_budget,
            min_score=SCHEMA_MIN_VARIABLE_SCORE,
        )
        quality_context = business_context or " ".join(
            str(value or "") for value in (
                intent.industry_type, intent.domain, intent.subdomain, intent.scenario_type,
                intent.type_of_data, intent.use_case, intent.entity_key, business_scenario or "",
            )
        )
        selected_source_ideas, quality_report = quality_engine.select(
            selected_source_ideas,
            context_text=quality_context,
            entity_key=resolved_entity_key,
        )
        selected_rows = [
            dict(idea.get("_json_source_spec") or {})
            for idea in selected_source_ideas
            if isinstance(idea.get("_json_source_spec"), dict)
        ]
        selected_names = {normalize_lookup_key(row.get("name")) for row in selected_rows}
        selection_report = {
            **source_selection_report,
            "source_relevance_selected_count": source_selection_report.get("selected_count", 0),
            "source_relevance_candidate_count": source_selection_report.get("candidate_count", 0),
            "selected_count": len(selected_rows),
            "quality_candidate_count": quality_report.get("candidate_count", len(selected_source_ideas)),
            "quality_selected_count": quality_report.get("selected_count", len(selected_rows)),
            "quality_duplicates_removed": quality_report.get("semantic_duplicates_removed", 0),
            "quality_low_score_removed": quality_report.get("low_quality_candidates_removed", 0),
            "quality_truncated": quality_report.get("quality_budget_truncated", 0),
        }

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
            excluded_field_names=excluded_field_names,
            business_scenario=business_scenario,
            context_text=business_context,
            candidate_variables_override=selected_source_ideas,
        )

        actual_names = {self._normalize_variable_key(field.name) for field in fields}
        actual_source_names = {
            self._normalize_variable_key(str((field.provenance or {}).get("source_json_name") or field.name))
            for field in fields
            if str((field.provenance or {}).get("generated_from") or "").strip().lower() == "mongodb_json_source"
        }
        missing_source_fields = sorted(selected_names - actual_source_names)
        if missing_source_fields:
            raise ValueError(
                "JSON-source variables were lost during deterministic compilation: " + ", ".join(missing_source_fields[:25])
            )
        if normalized_type == "transactional" and resolved_entity_key:
            resolved_key = normalize_lookup_key(resolved_entity_key)
            if resolved_key not in actual_names and resolved_key not in actual_source_names and resolved_key not in external_keys:
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
            "Full scalar source names and JSON paths remain in provenance; runtime names use concise unique JSON-path expressions without changing source identity.",
            "Standard enum values and declared numeric constraints from the source documents are authoritative.",
            "Arrays and complex objects are not emitted as fake flat scalar variables.",
        ]
        if resolved_entity_key:
            hard_constraints.append(f"'{resolved_entity_key}' is the authoritative entity key for transactional grouping.")
        warnings = [
            f"Source catalog selection: {selection_report.get('selected_count', 0)} final source variables retained after relevance and quality filtering from {selection_report.get('source_relevance_candidate_count', selection_report.get('candidate_count', 0))} catalog candidates.",
            "Adding or replacing industry/domain standard JSONs changes the available source catalog without requiring code changes.",
        ]
        # Reuse source metadata already loaded into the catalog; this avoids an extra MongoDB query on an uncached proposal.
        source_docs = list(catalog_payload.get("sources") or [])
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
