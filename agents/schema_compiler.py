"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship, VariableIdea
from core.telecom_registry import EntityDef, TelecomRegistry
from config.industry_profiles import get_profile
import re
from difflib import SequenceMatcher
from core.scenario_semantics import classify_outcome_mode
from core.json_domain_policy import LOW_BALANCE_MAIN_MODEL_IDS, LOW_BALANCE_SOURCE_IDS, expanded_scalar_catalog, is_json_grounded_domain
from core.low_balance_variable_policy import official_catalog_by_name, material_low_balance_catalog
from core.variable_quality import VariableQualityEngine
from config.runtime import SCHEMA_MAX_VARIABLES, SCHEMA_MIN_VARIABLE_SCORE


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1]


class SchemaCompiler:
    """Compiles scenario schemas from the approved telecom standards registry."""

    @staticmethod
    def _source_id(entity: EntityDef) -> str:
        return str(entity.canonical_id).split("__", 1)[0]

    @classmethod
    def _is_allowed_source(cls, entity: EntityDef, source_ids: set[str] | None) -> bool:
        return not source_ids or cls._source_id(entity) in source_ids

    # These fields are part of the public telecom row contract. Keep their exact
    # names even when the rest of the schema uses fresh
    # scenario-specific names. They are stable subscriber/account contact anchors.
    REQUIRED_TELECOM_FIELDS = ("subscriber_id", "account_id", "msisdn")
    REDUNDANT_MSISDN_FIELDS = {
        "phonenumber", "mobile_number", "mobilenumber", "telephone_number",
        "telephonenumber", "subscriber_phone", "subscriber_mobile",
    }
    UNSUPPORTED_NESTED_DTYPES = {"object", "array"}
    @classmethod
    def is_mandatory_telecom_field(cls, name: str | None) -> bool:
        normalized = cls._normalize_variable_name(name or "")
        return normalized in {cls._normalize_variable_name(item) for item in cls.REQUIRED_TELECOM_FIELDS}

    def __init__(self, registry: TelecomRegistry | None = None):
        self.registry = registry or TelecomRegistry()

    def resolve(
        self,
        intent: ScenarioIntent,
        domain_query: str | None = None,
        business_scenario: str | None = None,
        use_case: str | None = None,
        scenario_type: str | None = None,
        type_of_data: str | None = None,
        entity_key: str | None = None,
        source_ids: set[str] | None = None,
    ) -> tuple[list[EntityDef], list[str]]:
        """Resolve only concepts supported by the current request and optional source boundary."""
        resolved: list[EntityDef] = []
        unresolved: list[str] = []
        seen: set[str] = set()

        def add(entity: EntityDef | None, *, first: bool = False) -> None:
            if entity is None or not self._is_allowed_source(entity, source_ids) or entity.canonical_id in seen:
                return
            if first:
                resolved.insert(0, entity)
            else:
                resolved.append(entity)
            seen.add(entity.canonical_id)

        for candidate in self.registry.search(domain_query or intent.domain, limit=None):
            add(self.registry.resolve_entity(candidate["canonical_id"]))

        for requested in intent.requested_entities:
            add(self.registry.resolve_entity(requested))

        for idea in intent.candidate_variables:
            for owner in self.registry.entities_with_attribute(str(idea.name or "")):
                add(owner)

        if entity_key:
            key_entities = [
                e for e in self.registry.entities_with_attribute(entity_key)
                if self._is_allowed_source(e, source_ids)
            ]
            if key_entities:
                add(key_entities[0], first=True)
            elif not self.is_mandatory_telecom_field(entity_key):
                unresolved.append(f"Entity key '{entity_key}' is not a field in the approved telecom registry")

        normalized_use_case = (use_case or intent.use_case or intent.subdomain or "").strip().lower()
        if normalized_use_case and normalized_use_case != "unknown" and not source_ids:
            for candidate in self.registry.catalog_summary(domain=normalized_use_case, limit=None):
                add(self.registry.resolve_entity(candidate["canonical_id"]))

        if not source_ids and (intent.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            for anchor_id in ("subscriber", "customer_account", "customer"):
                if self.registry.entity_exists(anchor_id):
                    add(self.registry.get_entity(anchor_id))

        context_terms = " ".join(
            str(value or "") for value in (
                domain_query or intent.domain,
                business_scenario or "",
                use_case or intent.use_case,
                scenario_type or intent.scenario_type,
                *list(intent.requested_entities),
            )
        ).strip()
        if context_terms and not source_ids:
            for candidate in self.registry.search(context_terms, limit=None):
                add(self.registry.resolve_entity(candidate["canonical_id"]))

        if normalized_use_case == "prepaid" and not source_ids and self.registry.entity_exists("subscriber"):
            add(self.registry.get_entity("subscriber"))

        return resolved, unresolved

    def _runtime_generation_contract(
        self,
        attr: object,
        params: dict,
        country: str | None,
    ) -> tuple[str, dict]:
        """Translate standards/generation-profile generator names to executable runtime names."""
        name = str(getattr(attr, "name", ""))
        dtype = str(getattr(attr, "dtype", "string") or "string").lower()
        generator = str(getattr(attr, "generator", "") or "").strip().lower()
        runtime_params = dict(params or {})

        # Translate JSON Schema/OpenAPI validation keywords into the generator's
        # canonical parameter names without discarding the original constraints.
        if "minimum" in runtime_params and "min" not in runtime_params:
            runtime_params["min"] = runtime_params["minimum"]
        if "maximum" in runtime_params and "max" not in runtime_params:
            runtime_params["max"] = runtime_params["maximum"]
        if "minLength" in runtime_params and "min_length" not in runtime_params:
            runtime_params["min_length"] = runtime_params["minLength"]
        if "maxLength" in runtime_params and "max_length" not in runtime_params:
            runtime_params["max_length"] = runtime_params["maxLength"]

        if generator == "unique_id":
            return "prefixed_int", runtime_params
        if generator == "msisdn":
            iso = str(country or runtime_params.get("country") or "IN").strip().upper()
            dial_codes = {
                "IN": "+91", "US": "+1", "CA": "+1", "GB": "+44", "AU": "+61",
                "AE": "+971", "SG": "+65", "DE": "+49", "FR": "+33", "IT": "+39",
            }
            dial = dial_codes.get(iso, iso if iso.startswith("+") else "+" + iso)
            runtime_params["country_codes"] = [dial]
            runtime_params["country"] = iso
            return "e164_phone", runtime_params
        if generator == "timestamp":
            runtime_params.setdefault("days_back", 365)
            return "recent_datetime", runtime_params
        if generator == "range":
            return ("uniform_int" if dtype in {"int", "integer"} else "uniform"), runtime_params
        if generator == "reference":
            target = str(runtime_params.get("target") or "").strip()
            target_entity = self.registry.resolve_entity(target) if target else None
            target_attr = None
            if target_entity is not None:
                preferred_name = f"{target_entity.canonical_id}_id".lower()
                target_attr = next((item for item in target_entity.attributes if item.name.lower() == preferred_name), None)
                if target_attr is None:
                    target_attr = next((item for item in target_entity.attributes if item.name.lower().endswith("_id")), None)
            if target_attr is not None:
                target_params = dict(target_attr.params or {})
                if str(target_attr.generator or "").strip().lower() == "unique_id":
                    return "prefixed_int", target_params
                if target_attr.generator:
                    generator_name, translated = self._runtime_generation_contract(target_attr, target_params, country)
                    return generator_name, translated
            # A reference in a flattened dataset may not have a separate target row.
            # Generate a structurally valid target-style identifier rather than returning
            # an unsupported runtime generator or an empty value.
            if target:
                clean_target = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_").upper()
                runtime_params.setdefault("prefix", f"{clean_target or 'REF'}-")
            runtime_params.setdefault("digits", 10)
            return "prefixed_int", runtime_params
        if generator == "dependent_choice":
            if not runtime_params.get("mapping"):
                raise ValueError(f"Dependent choice field '{name}' is missing its mapping")
            return "dependent_choice", runtime_params
        # `generic` in the standards-derived profile means "no executable generator was
        # declared"; it must never reach the runtime because the old generic path emitted
        # field-name placeholders. Resolve it using the attribute's concrete dtype/semantics.
        if generator in {"generic", "string", "text"}:
            generator = ""
        # Flat synthetic records cannot safely materialize arbitrary nested JSON objects/arrays.
        if dtype in {"object", "array"}:
            return None, runtime_params
        if not generator:
            enum_values = tuple(getattr(attr, "enum_values", ()) or ())
            if enum_values:
                runtime_params = {"choices": list(enum_values), "weights": [1.0] * len(enum_values)}
                return "weighted_choice", runtime_params
            if name.lower().endswith("_id"):
                return "prefixed_int", {"prefix": f"{name[:-3].upper()}-", "digits": 10}
            if dtype == "datetime":
                runtime_params.setdefault("days_back", 365)
                return "recent_datetime", runtime_params
            if dtype in {"float", "decimal", "number", "numeric"}:
                runtime_params.setdefault("min", 0)
                runtime_params.setdefault("max", 100)
                runtime_params.setdefault("precision", 2)
                return "uniform", runtime_params
            if dtype in {"int", "integer"}:
                runtime_params.setdefault("min", 0)
                runtime_params.setdefault("max", 100)
                return "uniform_int", runtime_params
            return "semantic_string", runtime_params
        return generator, runtime_params

    def _expansion_score(
        self,
        entity: EntityDef,
        *,
        scenario_type: str | None,
        type_of_data: str | None,
        use_case: str | None,
    ) -> int:
        """Rank graph-expansion entities using backend-authoritative proposal inputs."""
        text = " ".join([entity.canonical_id, entity.name, entity.domain, entity.description, *entity.aliases]).lower()
        score = 0
        scenario_tokens = set(_tokens(scenario_type or ""))
        use_case_tokens = set(_tokens(use_case or ""))
        score += 8 * sum(1 for token in scenario_tokens if token in text)
        score += 10 * sum(1 for token in use_case_tokens if token in text)
        if use_case and entity.domain.lower() == use_case.strip().lower():
            score += 25
        if type_of_data == "transactional":
            score += 4 if any(a.name.endswith("_id") for a in entity.attributes) else 0
            score += 4 if any("timestamp" in a.name for a in entity.attributes) else 0
        elif type_of_data == "aggregational":
            score += 4 if any(a.dtype in {"float", "integer", "decimal"} for a in entity.attributes) else 0
            score += 2 if not any(a.name.endswith("_id") for a in entity.attributes) else 0
        return score

    @staticmethod
    def _normalize_variable_name(name: str) -> str:
        text = str(name or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = "scenario_attribute"
        if text[0].isdigit():
            text = f"feature_{text}"
        return text[:120]

    @staticmethod
    def _idea_tokens(idea: dict[str, object]) -> set[str]:
        return set(_tokens(" ".join(str(idea.get(k, "")) for k in ("name", "description", "role", "grain"))))

    def _choose_entity_for_idea(
        self,
        idea: dict[str, object],
        entities: list[EntityDef],
        matched_attr: tuple[EntityDef, object] | None,
        entity_key: str | None,
    ) -> EntityDef | None:
        if matched_attr is not None:
            return matched_attr[0]
        tokens = self._idea_tokens(idea)
        role = str(idea.get("role") or "other").lower()
        grain = str(idea.get("grain") or "transaction").lower()
        best: tuple[int, str, EntityDef] | None = None
        for entity in entities:
            text = " ".join([entity.canonical_id, entity.name, entity.description, *entity.aliases]).lower()
            score = sum(3 for token in tokens if token in text)
            if entity_key and entity_key.lower() in {a.name.lower() for a in entity.attributes}:
                score += 20 if grain == "entity" else 4
            if grain in {"event", "transaction"} and any("event" in a.name.lower() or "transaction" in a.description.lower() for a in entity.attributes):
                score += 5
            if role in {"status", "decision"} and any("status" in a.name.lower() or "decision" in a.description.lower() for a in entity.attributes):
                score += 4
            candidate = (score, entity.canonical_id, entity)
            if score > 0 and (best is None or score > best[0] or (score == best[0] and entity.canonical_id < best[1])):
                best = candidate
        if best:
            return best[2]
        return entities[0] if entities else None

    def _match_registry_attribute(
        self,
        idea: dict[str, object],
        entities: list[EntityDef],
        used_registry: set[tuple[str, str]],
    ) -> tuple[EntityDef, object] | None:
        """Find an approved registry attribute with similar semantics without replaying its name."""
        idea_name = str(idea.get("name") or "").strip().lower()
        idea_text = " ".join(str(idea.get(k, "")) for k in ("name", "description"))
        idea_tokens = self._idea_tokens(idea)
        role = str(idea.get("role") or "").lower()
        normalized_idea_name = self._normalize_variable_name(idea_name)
        candidates: list[tuple[float, str, str, EntityDef, object]] = []
        for entity in entities:
            entity_text = " ".join([entity.canonical_id, entity.name, entity.description, *entity.aliases]).lower()
            entity_tokens = set(_tokens(entity_text))
            entity_overlap = len(idea_tokens & entity_tokens)
            for attr in entity.attributes:
                key = (entity.canonical_id, attr.name)
                if key in used_registry:
                    continue
                attr_dtype = str(getattr(attr, "dtype", "")).strip().lower()
                requested_dtype = str(idea.get("dtype") or "string").strip().lower()
                compatible = (
                    (requested_dtype in {"float", "decimal"} and attr_dtype in {"float", "decimal", "number", "numeric"})
                    or (requested_dtype == "integer" and attr_dtype in {"int", "integer", "bigint", "smallint"})
                    or (requested_dtype == "datetime" and attr_dtype in {"datetime", "timestamp", "date"})
                    or (requested_dtype == "date" and attr_dtype in {"date", "datetime", "timestamp"})
                    or (requested_dtype == "boolean" and attr_dtype in {"bool", "boolean"})
                    or (requested_dtype == "categorical" and (bool(getattr(attr, "enum_values", ())) or str(getattr(attr, "generator", "")).lower() in {"dependent_choice", "categorical", "weighted_choice"}))
                    or (requested_dtype == "string" and attr_dtype in {"string", "str", "object", "uuid", "varchar", "text"})
                )
                if not compatible:
                    continue
                if role in {"measurement", "metric"} and attr_dtype not in {"float", "decimal", "number", "numeric", "int", "integer"}:
                    continue
                if role == "timing" and attr_dtype not in {"datetime", "timestamp", "date"}:
                    continue
                attr_text = " ".join([attr.name, attr.description]).lower()
                attr_tokens = set(_tokens(attr_text))
                attr_overlap = len(idea_tokens & attr_tokens)
                exact_name_bonus = 8.0 if self._normalize_variable_name(attr.name) == normalized_idea_name else 0.0
                ratio = SequenceMatcher(None, idea_text.lower(), attr_text).ratio()
                role_bonus = 3.0 if (
                    (role == "timing" and attr.dtype.lower() in {"datetime", "date"})
                    or (role in {"measurement", "metric"} and attr.dtype.lower() in {"float", "integer", "decimal"})
                    or (role in {"status", "decision"} and ("status" in attr.name.lower() or "status" in attr.description.lower()))
                    or (role == "identity" and attr.name.lower().endswith("_id"))
                ) else 0.0
                # Entity context is deliberately strong for colliding attribute names such as
                # subscriber_id, account_id and usage_event_id. Matching the field concept to the
                # right registry entity is more important than a generic lexical match on the role.
                score = exact_name_bonus + entity_overlap * 4.0 + attr_overlap * 2.5 + ratio * 2.0 + role_bonus
                exact = self._normalize_variable_name(attr.name) == normalized_idea_name
                strong_semantic = (
                    attr_overlap >= 2
                    or (attr_overlap >= 1 and ratio >= 0.50 and entity_overlap >= 1)
                )
                # Never satisfy an explicit business role with a merely related technical
                # attribute. In particular, outcome/decision/status ideas must resolve to
                # status/decision/result/reason concepts rather than usage/unit/reference fields.
                role_text = f"{attr.name} {attr.description}".lower()
                role_specific_ok = True
                if role in {"status", "decision"} or any(token in idea_name for token in ("outcome", "decision", "result")):
                    role_specific_ok = any(
                        token in role_text
                        for token in ("status", "state", "outcome", "decision", "result", "reason", "eligible", "suppression")
                    )
                elif role in {"measurement", "metric"}:
                    role_specific_ok = any(
                        token in role_text
                        for token in ("amount", "balance", "quantity", "value", "volume", "measure", "rate", "percentage")
                    ) or attr_dtype in {"float", "decimal", "number", "numeric", "int", "integer"}
                if exact or (strong_semantic and score >= 8.0 and role_specific_ok):
                    candidates.append((score, entity.canonical_id, attr.name, entity, attr))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        return candidates[0][3], candidates[0][4]

    def _fresh_name(
        self,
        requested: str,
        description: str,
        used: set[str],
        registry_names: set[str],
        entity_key: str | None,
    ) -> str:
        """Preserve semantically meaningful requested names; only disambiguate collisions."""
        base = self._normalize_variable_name(requested)
        if base and base not in used:
            return base

        # Only the schema-local collision requires renaming. Registry vocabulary itself is
        # not a collision: a standards-backed concept such as recharge_timestamp or status
        # should keep its public, recognizable field name.
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
            return "recent_datetime", "date", {
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
    def _json_source_contract(spec: dict[str, object], country: str | None) -> tuple[str, str, dict]:
        """Create an executable generator contract from a flattened Swagger scalar leaf."""
        dtype = str(spec.get("dtype") or "string").strip().lower()
        fmt = str(spec.get("format") or "").strip().lower()
        enum_values = list(spec.get("enum_values") or [])
        params: dict = {}

        if enum_values:
            return "weighted_choice", "categorical", {"choices": enum_values, "weights": [1.0] * len(enum_values)}

        # Low Balance has an explicit public customer identity contract. The source field is
        # still TMF629 Customer.id; only the deterministic synthetic representation is fixed here.
        if SchemaCompiler._normalize_variable_name(str(spec.get("name") or "")) == "customer_id":
            return "prefixed_int", "string", {"prefix": "cust-", "digits": 8}

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
        if fmt in {"date-time", "datetime", "timestamp"} or dtype in {"date-time", "datetime", "timestamp"}:
            return "recent_datetime", "datetime", {
                "timezone": "Asia/Kolkata" if str(country or "IN").upper() == "IN" else "UTC",
                "days_back": 365,
            }
        if dtype == "date":
            return "recent_datetime", "date", {
                "timezone": "Asia/Kolkata" if str(country or "IN").upper() == "IN" else "UTC",
                "days_back": 365,
            }
        if dtype in {"integer", "int", "bigint", "smallint"}:
            return "uniform_int", "integer", {"min": 0, "max": 100, "precision": 0}
        if dtype in {"number", "float", "double", "decimal", "numeric"}:
            return "uniform", "float", {"min": 0.0, "max": 1000.0, "precision": 2}
        # Identifier/reference/URI semantics are handled by semantic_string, which already
        # creates deterministic synthetic identifiers and example.test URIs safely.
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
        entities: list[EntityDef],
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
        excluded_keys = {
            self._normalize_variable_name(name)
            for name in (excluded_field_names or [])
            if self._normalize_variable_name(name)
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
            key = self._normalize_variable_name(str(item.get("name") or ""))
            if not key or key in seen_idea_keys:
                return
            seen_idea_keys.add(key)
            ideas.append(item)

        # Mandatory telecom anchors first so they keep their exact public names and stable
        # entity grain. When the official registry does not expose a literal ``subscriber``
        # entity, the compiler still emits these application-contract fields deterministically.
        if include_application_telecom_anchors and (intent.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            mandatory_defs = {
                "subscriber_id": ("subscriber", "subscriber_id", "Stable subscriber identifier."),
                "account_id": ("subscriber", "account_id", "Stable subscriber account identifier."),
                "msisdn": ("subscriber", "msisdn", "Subscriber MSISDN / mobile telephone number."),
            }
            for field_name in self.REQUIRED_TELECOM_FIELDS:
                if self._normalize_variable_name(field_name) in excluded_keys:
                    continue
                _, _, desc = mandatory_defs[field_name]
                add_idea({
                    "name": field_name,
                    "description": desc,
                    "role": "identity" if field_name.endswith("_id") else "profile",
                    "grain": "entity",
                    "dtype": "string",
                    "depends_on": [],
                })

        if include_all_json_source_scalars:
            # Low Balance & Top-up uses the supplied TMF654/TMF629 Swagger files as the
            # complete standards source. Include every safe scalar leaf, including scalar
            # properties inside referenced objects, while keeping arrays out of the flat row.
            for spec in expanded_scalar_catalog():
                model = str(spec.get("model") or "")
                field_key = self._normalize_variable_name(str(spec.get("name") or ""))
                dtype = str(spec.get("dtype") or "string").lower()
                if dtype in self.UNSUPPORTED_NESTED_DTYPES or field_key in excluded_keys:
                    continue
                add_idea({
                    "name": str(spec["name"]),
                    "description": str(spec.get("description") or f"{model} {spec['path']} from the supplied official Swagger model."),
                    "role": (
                        "timing" if str(spec.get("format") or "").lower() == "date-time" or dtype in {"date", "datetime", "timestamp"}
                        else "measurement" if dtype in {"integer", "number", "float", "double", "decimal"}
                        else "status" if str(spec.get("enum_values") or []) and any(token in str(spec.get("path") or "").lower() for token in ("status", "state", "reason"))
                        else "identity" if str(spec.get("path") or "").lower().endswith(".id")
                        else "other"
                    ),
                    "grain": (
                        "transaction"
                        if normalized_type == "transactional"
                        and str(spec.get("model_grain") or "transaction") == "entity"
                        and any(token in str(spec.get("path") or "").lower() for token in (
                            "remainingvalue.amount", "reservedvalue.amount",
                            "requesteddatetime", "confirmationdatetime",
                            "requesteddate", "confirmationdate",
                        ))
                        else str(spec.get("model_grain") or "transaction")
                    ),
                    "dtype": (
                        "integer" if dtype in {"integer", "int"}
                        else "float" if dtype in {"number", "float", "double", "decimal"}
                        else "datetime" if dtype in {"date-time", "datetime", "timestamp"}
                        else "date" if dtype == "date"
                        else "categorical" if spec.get("enum_values")
                        else "boolean" if dtype in {"boolean", "bool"}
                        else "string"
                    ),
                    "depends_on": [],
                    "_json_source_spec": spec,
                    "_json_source": True,
                    "_registry_entity": f"{spec.get('source_id', '')}__{spec.get('model', '')}",
                    "_registry_entity_name": str(spec.get("model") or ""),
                    "_registry_required": bool(spec.get("required")),
                    "_registry_nullable": not bool(spec.get("required")),
                }, preserve_name=True)

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
            self._normalize_variable_name(str(item.get("name") or "")) == "subscriber_id"
            for item in ideas
        )
        explicit_customer_entity_key = self._normalize_variable_name(entity_key or "") == "customer_id"
        if has_subscriber_anchor and not explicit_customer_entity_key:
            ideas = [
                item for item in ideas
                if self._normalize_variable_name(str(item.get("name") or "")) != "customer_id"
            ]

        if include_all_registry_scalars:
            # Generic registry-grounded domains retain broad scalar coverage. Domain-specific
            # grounded flows can disable this and compile only the scenario's selected ideas.
            for entity in entities:
                entity_grain = (
                    "entity"
                    if entity.canonical_id in {"subscriber", "customer", "customer_account", "prepaid_account"}
                    else "transaction"
                )
                for attr in entity.attributes:
                    attr_name = str(attr.name or "")
                    attr_key = self._normalize_variable_name(attr_name)
                    if not attr_key:
                        continue
                    # subscriber_id is the canonical telecom subscriber identity. Do not add
                    # a second customer_id column to a subscriber-anchored proposal unless the
                    # caller explicitly selected customer_id as the transactional entity key.
                    if attr_key in excluded_keys:
                        continue
                    if (
                        has_subscriber_anchor
                        and not explicit_customer_entity_key
                        and attr_key == "customer_id"
                    ):
                        continue
                    if attr_key in self.REDUNDANT_MSISDN_FIELDS and any(
                        self._normalize_variable_name(str(item.get("name") or "")) == "msisdn"
                        for item in ideas
                    ):
                        continue
                    raw_dtype = str(attr.dtype or "string").lower()
                    if raw_dtype in self.UNSUPPORTED_NESTED_DTYPES:
                        continue
                    add_idea({
                        "name": attr_name,
                        "description": attr.description or f"{entity.name} {attr_name} attribute.",
                        "role": (
                            "timing" if raw_dtype in {"datetime", "timestamp", "date"}
                            else "measurement" if raw_dtype in {"float", "decimal", "number", "numeric", "int", "integer"}
                            else "status" if "status" in attr_name.lower() or "state" in attr_name.lower()
                            else "identity" if attr_name.lower().endswith("_id")
                            else "other"
                        ),
                        "grain": entity_grain,
                        "dtype": (
                            "integer" if raw_dtype in {"int", "integer", "bigint", "smallint"} else
                            "float" if raw_dtype in {"float", "decimal", "number", "numeric"} else
                            "datetime" if raw_dtype in {"datetime", "timestamp"} else
                            "date" if raw_dtype == "date" else
                            "categorical" if attr.enum_values or str(attr.generator).lower() in {"weighted_choice", "dependent_choice", "categorical"} else
                            "boolean" if raw_dtype in {"bool", "boolean"} else "string"
                        ),
                        "depends_on": list(attr.depends_on),
                        "_registry_entity": entity.canonical_id,
                        "_registry_entity_name": entity.name,
                        "_registry_required": bool(attr.required),
                        "_registry_nullable": bool(attr.nullable),
                        "_registry_exact": True,
                    }, preserve_name=True)

        # Explicit telecom de-duplication before field construction.
        if any(self._normalize_variable_name(str(item.get("name") or "")) == "msisdn" for item in ideas):
            ideas = [
                item for item in ideas
                if self._normalize_variable_name(str(item.get("name") or "")) not in self.REDUNDANT_MSISDN_FIELDS
            ]

        # Quality-gate the broad candidate pool before converting it into executable fields.
        # ``max_variables`` remains an optional caller override; otherwise the environment
        # default applies. The selector removes semantic duplicates and low-information API
        # metadata without dropping mandatory/application contracts.
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
        ideas, quality_report = quality_engine.select(
            ideas,
            context_text=selection_context,
            entity_key=entity_key,
        )

        # Defensive post-selection guard for the same redundancy rule. This prevents any
        # future quality-engine dependency closure change from reintroducing customer_id into
        # a subscriber-anchored telecom proposal.
        if has_subscriber_anchor and not explicit_customer_entity_key:
            ideas = [
                item for item in ideas
                if self._normalize_variable_name(str(item.get("name") or "")) != "customer_id"
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
        used_registry: set[tuple[str, str]] = set()
        name_map: dict[str, str] = {}
        registry_names = {
            self._normalize_variable_name(attr.name)
            for entity in entities
            for attr in entity.attributes
        }

        # First assign names so later dependency references can be resolved to the fresh names.
        for idea in ideas:
            current_requested = str(idea.get("name") or "scenario_attribute")
            if current_requested in self.REQUIRED_TELECOM_FIELDS:
                fresh = current_requested
            elif bool(idea.get("_preserve_name")):
                fresh = current_requested
                if fresh in used_names:
                    owner = str(idea.get("_registry_entity") or "registry").strip().lower().replace(" ", "_")
                    scoped = f"{owner}_{current_requested}"
                    fresh = scoped if scoped not in used_names else self._fresh_name(scoped, str(idea.get("description") or ""), used_names, set(), entity_key)
            else:
                fresh = self._fresh_name(
                    current_requested,
                    str(idea.get("description") or ""),
                    used_names,
                    registry_names,
                    entity_key,
                )
            used_names.add(fresh)
            name_map[self._normalize_variable_name(str(idea.get("name") or ""))] = fresh

        profile = get_profile((intent.industry_type or "telecom"), country)
        country_currency = str(profile.get("currency") or "").upper()
        for idea in ideas:
            original_name = str(idea.get("name") or "scenario_attribute")
            fresh = name_map[self._normalize_variable_name(original_name)]
            matched = None
            attr = None
            source_spec = idea.get("_json_source_spec") if isinstance(idea.get("_json_source_spec"), dict) else None
            if source_spec is not None:
                source_model = str(source_spec.get("model") or "")
                entity = next((e for e in entities if e.name == source_model), None)
                runtime_generator, dtype, params = self._json_source_contract(source_spec, country)
            # Mandatory public telecom anchors use authoritative subscriber attributes
            # directly, rather than allowing a generic account_id match to resolve to a
            # different entity.
            if original_name in self.REQUIRED_TELECOM_FIELDS:
                subscriber = next((e for e in entities if e.canonical_id == "subscriber"), None)
                if subscriber is not None:
                    attr = next((a for a in subscriber.attributes if a.name == original_name), None)
                    if attr is not None:
                        matched = (subscriber, attr)
            normalized_original_name = self._normalize_variable_name(original_name)
            if source_spec is None:
                matched = self._match_registry_attribute(idea, entities, used_registry)
            role = str(idea.get("role") or "other").lower()
            idea_text = f"{original_name} {idea.get('description', '')}".lower()
            outcome_semantic = role in {"status", "decision"} or any(token in idea_text for token in ("status", "state", "outcome", "decision", "result"))
            if source_spec is not None:
                # Contract was already compiled from the authoritative Swagger leaf above.
                pass
            elif original_name in self.REQUIRED_TELECOM_FIELDS:
                entity = next((e for e in entities if e.canonical_id == "subscriber"), None)
                if original_name == "subscriber_id":
                    runtime_generator, dtype, params = (
                        "prefixed_int", "string", {"prefix": "SUB-", "digits": 10}
                    )
                elif original_name == "account_id":
                    runtime_generator, dtype, params = (
                        "id_mirror", "string",
                        {"prefix": "ACC-", "source_field": "subscriber_id", "source_prefix": "SUB-"}
                    )
                else:
                    iso = str(country or "IN").strip().upper()
                    dial_codes = {
                        "IN": "+91", "US": "+1", "CA": "+1", "GB": "+44", "AU": "+61",
                        "AE": "+971", "SG": "+65", "DE": "+49", "FR": "+33", "IT": "+39",
                    }
                    dial = dial_codes.get(iso, iso if iso.startswith("+") else "+" + iso)
                    runtime_generator, dtype, params = (
                        "e164_phone", "string", {"country_codes": [dial], "country": iso}
                    )
            elif matched is not None:
                entity, attr = matched
                used_registry.add((entity.canonical_id, attr.name))
                exact_registry_name = (
                    self._normalize_variable_name(attr.name)
                    == self._normalize_variable_name(original_name)
                )
                if exact_registry_name:
                    params = dict(attr.params or {})
                    if attr.generator == "msisdn":
                        params["country"] = str(country or params.get("country") or "IN").upper()
                    if country_currency and (attr.name.lower() == "currency" or "currency" in params):
                        params["currency"] = country_currency
                    translated = self._runtime_generation_contract(attr, params, country)
                    if translated is None:
                        runtime_generator, dtype, params = self._generic_contract_for_idea(idea, country, scenario_mode) or (None, None, None)
                    else:
                        runtime_generator, params = translated
                        dtype = attr.dtype
                        if attr.generator == "msisdn":
                            dtype = "string"
                elif getattr(attr, "enum_values", ()):
                    # A semantically renamed variable may still map to an official enum field
                    # (for example topup_status -> TopupBalance.status). Preserve the Swagger
                    # enum exactly; never substitute scenario-mode values for a standard enum.
                    choices = list(getattr(attr, "enum_values", ()) or ())
                    entity = entity or self._choose_entity_for_idea(idea, entities, matched, entity_key)
                    runtime_generator, dtype, params = (
                        "weighted_choice",
                        "categorical",
                        {"choices": choices, "weights": [1.0] * len(choices)},
                    )
                else:
                    entity = entity or self._choose_entity_for_idea(idea, entities, None, entity_key)
                    runtime_generator, dtype, params = self._generic_contract_for_idea(idea, country, scenario_mode) or (None, None, None)
            else:
                entity = self._choose_entity_for_idea(idea, entities, None, entity_key)
                runtime_generator, dtype, params = self._generic_contract_for_idea(idea, country, scenario_mode) or (None, None, None)

            if not runtime_generator or not dtype or not isinstance(params, dict):
                # Fail closed: a proposal field must have a concrete deterministic generator.
                continue
            # A field is executable only when the compiler has a concrete generator.
            # This is a second safety gate for registry attributes whose official model is
            # structurally richer than the flat synthetic CSV contract.
            if runtime_generator is None:
                continue

            role = str(idea.get("role") or "other")
            grain = str(idea.get("grain") or ("entity" if original_name == entity_key else "transaction"))
            raw_dependencies = [str(dep) for dep in (idea.get("depends_on") or []) if str(dep).strip()]
            unresolved_dependencies = [
                dep for dep in raw_dependencies
                if self._normalize_variable_name(dep) not in name_map
                and self._normalize_variable_name(dependency_aliases.get(self._normalize_variable_name(dep), dep)) not in name_map
            ]
            if unresolved_dependencies:
                # A selected field with an unrepresentable prerequisite cannot produce a faithful
                # executable contract. Drop it rather than silently weakening its dependency chain.
                continue
            deps = self._merge_dependencies(idea, name_map, grain, entity_key)
            if original_name == entity_key or original_name in self.REQUIRED_TELECOM_FIELDS:
                grain = "entity"
                deps = []
            registry_required = bool(idea.get("_registry_required")) if idea.get("_registry_required") is not None else False
            registry_nullable = bool(idea.get("_registry_nullable")) if idea.get("_registry_nullable") is not None else True
            required = original_name == entity_key or original_name in self.REQUIRED_TELECOM_FIELDS or registry_required
            nullable = False if required else registry_nullable
            description = str(idea.get("description") or "").strip() or f"Scenario-specific {role.replace('_', ' ')} attribute for {intent.domain}."
            source_from_json = isinstance(source_spec, dict)
            provenance = {
                "generated_from": "official_json_source" if source_from_json else ("registry_attribute" if idea.get("_preserve_name") else "semantic_variable_idea"),
                "canonical_entity": entity.canonical_id if entity else None,
                "source_registry_attribute": getattr(attr, "name", None) if matched and not source_from_json else None,
                "source_json_id": source_spec.get("source_id") if source_from_json else None,
                "source_json_model": source_spec.get("model") if source_from_json else None,
                "source_json_path": source_spec.get("path") if source_from_json else None,
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
                # Registry formulas reference canonical registry field names. Because this
                # proposal deliberately renames matched concepts into fresh names, do not carry
                # an un-translated formula into the new schema. Derived formulas are synthesized
                # later only when their dependencies are explicitly present in the fresh idea set.
                formula=None,
                provenance=provenance,
                scope=grain,
            ))
        return fields

    def _compile_low_balance_json_grounded(
        self,
        intent: ScenarioIntent,
        *,
        business_scenario: str,
        use_case: str,
        country: str | None,
        type_of_data: str | None,
        entity_key: str | None,
        scenario_type: str | None = None,
        excluded_field_names: list[str] | None = None,
    ) -> ScenarioSchema:
        """Compile Low Balance & Top-up from TMF654 + TMF629 Swagger-backed registry concepts."""
        normalized_country = str(country or "IN").strip().upper()
        normalized_type = str(type_of_data or intent.type_of_data or "transactional").strip().lower()
        normalized_scenario = str(scenario_type or intent.scenario_type or "").strip()
        scenario_mode = classify_outcome_mode(
            scenario_type=normalized_scenario,
            expected_outcome="",
            business_response="",
            business_scenario=business_scenario,
        )

        # Gemini is a selector/reviewer only for this source-bound domain. Its selected names are
        # used to prioritize ordering, while the deterministic breadth guard adds every material
        # official scalar leaf from the two supplied Swagger files. This prevents the final width
        # from depending on how many fields Gemini happened to mention while keeping the source
        # boundary strict.
        catalog = official_catalog_by_name()
        llm_selected_names = [
            self._normalize_variable_name(idea.name)
            for idea in intent.candidate_variables
            if self._normalize_variable_name(idea.name) in catalog
        ]
        ordered_names: list[str] = []
        seen_names: set[str] = set()
        for name in llm_selected_names + [
            self._normalize_variable_name(row.get("name"))
            for row in material_low_balance_catalog()
        ]:
            if name and name in catalog and name not in seen_names:
                seen_names.add(name)
                ordered_names.append(name)

        selected_source_ideas: list[dict[str, object]] = []
        clean_selected_ideas: list[VariableIdea] = []
        for name in ordered_names:
            spec = catalog[name]
            clean_idea = VariableIdea.model_validate({
                "name": str(spec["name"]),
                "description": str(spec.get("description") or "")[:500],
                "role": (
                    "timing" if str(spec.get("format") or "").lower() == "date-time" or str(spec.get("dtype") or "").lower() in {"date", "datetime", "timestamp"}
                    else "measurement" if str(spec.get("dtype") or "").lower() in {"integer", "number", "float", "double", "decimal"}
                    else "status" if spec.get("enum_values") and any(token in str(spec.get("path") or "").lower() for token in ("status", "state", "reason"))
                    else "identity" if str(spec.get("path") or "").lower().endswith(".id")
                    else "configuration" if any(token in str(spec.get("path") or "").lower() for token in ("isautotopup", "recurringperiod", "numberofperiods"))
                    else "profile" if str(spec.get("model") or "").lower() == "customer"
                    else "other"
                ),
                "grain": (
                    "entity" if str(spec.get("model") or "").lower() == "customer"
                    else "transaction"
                ),
                "dtype": (
                    "integer" if str(spec.get("dtype") or "").lower() in {"integer", "int"}
                    else "float" if str(spec.get("dtype") or "").lower() in {"number", "float", "double", "decimal"}
                    else "datetime" if str(spec.get("format") or "").lower() in {"date-time", "datetime", "timestamp"} or str(spec.get("dtype") or "").lower() in {"date-time", "datetime", "timestamp"}
                    else "date" if str(spec.get("dtype") or "").lower() == "date"
                    else "categorical" if spec.get("enum_values")
                    else "boolean" if str(spec.get("dtype") or "").lower() in {"boolean", "bool"}
                    else "string"
                ),
                "depends_on": [],
            })
            clean_selected_ideas.append(clean_idea)
            selected_source_ideas.append({
                **clean_idea.model_dump(),
                "name": str(spec["name"]),
                "description": str(spec.get("description") or "")[:500],
                "_json_source_spec": dict(spec),
                "_json_source": True,
                "_preserve_name": True,
                "_force_include": True,
                "_registry_entity": f"{spec.get('source_id', '')}__{spec.get('model', '')}",
                "_registry_entity_name": str(spec.get("model") or ""),
                "_registry_required": bool(spec.get("required")),
                "_registry_nullable": not bool(spec.get("required")),
            })

        # Persist only the clean semantic selection on the intent. Source metadata is passed
        # separately to _build_fresh_fields and never enters VariableIdea.
        intent = intent.model_copy(update={"candidate_variables": clean_selected_ideas})

        # Low Balance is intentionally compiled from the three primary resources only.
        # Create/Update/Event/Ref schemas describe API transport shapes, not the business
        # entities we want as flat synthetic-data columns. Restricting the entity pool here
        # also prevents the same attribute name from matching a transport model instead of the
        # canonical Bucket/TopupBalance/Customer model.
        entities: list[EntityDef] = []
        unresolved: list[str] = []
        for canonical_id in LOW_BALANCE_MAIN_MODEL_IDS:
            try:
                ent = self.registry.get_entity(canonical_id)
            except KeyError:
                ent = None
            if ent is not None:
                entities.append(ent)

        if not entities:
            unresolved.append("The Low Balance & Top-up Swagger sources did not yield the expected Bucket, TopupBalance, or Customer models.")

        fields = self._build_fresh_fields(
            intent,
            entities,
            entity_key=None,
            country=normalized_country,
            type_of_data=normalized_type,
            scenario_mode=scenario_mode,
            max_variables=max(len(selected_source_ideas), SCHEMA_MAX_VARIABLES),
            include_all_registry_scalars=False,
            include_all_json_source_scalars=False,
            include_application_telecom_anchors=False,
            excluded_field_names=excluded_field_names,
            candidate_variables_override=selected_source_ideas,
            business_scenario=business_scenario,
            context_text=" ".join(
                str(value or "") for value in (
                    intent.industry_type, intent.domain, intent.subdomain, normalized_scenario,
                    normalized_type, country or "", use_case or "", business_scenario,
                )
            ),
        )
        field_names = [f.name for f in fields]
        if not fields:
            unresolved.append("The Low Balance & Top-up scenario did not yield any usable semantic variables.")
        if (
            entity_key
            and self._normalize_variable_name(entity_key) not in {self._normalize_variable_name(n) for n in field_names}
            and not self.is_mandatory_telecom_field(entity_key)
        ):
            unresolved.append(f"Requested entity key '{entity_key}' could not be represented by the proposed variables")

        entity_ids = {e.canonical_id for e in entities}
        relationships: list[SchemaRelationship] = []
        for entity in entities:
            if self._source_id(entity) not in set(LOW_BALANCE_SOURCE_IDS):
                continue
            for rel in entity.relationships:
                if rel.target in entity_ids and self._source_id(self.registry.get_entity(rel.target)) in set(LOW_BALANCE_SOURCE_IDS):
                    target = self.registry.get_entity(rel.target)
                    relationships.append(SchemaRelationship(
                        source_entity=entity.canonical_id,
                        target_entity=target.canonical_id,
                        relation=rel.relation,
                        cardinality=rel.cardinality,
                        required=rel.required,
                        source_references=[s.get("reference", "") for s in entity.sources],
                    ))

        standards = self.registry.standards_for_entities([e.canonical_id for e in entities])
        hard_constraints = [
            "scenarioId is identifier-only and does not select variables or business rules.",
            "Low Balance & Top-up standards grounding is restricted to the supplied TMF654 Prepay Balance Management and TMF629 Customer Management Swagger/OpenAPI artifacts.",
            "The Low Balance & Top-up compiler permits executable variables only when they are exact scalar leaves from the supplied TMF654/TMF629 Swagger models or are explicitly layered in from MongoDB after proposal generation.",
            "One-to-many array properties are excluded from the flat record contract rather than converted into fake scalar values.",
            "Standard-backed enum values are copied from the official Swagger definitions and cannot be replaced with invented values.",
            "The LLM cannot create scenario-derived executable variables. Variables absent from the two Swagger sources must be supplied through MongoDB.",
            f"Scenario outcome mode is derived from scenarioType and the business scenario: {scenario_mode}.",
            "Transactional entity-level fields remain stable across a subscriber history; transaction/event fields are regenerated per record.",
        ]
        if entity_key:
            hard_constraints.append(f"'{entity_key}' is the authoritative entity key for transactional grouping.")
        warnings = [
            "Any additional non-TMF business variable must be provided through MongoDB; it is never invented by the LLM.",
        ]
        return ScenarioSchema(
            domain=intent.domain,
            subdomain=intent.subdomain,
            applicable_standards=standards,
            entities=[
                ResolvedConcept(
                    canonical_id=e.canonical_id,
                    name=e.name,
                    source_model="/".join(sorted({s["standard"] for s in e.sources})),
                    source_references=[s.get("reference", "") for s in e.sources],
                    selected_attributes=[a.name for a in e.attributes],
                ) for e in entities if self._source_id(e) in set(LOW_BALANCE_SOURCE_IDS)
            ],
            relationships=relationships,
            fields=fields,
            hard_constraints=hard_constraints,
            unresolved_items=unresolved,
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
    ) -> ScenarioSchema:
        requested = intent
        normalized_industry = (industry_type or intent.industry_type or "telecom").strip().lower()
        if normalized_industry not in {"telecom", "telecommunications"}:
            raise ValueError(
                f"Unsupported industryType '{industry_type or intent.industry_type}'. The telecom registry only supports Telecommunications/Telecom."
            )
        if is_json_grounded_domain(domain_query or intent.domain):
            return self._compile_low_balance_json_grounded(
                requested,
                business_scenario=business_scenario or "",
                use_case=use_case or requested.use_case or "",
                country=country,
                type_of_data=type_of_data or requested.type_of_data,
                entity_key=entity_key,
                scenario_type=scenario_type,
                excluded_field_names=excluded_field_names,
            )
        if selected_entities is not None:
            normalized: list[str] = []
            for value in selected_entities:
                ent = self.registry.resolve_entity(value)
                if ent is None:
                    raise ValueError(f"HITL selected unknown registry entity '{value}'")
                normalized.append(ent.canonical_id)
            requested = requested.model_copy(update={"requested_entities": normalized})

        entities, unresolved = self.resolve(
            requested,
            domain_query=domain_query,
            business_scenario=business_scenario,
            use_case=use_case,
            scenario_type=scenario_type,
            type_of_data=type_of_data,
            entity_key=entity_key,
        )
        entity_ids = {e.canonical_id for e in entities}
        # Expand only as far as the request semantics and registry relationships justify.
        # There is deliberately no minimum-width padding loop.
        frontier = list(entities)
        visited = set(entity_ids)
        # The registry is built only from official source artifacts. Once the scenario has
        # established its approved semantic anchors, relationship expansion may use the complete
        # official model graph. There is deliberately no application-level entity/hop ceiling.
        allowed_expansion_ids = {item["canonical_id"] for item in self.registry.catalog_summary(limit=None)} | entity_ids
        # Expand until the approved registry graph reaches a fixed point. There is no
        # hop-count or entity-count ceiling; the current scenario and registry graph are
        # the constraints.
        while frontier:
            candidates: list[str] = []
            for entity in frontier:
                for rel in entity.relationships:
                    if rel.target in allowed_expansion_ids and rel.target not in visited and self.registry.entity_exists(rel.target):
                        candidates.append(rel.target)
                for related in self.registry.related_entities(entity.canonical_id):
                    if related.canonical_id in allowed_expansion_ids and related.canonical_id not in visited:
                        candidates.append(related.canonical_id)
            unique = list(dict.fromkeys(candidates))
            unique.sort(key=lambda cid: (-self._expansion_score(self.registry.get_entity(cid), scenario_type=scenario_type, type_of_data=type_of_data, use_case=use_case), cid))
            new_frontier: list[EntityDef] = []
            for cid in unique:
                visited.add(cid)
                ent = self.registry.get_entity(cid)
                entities.append(ent)
                entity_ids.add(cid)
                new_frontier.append(ent)
            frontier = new_frontier

        scenario_mode = classify_outcome_mode(
            scenario_type=scenario_type or requested.scenario_type,
            expected_outcome=expected_outcome or "",
            business_response=business_response or "",
            business_scenario=business_scenario or "",
        )
        fields = self._build_fresh_fields(
            requested,
            entities,
            entity_key=entity_key,
            country=country,
            type_of_data=type_of_data,
            scenario_mode=scenario_mode,
            max_variables=max_variables,
            excluded_field_names=excluded_field_names,
            business_scenario=business_scenario,
            context_text=" ".join(
                str(value or "") for value in (
                    requested.industry_type, requested.domain, requested.subdomain,
                    scenario_type or requested.scenario_type, type_of_data or requested.type_of_data,
                    use_case or requested.use_case, entity_key or "", business_scenario or "",
                    business_response or "", expected_outcome or "", country or "",
                )
            ),
        )
        field_names = [f.name for f in fields]
        if not fields:
            unresolved.append("The scenario did not yield any usable semantic variables.")
        if (
            entity_key
            and self._normalize_variable_name(entity_key) not in {self._normalize_variable_name(n) for n in field_names}
            and not self.is_mandatory_telecom_field(entity_key)
        ):
            unresolved.append(f"Requested entity key '{entity_key}' could not be represented by the proposed variables")

        relationships: list[SchemaRelationship] = []
        for entity in entities:
            for rel in entity.relationships:
                if rel.target in entity_ids:
                    target = self.registry.get_entity(rel.target)
                    relationships.append(SchemaRelationship(
                        source_entity=entity.canonical_id,
                        target_entity=target.canonical_id,
                        relation=rel.relation,
                        cardinality=rel.cardinality,
                        required=rel.required,
                        source_references=[s.get("reference", "") for s in entity.sources],
                    ))

        applicable_standards = self.registry.standards_for_entities([e.canonical_id for e in entities])
        hard_constraints = [
            "scenarioId is identifier-only and does not select variables or business rules.",
            "The proposal is a fresh semantic variable set derived from the current scenario context; it is not a replay of a fixed scenario template.",
            "Schema width is quality-gated: the compiler maximizes scenario-relevant analytical coverage up to the configured variable budget, removes semantic duplicates and low-value transport metadata, and never drops mandatory contracts for size.",
            "Each semantic variable is compiled into a deterministic executable generator contract.",
            "Transactional entity-grain variables are stable across the entity history; transaction/event/derived variables are regenerated per transaction/event.",
            "When subscriber_id is the telecom identity anchor, customer_id is excluded as a redundant proposal field unless customer_id is explicitly requested as the transactional entity key.",
            "Generated records must pass deterministic type, choice, dependency, temporal, formula and scenario-semantic validation.",
            f"Scenario outcome mode is derived dynamically from the complete request context: {scenario_mode}.",
        ]
        if entity_key:
            hard_constraints.append(f"'{entity_key}' is the authoritative entity key for transactional grouping.")
        if (requested.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            hard_constraints.append("subscriber_id, account_id and msisdn are mandatory non-null entity-level fields in telecom scenario schemas.")
            hard_constraints.append("Schema width is broad but quality-gated over scalar attributes on the resolved scenario registry graph; semantic duplicates and low-value API metadata are deprioritized, while unsupported nested object/array structures are excluded from the flat record contract.")
        warnings = [
            "Scenario IDs are identifiers only; variables and generation rules are derived from the current request, registry grounding, and scenario semantics.",
        ]
        return ScenarioSchema(
            domain=domain_query or requested.domain or requested.subdomain,
            subdomain=requested.subdomain,
            applicable_standards=applicable_standards,
            entities=[
                ResolvedConcept(
                    canonical_id=e.canonical_id,
                    name=e.name,
                    source_model="/".join(sorted({s["standard"] for s in e.sources})),
                    source_references=[s.get("reference", "") for s in e.sources],
                    selected_attributes=[a.name for a in e.attributes],
                ) for e in entities
            ],
            relationships=relationships,
            fields=fields,
            hard_constraints=hard_constraints,
            unresolved_items=unresolved,
            warnings=warnings,
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
