"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship
from core.telecom_registry import EntityDef, TelecomRegistry
from config.industry_profiles import get_profile
import re
from difflib import SequenceMatcher
from core.scenario_semantics import classify_outcome_mode
from core.pdf_domain_policy import (
    PDF_CATALOG, catalog_for_request, is_pdf_grounded_domain, pdf_provenance,
    use_case_for, PDF_SOURCE_STANDARDS, PDF_SOURCE_NAMES,
)


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1]


class SchemaCompiler:
    """Compiles a scenario schema only from the runtime standards registry."""

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
    ) -> tuple[list[EntityDef], list[str]]:
        """Resolve concepts from the approved registry.

        The semantic inputs are deliberately separated: ``domain`` establishes the
        primary registry slice, while the LLM's requested concepts refine that slice.
        ``business_scenario`` is interpreted by the intent agent; it is not used as an
        uncontrolled full-catalog search. ``scenario_id`` is intentionally absent.
        """
        resolved: list[EntityDef] = []
        unresolved: list[str] = []
        seen: set[str] = set()

        # Domain is the primary selector. Only these registry results establish the
        # initial domain boundary; this prevents a long natural-language request from
        # accidentally matching unrelated telecom entities.
        domain_candidates = self.registry.search(domain_query or intent.domain, limit=None)
        for candidate in domain_candidates:
            entity = self.registry.resolve_entity(candidate["canonical_id"])
            if entity and entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)

        # Explicit concepts extracted by the LLM are allowed only when they exist in the
        # authoritative registry. This is where businessScenario influences the proposal.
        # No free-form entity names are copied directly into the schema.
        for requested in intent.requested_entities:
            entity = self.registry.resolve_entity(requested)
            if entity is None:
                # Requested entities are semantic hints from the LLM, not executable
                # registry IDs. A business concept can legitimately have no one-to-one
                # registry entity (for example ``retention_intervention``). Never guess a
                # registry entity from a fuzzy match here: an incorrect entity silently
                # changes the generated schema. Candidate variables, the requested entity
                # key, domain grounding, and approved relationship expansion are the safe
                # sources for executable registry entities. Unknown hints are therefore
                # ignored rather than treated as confirmation-blocking errors.
                continue
            if entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)

        # Candidate variable names can identify the most relevant registry entities even
        # when the LLM did not explicitly list those entities. This keeps semantic grounding
        # dynamic while preventing a generic domain search from selecting an unrelated field
        # owner when common attributes collide across telecom entities.
        for idea in intent.candidate_variables:
            idea_name = str(idea.name or "").strip()
            if not idea_name:
                continue
            owners = self.registry.entities_with_attribute(idea_name)
            for owner in owners:
                if owner.canonical_id not in seen:
                    resolved.append(owner)
                    seen.add(owner.canonical_id)

        # Entity key is authoritative and should be present whenever the registry supports
        # it. Entity keys are field names, not entity IDs, so resolve them by inspecting
        # approved registry attributes rather than aliases/canonical entity IDs.
        if entity_key:
            key_entities = self.registry.entities_with_attribute(entity_key)
            key_entity = key_entities[0] if key_entities else None
            if key_entity and key_entity.canonical_id not in seen:
                resolved.insert(0, key_entity)
                seen.add(key_entity.canonical_id)
            elif key_entity is None and not self.is_mandatory_telecom_field(entity_key):
                # subscriber_id/account_id/msisdn are stable application-level telecom
                # contract anchors. They are deliberately injected by the compiler even
                # when an official model uses a different identifier name or nests the
                # identifier under a different object. They must not become an unresolved
                # registry requirement and must never block HITL confirmation.
                unresolved.append(f"Entity key '{entity_key}' is not a field in the approved telecom registry")

        # Use-case is a semantic selector within telecom. It contributes only approved
        # concepts whose registry domain matches the use case; no LLM invention occurs here.
        normalized_use_case = (use_case or intent.use_case or intent.subdomain or "").strip().lower()
        if normalized_use_case and normalized_use_case != "unknown":
            use_case_candidates = self.registry.catalog_summary(domain=normalized_use_case, limit=None)
            for candidate in use_case_candidates:
                entity = self.registry.resolve_entity(candidate["canonical_id"])
                if entity and entity.canonical_id not in seen:
                    resolved.append(entity)
                    seen.add(entity.canonical_id)

        # Subscriber/account/MSISDN are mandatory for telecom proposals. Bring their
        # authoritative registry entities into the compile set even when the LLM does not
        # explicitly mention them. This guarantees a stable subscriber/account/contact
        # context and allows the mandatory fields to use their real registry generators.
        if (intent.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            for anchor_id in ("subscriber", "customer_account", "customer"):
                if self.registry.entity_exists(anchor_id) and anchor_id not in seen:
                    resolved.append(self.registry.get_entity(anchor_id))
                    seen.add(anchor_id)

        # Maximize scenario coverage without dumping an unrelated universal catalog.
        # Lexically match the complete business request/domain/use-case against every
        # registry entity; every matching entity then participates in graph expansion.
        # This is deliberately unbounded because the registry itself is the source of truth.
        context_terms = " ".join(
            str(value or "") for value in (
                domain_query or intent.domain,
                business_scenario or "",
                use_case or intent.use_case,
                scenario_type or intent.scenario_type,
                *list(intent.requested_entities),
            )
        ).strip()
        if context_terms:
            for candidate in self.registry.search(context_terms, limit=None):
                entity = self.registry.resolve_entity(candidate["canonical_id"])
                if entity and entity.canonical_id not in seen:
                    resolved.append(entity)
                    seen.add(entity.canonical_id)

        # Prepaid is subscriber-centric in the platform contract; graph expansion will add
        # the connected prepaid/account/recharge/bucket/charging concepts when the registry
        # relationships support them.
        if normalized_use_case == "prepaid" and self.registry.entity_exists("subscriber"):
            if "subscriber" not in seen:
                resolved.append(self.registry.get_entity("subscriber"))
                seen.add("subscriber")

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
                if exact or (strong_semantic and score >= 8.0):
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
        if any(token in n for token in ("_amount", "_balance", "_quota", "_score", "_rate", "_percentage", "_percent")):
            return "float"
        if any(token in n for token in ("_count", "_days", "_months", "_hours", "_minutes", "number_of", "num_")):
            return "integer"
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
        if dtype in {"integer", "int"} or role == "metric":
            return "uniform_int", "integer", {"min": 0, "max": 100, "precision": 0}
        if dtype in {"float", "decimal", "number", "numeric"} or role == "measurement":
            return "uniform", "float", {"min": 0.0, "max": 1000.0, "precision": 2}
        if dtype == "boolean":
            return "weighted_choice", "boolean", {"choices": [False, True], "weights": [0.5, 0.5]}

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
    ) -> list[GeneratedSchemaField]:
        raw_ideas = [idea.model_dump() for idea in intent.candidate_variables]
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
        if (intent.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            mandatory_defs = {
                "subscriber_id": ("subscriber", "subscriber_id", "Stable subscriber identifier."),
                "account_id": ("subscriber", "account_id", "Stable subscriber account identifier."),
                "msisdn": ("subscriber", "msisdn", "Subscriber MSISDN / mobile telephone number."),
            }
            for field_name in self.REQUIRED_TELECOM_FIELDS:
                _, _, desc = mandatory_defs[field_name]
                add_idea({
                    "name": field_name,
                    "description": desc,
                    "role": "identity" if field_name.endswith("_id") else "profile",
                    "grain": "entity",
                    "dtype": "string",
                    "depends_on": [],
                })

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

        # Comprehensive registry grounding:
        #
        # The proposal must retain broad semantic coverage (the prior contract exposed
        # hundreds of registry-backed variables), while the generation side must stay safe.
        # Therefore we include every *materializable scalar* attribute on the resolved
        # scenario graph, not only the LLM's candidate list. This preserves dynamic width
        # without reintroducing arbitrary nested `{}` objects or semantically unrelated
        # placeholder contracts.
        #
        # Nested object/array attributes are intentionally excluded from the flat synthetic
        # row contract. Their scalar children remain available when the official model exposes
        # them as separate attributes.
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

        # Intentionally do not truncate candidate variables. ``max_variables`` remains only
        # for backward compatibility with older callers and is deliberately ignored.

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
            # Mandatory public telecom anchors use authoritative subscriber attributes
            # directly, rather than allowing a generic account_id match to resolve to a
            # different entity.
            if original_name in self.REQUIRED_TELECOM_FIELDS:
                subscriber = next((e for e in entities if e.canonical_id == "subscriber"), None)
                if subscriber is not None:
                    attr = next((a for a in subscriber.attributes if a.name == original_name), None)
                    if attr is not None:
                        matched = (subscriber, attr)
            if matched is None:
                matched = self._match_registry_attribute(idea, entities, used_registry)
            role = str(idea.get("role") or "other").lower()
            idea_text = f"{original_name} {idea.get('description', '')}".lower()
            outcome_semantic = role in {"status", "decision"} or any(token in idea_text for token in ("status", "state", "outcome", "decision", "result"))
            if original_name in self.REQUIRED_TELECOM_FIELDS:
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
                    # Never borrow executable params from a fuzzy/non-exact attribute.
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
            deps = self._merge_dependencies(idea, name_map, grain, entity_key)
            if original_name == entity_key or original_name in self.REQUIRED_TELECOM_FIELDS:
                grain = "entity"
                deps = []
            registry_required = bool(idea.get("_registry_required")) if idea.get("_registry_required") is not None else False
            registry_nullable = bool(idea.get("_registry_nullable")) if idea.get("_registry_nullable") is not None else True
            required = original_name == entity_key or original_name in self.REQUIRED_TELECOM_FIELDS or registry_required
            nullable = False if required else registry_nullable
            description = str(idea.get("description") or "").strip() or f"Scenario-specific {role.replace('_', ' ')} attribute for {intent.domain}."
            provenance = {
                "generated_from": "registry_attribute" if idea.get("_preserve_name") else "semantic_variable_idea",
                "canonical_entity": entity.canonical_id if entity else None,
                "source_registry_attribute": getattr(attr, "name", None) if matched else None,
                "grain": grain,
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

    def _compile_pdf_grounded(
        self,
        intent: ScenarioIntent,
        *,
        business_scenario: str,
        use_case: str,
        country: str | None,
        type_of_data: str | None,
        entity_key: str | None,
        scenario_type: str | None = None,
        industry_type: str | None = None,
    ) -> ScenarioSchema:
        """Compile Low Balance & Top-up from the supplied PDF catalog.

        The PDF catalog remains the semantic source of truth for domain fields. The
        application-level subscriber/account/MSISDN anchors are added separately because
        they are a mandatory telecom row contract, not PDF semantic claims.

        The compiler intentionally emits the *maximum relevant flat scalar width* from the
        selected PDF resources. Scenario inputs still control resource selection, scope,
        country-aware generators, status semantics, and business-specific categorical bias.
        """
        catalog = catalog_for_request(business_scenario, use_case)
        request_text = f"{business_scenario} {use_case}".lower()
        normalized_country = str(country or "IN").strip().upper()
        normalized_type = str(type_of_data or intent.type_of_data or "transactional").strip().lower()
        normalized_scenario = str(scenario_type or intent.scenario_type or "").strip()
        scenario_mode = classify_outcome_mode(
            scenario_type=normalized_scenario,
            expected_outcome="",
            business_response="",
            business_scenario=business_scenario or "",
        )

        resources: list[str] = ["bucket", "topupbalance"]
        if any(token in request_text for token in ("transfer", "colleague", "send balance")) and "transferbalance" in catalog:
            resources.append("transferbalance")
        if any(token in request_text for token in ("reserve", "reservation", "reserve balance")) and "reservebalance" in catalog:
            resources.append("reservebalance")
        if any(token in request_text for token in ("usage", "voice call", "voicemail", "usage specification", "rated", "billed")):
            for rid in ("usage", "usagespecification"):
                if rid in catalog:
                    resources.append(rid)
        resources = list(dict.fromkeys(r for r in resources if r in catalog))

        # Application-level telecom anchors are always present, regardless of entityKey.
        fields: list[GeneratedSchemaField] = [
            GeneratedSchemaField(
                name="subscriber_id",
                dtype="string",
                description="Stable synthetic subscriber identifier used to group transactional history.",
                gen="prefixed_int",
                params={"prefix": "SUB-", "digits": 10},
                depends_on=[],
                nullable=False,
                required=True,
                formula=None,
                scope="entity",
                useCase=use_case or "prepaid",
                provenance={"source_policy": "application_contract", "source": "APPLICATION_REQUIRED", "reason": "mandatory telecom identity anchor"},
            ),
            GeneratedSchemaField(
                name="account_id",
                dtype="string",
                description="Stable synthetic subscriber account identifier linked to subscriber_id.",
                gen="id_mirror",
                params={"prefix": "ACC-", "source_field": "subscriber_id", "source_prefix": "SUB-"},
                depends_on=["subscriber_id"],
                nullable=False,
                required=True,
                formula=None,
                scope="entity",
                useCase=use_case or "prepaid",
                provenance={"source_policy": "application_contract", "source": "APPLICATION_REQUIRED", "reason": "mandatory telecom identity anchor"},
            ),
            GeneratedSchemaField(
                name="msisdn",
                dtype="string",
                description="Synthetic subscriber MSISDN in a country-aware E.164-like format.",
                gen="e164_phone",
                params={"country_codes": [], "country": normalized_country},
                depends_on=[],
                nullable=False,
                required=True,
                formula=None,
                scope="entity",
                useCase=use_case or "prepaid",
                provenance={"source_policy": "application_contract", "source": "APPLICATION_REQUIRED", "reason": "mandatory telecom identity anchor"},
            ),
        ]
        dial_codes = {
            "IN": "+91", "US": "+1", "CA": "+1", "GB": "+44", "AU": "+61",
            "AE": "+971", "SG": "+65", "DE": "+49", "FR": "+33", "IT": "+39",
        }
        fields[2].params["country_codes"] = [dial_codes.get(normalized_country, normalized_country if normalized_country.startswith("+") else "+" + normalized_country)]

        profile = get_profile("telecom", normalized_country)
        currency = str(profile.get("currency") or ("INR" if normalized_country == "IN" else "")).upper() or "INR"

        def item_base_type(resource: str) -> str:
            return {
                "bucket": "Bucket",
                "topupbalance": "TopupBalance",
                "transferbalance": "TransferBalance",
                "reservebalance": "ReserveBalance",
                "usage": "Usage",
                "usagespecification": "UsageSpecification",
            }.get(resource, resource)

        def item_type(resource: str) -> str:
            # TMF @type is an extensibility/type discriminator. Use the resource
            # type itself rather than unrelated subscriber/product labels.
            return item_base_type(resource)

        def operation_status_params() -> dict[str, object]:
            # Normal/happy-path transactional operations are successful. Other scenario types
            # retain a complete executable vocabulary while deterministic scenario semantics
            # select the matching outcome.
            choices = ["COMPLETED", "FAILED", "PENDING"]
            return {"choices": choices, "weights": [1.0, 0.0, 0.0] if scenario_mode == "positive" else [1.0, 1.0, 1.0]}

        def field_params(resource: str, pdf_field: str, dtype: str) -> tuple[str, dict[str, object]]:
            if dtype == "categorical":
                choices = list(catalog[resource].get("choices", {}).get(pdf_field, []))
                if resource == "bucket" and pdf_field == "status":
                    # Normal means an active bucket; preserve the complete documented vocabulary.
                    return "weighted_choice", {"choices": choices or ["active", "expired", "suspended"], "weights": [1.0, 0.0, 0.0] if scenario_mode == "positive" and len(choices) == 3 else [1.0] * len(choices)}
                if choices:
                    return "weighted_choice", {"choices": choices, "weights": [1.0] * len(choices)}
                return "semantic_string", {}
            if pdf_field == "status" and resource in {"topupbalance", "transferbalance", "reservebalance"}:
                return "weighted_choice", operation_status_params()
            if pdf_field in {"isAutoTopup"}:
                return "weighted_choice", {"choices": [True, False], "weights": [0.55, 0.45]}
            if pdf_field == "numberOfPeriods":
                return "uniform_int", {"min": 1, "max": 12}
            if pdf_field == "recurringPeriod":
                return "weighted_choice", {"choices": ["weekly", "monthly"], "weights": [0.35, 0.65]}
            if pdf_field in {"remainingValue", "reservedValue", "amount", "transferCost"}:
                return "uniform", {"min": 0, "max": 1000, "precision": 2, "currency": currency}
            if pdf_field.endswith("_units") or pdf_field == "amount_units" or pdf_field == "transferCost_units":
                return "constant", {"value": currency}
            if pdf_field in {"requestedDate", "confirmationDate", "usageDate", "lastUpdate"}:
                return "recent_datetime", {"days_back": 365, "timezone": "Asia/Kolkata" if normalized_country == "IN" else "UTC", "timestamp_format": "dd/mm/yyyy hh:mm a"}
            if pdf_field == "usageType" and resource in {"bucket", "topupbalance"}:
                return "constant", {"value": "currency"}
            if pdf_field == "@baseType":
                return "constant", {"value": item_base_type(resource)}
            if pdf_field == "@type":
                return "constant", {"value": item_type(resource)}
            if pdf_field == "reason" and resource in {"topupbalance", "transferbalance", "reservebalance"}:
                return "weighted_choice", {
                    "choices": ["LOW_BALANCE", "CUSTOMER_REQUEST", "VALIDITY_EXPIRY", "DATA_EXHAUSTED"],
                    "weights": [0.65, 0.20, 0.10, 0.05],
                }
            if pdf_field in {"isShared"}:
                return "weighted_choice", {"choices": [False, True], "weights": [0.85, 0.15]}
            return {
                "string": "semantic_string", "datetime": "recent_datetime", "integer": "uniform_int",
                "float": "uniform", "boolean": "weighted_choice", "categorical": "weighted_choice",
            }.get(dtype, "semantic_string"), ({"days_back": 365} if dtype == "datetime" else ({"min": 1, "max": 12} if dtype == "integer" else ({"min": 0, "max": 1000, "precision": 2} if dtype == "float" else {})))

        for resource in resources:
            item = catalog[resource]
            if resource == "bucket":
                entity_scope_fields = set(item["fields"].keys())
            else:
                entity_scope_fields = set()
            selected_attributes = list(item["fields"].keys())
            selected_concept = ResolvedConcept(
                canonical_id=resource,
                name=item["name"],
                source_model=item["standard"],
                source_references=[PDF_SOURCE_NAMES[item["source_id"]]],
                selected_attributes=selected_attributes,
            )
            # Stash selected concepts once, after the loop below.
            for pdf_field, (dtype, description) in item["fields"].items():
                gen, params = field_params(resource, pdf_field, dtype)
                # Keep schema metadata fields executable but stable at entity scope for bucket.
                grain = "entity" if resource == "bucket" else "transaction"
                nullable = False
                required = False
                if resource == "topupbalance" and pdf_field in {"numberOfPeriods", "recurringPeriod", "voucher"}:
                    # One-time top-ups do not require recurring configuration or vouchers.
                    nullable = True
                base = self._normalize_variable_name(f"{resource}_{pdf_field}")
                fields.append(GeneratedSchemaField(
                    name=base,
                    dtype=dtype,
                    description=description,
                    gen=gen,
                    params=params,
                    depends_on=[],
                    nullable=nullable,
                    required=required,
                    formula=None,
                    scope=grain,
                    useCase=use_case_for(resource, pdf_field, business_scenario, use_case),
                    provenance=pdf_provenance(resource, pdf_field, use_case_for(resource, pdf_field, business_scenario, use_case)),
                ))

        # Annotate selected concepts exactly once from the chosen resources.
        selected_entities = [
            ResolvedConcept(
                canonical_id=resource,
                name=catalog[resource]["name"],
                source_model=catalog[resource]["standard"],
                source_references=[PDF_SOURCE_NAMES[catalog[resource]["source_id"]]],
                selected_attributes=list(catalog[resource]["fields"].keys()),
            )
            for resource in resources
        ]
        selected_standards = sorted({catalog[r]["standard"] for r in resources})

        # Width is intentionally the full scalar catalog for this scenario, not an LLM-driven
        # subset. This is the concrete implementation of "maximum relevant variables".
        standards = [
            {
                "standard": standard,
                "source_policy": "supplied_pdf_only",
                "source_documents": [
                    PDF_SOURCE_NAMES[sid]
                    for sid in ("tmf654_v4", "tmf635_v4")
                    if ("TMF654" if sid == "tmf654_v4" else "TMF635") == standard
                ],
            }
            for standard in selected_standards
        ]
        hard_constraints = [
            "scenarioId is identifier-only and does not select variables or business rules.",
            "All business/domain semantic fields are grounded only in the supplied TMF654/TMF635 PDFs for Low Balance & Top-up.",
            "Application-level telecom identity anchors subscriber_id, account_id and msisdn are always included and are backend contract fields, not PDF semantic claims.",
            "The proposal uses the full scalar field set of every PDF resource selected by the businessScenario/useCase context; there is no variable-count cap or truncation.",
            "scenarioType, country, typeOfData, industryType, domain, useCase and businessScenario affect executable scope/generation semantics; scenarioId and entityKey do not ideate variables.",
            "For Normal scenarios, successful top-up operation status is deterministic (COMPLETED) and bucket status is active unless an explicit adverse outcome is requested.",
            "Top-up requestedDate must not occur after confirmationDate; recurring fields are coherent with isAutoTopup; bucket balance/usage fields are internally consistent.",
            "Transactional entity-level fields are stable across history records; transaction-level fields are regenerated per transaction.",
        ]
        warnings = [
            "PDF-grounded mode does not invent trigger-threshold fields absent from the supplied PDFs; the request is represented using the complete applicable scalar PDF model plus mandatory telecom identity anchors.",
        ]
        return ScenarioSchema(
            domain=intent.domain,
            subdomain=intent.subdomain,
            applicable_standards=standards,
            entities=selected_entities,
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
    ) -> ScenarioSchema:
        requested = intent
        normalized_industry = (industry_type or intent.industry_type or "telecom").strip().lower()
        if normalized_industry not in {"telecom", "telecommunications"}:
            raise ValueError(
                f"Unsupported industryType '{industry_type or intent.industry_type}'. The telecom registry only supports Telecommunications/Telecom."
            )
        if is_pdf_grounded_domain(domain_query or intent.domain):
            return self._compile_pdf_grounded(
                requested,
                business_scenario=business_scenario or "",
                use_case=use_case or requested.use_case or "",
                country=country,
                type_of_data=type_of_data or requested.type_of_data,
                entity_key=entity_key,
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
            "There is no artificial variable-count target or maximum; schema width is determined by the current scenario and approved registry grounding. All materializable scalar attributes on resolved entities are eligible for inclusion.",
            "Each semantic variable is compiled into a deterministic executable generator contract.",
            "Transactional entity-grain variables are stable across the entity history; transaction/event/derived variables are regenerated per transaction/event.",
            "Generated records must pass deterministic type, choice, dependency, temporal, formula and scenario-semantic validation.",
            f"Scenario outcome mode is derived dynamically from the complete request context: {scenario_mode}.",
        ]
        if entity_key:
            hard_constraints.append(f"'{entity_key}' is the authoritative entity key for transactional grouping.")
        if (requested.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            hard_constraints.append("subscriber_id, account_id and msisdn are mandatory non-null entity-level fields in telecom scenario schemas.")
            hard_constraints.append("Schema width is comprehensive over scalar attributes on the resolved scenario registry graph, with no fixed variable-count cap; unsupported nested object/array structures are excluded from the flat record contract.")
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
