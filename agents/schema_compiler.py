"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship
from core.telecom_registry import EntityDef, TelecomRegistry
from config.industry_profiles import get_profile
import re
from difflib import SequenceMatcher
from core.scenario_semantics import classify_outcome_mode


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1]


class SchemaCompiler:
    """Compiles a scenario schema only from the runtime standards registry."""

    # These fields are part of the public telecom row contract. Keep their exact
    # names even when the rest of the schema uses fresh
    # scenario-specific names. They are stable subscriber/account contact anchors.
    REQUIRED_TELECOM_FIELDS = ("subscriber_id", "account_id", "msisdn")

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
            elif key_entity is None:
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
            return "generic", runtime_params
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
                if score >= 3.0:
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
        """Create a descriptive name without blindly replaying registry vocabulary.

        The requested entity key is intentionally preserved because it is the public
        grouping key. Other names that exactly collide with the approved registry are
        given a descriptive semantic variant; no scenario-ID-specific mapping is used.
        """
        base = self._normalize_variable_name(requested)
        entity_key_norm = self._normalize_variable_name(entity_key or "")
        candidate_bases = [base]
        if base in registry_names and base != entity_key_norm:
            replacements = (
                ("_status", "_state"),
                ("_timestamp", "_event_time"),
                ("_amount", "_value"),
                ("_count", "_volume"),
                ("_id", "_key"),
            )
            for suffix, replacement in replacements:
                if base.endswith(suffix):
                    candidate_bases.insert(0, f"{base[:-len(suffix)]}{replacement}")
                    break
            desc_tokens = [token for token in _tokens(description) if len(token) > 2]
            if desc_tokens:
                candidate_bases.insert(0, self._normalize_variable_name("_".join(desc_tokens[:4])))
            candidate_bases.append(f"scenario_{base}")

        for candidate in candidate_bases:
            if candidate and candidate not in used:
                return candidate
        index = 2
        while f"{base}_{index}" in used:
            index += 1
        return f"{base}_{index}"

    def _generic_contract_for_idea(
        self,
        idea: dict[str, object],
        country: str | None,
        scenario_mode: str,
    ) -> tuple[str, str, dict[str, object]]:
        name = str(idea.get("name") or "scenario_attribute")
        dtype = str(idea.get("dtype") or "string").lower()
        role = str(idea.get("role") or "other").lower()
        desc = str(idea.get("description") or "")
        country_code = str(country or "IN").upper()

        if role == "identity" or name.lower().endswith(("_id", "_key")):
            return "prefixed_int", "string", {"prefix": f"{name[:-3].upper()}-" if name.lower().endswith("_id") else "ID-", "digits": 10}
        if dtype in {"datetime", "timestamp"} or role == "timing":
            return "recent_datetime", "datetime", {"timezone": "Asia/Kolkata" if country_code == "IN" else "UTC", "days_back": 365}
        if dtype == "date":
            return "recent_datetime", "date", {"timezone": "Asia/Kolkata" if country_code == "IN" else "UTC", "days_back": 365}
        if dtype in {"integer", "int"} or role == "metric":
            return "uniform_int", "integer", {"min": 0, "max": 100, "precision": 0}
        if dtype in {"float", "decimal"} or role == "measurement":
            return "uniform", "float", {"min": 0.0, "max": 1000.0, "precision": 2}
        if dtype in {"boolean", "bool"}:
            return "generic", "boolean", {}
        if dtype == "categorical" or role in {"status", "decision", "configuration", "categorical"}:
            lower = f"{name} {desc}".lower()
            mode = scenario_mode or "mixed"
            is_outcome_field = role in {"status", "decision"} or any(token in lower for token in ("status", "state", "outcome", "decision", "result"))
            if is_outcome_field:
                mode_values = {
                    "positive": {
                        "status": ["COMPLETED", "SUCCESS", "APPROVED"],
                        "decision": ["ACCEPTED", "APPROVED", "COMPLETED"],
                    },
                    "negative": {
                        "status": ["FAILED", "ERROR", "REJECTED"],
                        "decision": ["REJECTED", "DENIED", "FAILED"],
                    },
                    "suppression": {
                        "status": ["SUPPRESSED", "HELD", "SKIPPED"],
                        "decision": ["SUPPRESSED", "HELD", "NOT_SENT"],
                    },
                    "decline_or_no_response": {
                        "status": ["DECLINED", "NO_RESPONSE", "REJECTED"],
                        "decision": ["DECLINED", "NO_RESPONSE", "REJECTED"],
                    },
                    "concurrent": {
                        "status": ["NO_CLEAR_PRIORITY", "CONFLICT", "PENDING_PRIORITY"],
                        "decision": ["NO_CLEAR_PRIORITY", "CONFLICT", "SELECTED"],
                    },
                    "mixed": {
                        "status": ["COMPLETED", "FAILED", "PENDING"],
                        "decision": ["ACCEPTED", "DECLINED", "PENDING"],
                    },
                }
                mode_candidates = mode_values.get(mode, mode_values["mixed"])
                candidates = mode_candidates.get("decision" if role == "decision" else "status", mode_candidates["status"])
                choices = list(dict.fromkeys(candidates))
                return "weighted_choice", "categorical", {"choices": choices, "weights": [1.0] * len(choices)}
            if "channel" in lower or "method" in lower:
                choices = ["APP", "SMS", "WEB", "USSD", "OTHER"]
            elif "priority" in lower or "rank" in lower:
                choices = ["HIGH", "MEDIUM", "LOW"]
            elif "reason" in lower:
                choices = ["THRESHOLD", "CUSTOMER_ACTION", "SYSTEM_RULE", "OTHER"]
            else:
                choices = ["OTHER"]
            return "weighted_choice", "categorical", {"choices": choices, "weights": [1.0] * len(choices)}
        return "generic", "string", {}

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
        # entity grain. They are always registry-backed.
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

        # Comprehensive mode: every attribute on every entity that survived scenario/entity
        # resolution becomes a candidate unless the LLM already represented that concept.
        # This is the mechanism that actually maximizes schema width; it is not limited by a
        # fixed count. Exact registry attributes remain provenance-backed and deterministic.
        for entity in entities:
            for attr in entity.attributes:
                add_idea({
                    "name": attr.name,
                    "description": attr.description or f"{entity.name} {attr.name} attribute.",
                    "role": (
                        "timing" if str(attr.dtype).lower() in {"datetime", "timestamp", "date"}
                        else "measurement" if str(attr.dtype).lower() in {"float", "decimal", "number", "numeric", "int", "integer"}
                        else "status" if "status" in attr.name.lower() or "state" in attr.name.lower()
                        else "identity" if attr.name.lower().endswith("_id")
                        else "other"
                    ),
                    "grain": "entity" if entity.canonical_id in {"subscriber", "customer", "customer_account", "prepaid_account"} else "transaction",
                    "dtype": ("integer" if str(attr.dtype).lower() in {"int", "integer", "bigint", "smallint"} else
                              "float" if str(attr.dtype).lower() in {"float", "decimal", "number", "numeric"} else
                              "datetime" if str(attr.dtype).lower() in {"datetime", "timestamp"} else
                              "date" if str(attr.dtype).lower() == "date" else
                              "categorical" if attr.enum_values or str(attr.generator).lower() in {"weighted_choice", "dependent_choice", "categorical"} else
                              "boolean" if str(attr.dtype).lower() in {"bool", "boolean"} else "string"),
                    "depends_on": list(attr.depends_on),
                    "_registry_entity": entity.canonical_id,
                    "_registry_required": bool(attr.required),
                    "_registry_nullable": bool(attr.nullable),
                }, preserve_name=True)

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
            if matched is not None:
                entity, attr = matched
                used_registry.add((entity.canonical_id, attr.name))
                # Registry provenance still anchors the concept, but lifecycle/outcome fields
                # are compiled through the scenario-aware semantic contract. This prevents a
                # Normal/Failure/Suppression/etc. request from inheriting an incompatible
                # choice list simply because a registry attribute had a similar description.
                if outcome_semantic and scenario_mode:
                    runtime_generator, dtype, params = self._generic_contract_for_idea(idea, country, scenario_mode)
                else:
                    params = dict(attr.params or {})
                    if attr.generator == "msisdn":
                        params["country"] = str(country or params.get("country") or "IN").upper()
                    if country_currency and (attr.name.lower() == "currency" or "currency" in params):
                        params["currency"] = country_currency
                    runtime_generator, params = self._runtime_generation_contract(attr, params, country)
                    dtype = attr.dtype
                    if attr.generator == "msisdn":
                        dtype = "string"
            else:
                entity = self._choose_entity_for_idea(idea, entities, None, entity_key)
                runtime_generator, dtype, params = self._generic_contract_for_idea(idea, country, scenario_mode)

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
        allowed_expansion_ids = {
            item["canonical_id"] for item in self.registry.search(domain_query or intent.domain, limit=None)
        } | entity_ids
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
        if entity_key and self._normalize_variable_name(entity_key) not in {self._normalize_variable_name(n) for n in field_names}:
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
            "There is no artificial variable-count target or maximum; schema width is determined by the current scenario and approved registry grounding.",
            "Each semantic variable is compiled into a deterministic executable generator contract.",
            "Transactional entity-grain variables are stable across the entity history; transaction/event/derived variables are regenerated per transaction/event.",
            "Generated records must pass deterministic type, choice, dependency, temporal, formula and scenario-semantic validation.",
            f"Scenario outcome mode is derived dynamically from the complete request context: {scenario_mode}.",
        ]
        if entity_key:
            hard_constraints.append(f"'{entity_key}' is the authoritative entity key for transactional grouping.")
        if (requested.industry_type or "telecom").strip().lower() in {"telecom", "telecommunications"}:
            hard_constraints.append("subscriber_id, account_id and msisdn are mandatory non-null entity-level fields in telecom scenario schemas.")
            hard_constraints.append("Schema width is comprehensive by default: every attribute exposed by scenario-relevant registry entities is considered, with no variable-count cap.")
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
