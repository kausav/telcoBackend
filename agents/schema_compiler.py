"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship
from core.telecom_registry import EntityDef, TelecomRegistry
from config.industry_profiles import get_profile
import re


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) > 1]


class SchemaCompiler:
    """Compiles a scenario schema only from the runtime standards registry."""

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
        domain_candidates = self.registry.search(domain_query or intent.domain, limit=35)
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
                unresolved.append(f"Unknown concept '{requested}' is not in the approved telecom registry")
                continue
            if entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)

        # Entity key is authoritative and should be present whenever the registry supports
        # it. Entity keys are field names, not entity IDs, so resolve them by inspecting
        # approved registry attributes rather than aliases/canonical entity IDs.
        if entity_key:
            key_entity = None
            for candidate in self.registry.catalog_summary(limit=1000):
                entity = self.registry.resolve_entity(candidate["canonical_id"])
                if entity and any(a.name.lower() == entity_key.lower() for a in entity.attributes):
                    key_entity = entity
                    break
            if key_entity and key_entity.canonical_id not in seen:
                resolved.insert(0, key_entity)
                seen.add(key_entity.canonical_id)
            elif key_entity is None:
                unresolved.append(f"Entity key '{entity_key}' is not a field in the approved telecom registry")

        # Use-case is a semantic selector within telecom. It contributes only approved
        # concepts whose registry domain matches the use case; no LLM invention occurs here.
        normalized_use_case = (use_case or intent.use_case or intent.subdomain or "").strip().lower()
        if normalized_use_case and normalized_use_case != "unknown":
            use_case_candidates = self.registry.catalog_summary(domain=normalized_use_case, limit=35)
            for candidate in use_case_candidates:
                entity = self.registry.resolve_entity(candidate["canonical_id"])
                if entity and entity.canonical_id not in seen:
                    resolved.append(entity)
                    seen.add(entity.canonical_id)

        # Prepaid is subscriber-centric in the platform contract. Add only the approved
        # anchor; supporting account/event entities are added later by the graph expansion.
        if normalized_use_case == "prepaid" and self.registry.entity_exists("subscriber"):
            if "subscriber" not in seen:
                resolved.append(self.registry.get_entity("subscriber"))
                seen.add("subscriber")

        return resolved, unresolved

    @staticmethod
    def _root_entity_for_key(entity_key: str | None, entities: list[EntityDef]) -> str | None:
        if not entity_key:
            return None
        for entity in entities:
            if any(attr.name.lower() == entity_key.lower() for attr in entity.attributes):
                return entity.canonical_id
        return None

    def _eligible_field_count(self, entities: list[EntityDef], entity_ids: set[str], entity_key: str | None) -> int:
        """Count fields that will actually be projected into the response schema."""
        root_entity = self._root_entity_for_key(entity_key, entities)
        names: set[str] = set()
        for entity in entities:
            for attr in entity.attributes:
                params = dict(attr.params or {})
                target = params.get("target")
                if target and str(target) not in entity_ids:
                    continue
                if entity_key and entity_key.lower() == "subscriber_id" and attr.name == "customer_id":
                    continue
                if (
                    entity_key
                    and attr.name.lower() != entity_key.lower()
                    and entity.canonical_id == root_entity
                    and attr.name.endswith("_id")
                    and attr.name not in {"account_id"}
                ):
                    continue
                names.add(attr.name)
        return len(names)

    def _primary_key_for_entity(self, entity: EntityDef) -> str | None:
        preferred = f"{entity.canonical_id}_id"
        for attr in entity.attributes:
            if attr.name.lower() == preferred.lower():
                return attr.name
        for attr in entity.attributes:
            if attr.name.lower().endswith("_id"):
                return attr.name
        return None

    def _normalize_dependencies(
        self,
        entity: EntityDef,
        attr_name: str,
        dependencies: tuple[str, ...],
        entity_ids: set[str],
        field_names: set[str],
    ) -> list[str]:
        """Convert registry-level entity dependencies into executable field dependencies."""
        normalized: list[str] = []
        for dependency in dependencies:
            dep = str(dependency).strip()
            if not dep:
                continue
            if dep in field_names:
                normalized.append(dep)
                continue
            if dep in entity_ids:
                target = self.registry.get_entity(dep)
                target_key = self._primary_key_for_entity(target)
                if target_key and target_key in field_names:
                    normalized.append(target_key)
                continue
            # Unknown dependency text is not executable. Fail the proposal instead of
            # persisting a dangling dependency that can break /scenario/generate later.
            raise ValueError(
                f"Registry field '{entity.canonical_id}.{attr_name}' has unresolved dependency '{dependency}'"
            )
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _runtime_generation_contract(
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
            runtime_params["country_codes"] = [str(country or runtime_params.get("country") or "IN").upper()]
            return "e164_phone", runtime_params
        if generator == "timestamp":
            runtime_params.setdefault("days_back", 365)
            return "recent_datetime", runtime_params
        if generator == "range":
            return ("uniform_int" if dtype in {"int", "integer"} else "uniform"), runtime_params
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

    def compile(
        self,
        intent: ScenarioIntent,
        selected_entities: list[str] | None = None,
        min_variables: int = 35,
        domain_query: str | None = None,
        entity_key: str | None = None,
        industry_type: str | None = None,
        scenario_type: str | None = None,
        type_of_data: str | None = None,
        use_case: str | None = None,
        business_scenario: str | None = None,
        country: str | None = None,
    ) -> ScenarioSchema:
        requested = intent
        normalized_industry = (industry_type or intent.industry_type or "telecom").strip().lower()
        if normalized_industry not in {"telecom", "telecommunications"}:
            raise ValueError(f"Unsupported industryType '{industry_type or intent.industry_type}'. The telecom registry only supports Telecommunications/Telecom.")
        if selected_entities is not None:
            # HITL may select/add only entities already present in the authoritative registry.
            normalized: list[str] = []
            for value in selected_entities:
                entity = self.registry.resolve_entity(value)
                if entity is None:
                    raise ValueError(f"HITL selected unknown registry entity '{value}'")
                normalized.append(entity.canonical_id)
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
        domain_ids = {
            item["canonical_id"]
            for item in self.registry.search(domain_query or intent.domain, limit=50)
        }
        if domain_query and not domain_ids:
            unresolved.append(
                f"Business domain '{domain_query}' could not be mapped to an approved telecom registry domain"
            )
        selected_ids = set(entity_ids)
        if selected_ids & {"internet_access_service", "ip_uni", "subscriber_ethernet_service", "subscriber_uni"}:
            supporting_ids = {"subscriber_ethernet_service", "subscriber_uni", "subscriber", "product_offering", "customer_account"}
        elif selected_ids & {"recharge", "prepaid_account", "bucket", "balance_action_history", "usage_event", "charging_event"} or intent.subdomain == "prepaid":
            supporting_ids = {"subscriber", "customer_account", "usage_event", "charging_event", "online_charging_session", "cdr_record", "product_offering"}
        else:
            supporting_ids = {"subscriber", "product_offering", "customer_account", "customer"}
        allowed_expansion_ids = domain_ids | entity_ids | supporting_ids

        # Enforce the platform's minimum proposal width without allowing the LLM to
        # invent fields. Expand only through entities already present in the approved
        # registry graph, preferring nodes directly connected to the selected model.
        if domain_ids and min_variables > 0:
            frontier = list(entities)
            visited = set(entity_ids)
            while frontier and self._eligible_field_count(entities, entity_ids, entity_key) < min_variables:
                next_frontier: list[EntityDef] = []
                candidate_ids: list[str] = []
                for entity in frontier:
                    for rel in entity.relationships:
                        if rel.target not in visited and rel.target in allowed_expansion_ids and self.registry.entity_exists(rel.target):
                            candidate_ids.append(rel.target)
                    for candidate in self.registry.related_entities(entity.canonical_id):
                        if candidate.canonical_id not in visited and candidate.canonical_id in allowed_expansion_ids:
                            candidate_ids.append(candidate.canonical_id)
                # Stable de-duplication keeps schema generation reproducible. Rank only
                # already-approved registry nodes using the remaining business inputs.
                unique_candidate_ids = list(dict.fromkeys(candidate_ids))
                unique_candidate_ids.sort(
                    key=lambda cid: (
                        -self._expansion_score(
                            self.registry.get_entity(cid),
                            scenario_type=scenario_type,
                            type_of_data=type_of_data,
                            use_case=use_case,
                        ),
                        cid,
                    )
                )
                for candidate_id in unique_candidate_ids:
                    if candidate_id in visited:
                        continue
                    candidate = self.registry.get_entity(candidate_id)
                    visited.add(candidate_id)
                    entity_ids.add(candidate_id)
                    entities.append(candidate)
                    next_frontier.append(candidate)
                    if self._eligible_field_count(entities, entity_ids, entity_key) >= min_variables:
                        break
                frontier = next_frontier
                if not next_frontier:
                    break

        # Some valid telecom domains contain fewer than the platform minimum of 35
        # variables in the standards slice. Add only approved, domain-relevant
        # supporting entities. This fallback is deterministic and registry-backed;
        # it is never invented by the LLM.
        fallback_used = False
        if domain_ids and min_variables > 0 and self._eligible_field_count(entities, entity_ids, entity_key) < min_variables:
            if selected_ids & {"internet_access_service", "ip_uni", "subscriber_ethernet_service", "subscriber_uni"}:
                fallback_entities = (
                    "subscriber_ethernet_service", "subscriber_uni", "subscriber", "product_offering", "customer_account",
                )
            elif selected_ids & {"recharge", "prepaid_account", "bucket", "balance_action_history", "usage_event", "charging_event"} or intent.subdomain == "prepaid":
                fallback_entities = (
                    "customer_account", "usage_event", "charging_event", "online_charging_session", "cdr_record", "product_offering",
                )
            else:
                fallback_entities = (
                    "subscriber", "product_offering", "customer_account", "customer",
                    "prepaid_account", "usage_event", "charging_event",
                )

            for fallback_id in fallback_entities:
                if self._eligible_field_count(entities, entity_ids, entity_key) >= min_variables:
                    break
                if fallback_id in entity_ids or not self.registry.entity_exists(fallback_id):
                    continue
                entity = self.registry.get_entity(fallback_id)
                entities.append(entity)
                entity_ids.add(entity.canonical_id)
                fallback_used = True

        relationships: list[SchemaRelationship] = []
        for entity in entities:
            for rel in entity.relationships:
                if rel.target in entity_ids:
                    target = self.registry.get_entity(rel.target)
                    source_refs = [s.get("reference", "") for s in entity.sources]
                    relationships.append(SchemaRelationship(
                        source_entity=entity.canonical_id,
                        target_entity=target.canonical_id,
                        relation=rel.relation,
                        cardinality=rel.cardinality,
                        required=rel.required,
                        source_references=source_refs,
                    ))

        # Use only registry attributes. Scenario-specific fields can be added later as
        # explicit internal extensions through the registry/HITL, never by the LLM.
        profile = get_profile(normalized_industry, country)
        country_currency = str(profile.get("currency") or "").upper()
        fields: list[GeneratedSchemaField] = []
        seen_fields: set[str] = set()
        projected_field_names: set[str] = set()
        for entity in entities:
            for attr in entity.attributes:
                target = dict(attr.params or {}).get("target")
                if target and str(target) not in entity_ids:
                    continue
                if entity_key and entity_key.lower() == "subscriber_id" and attr.name == "customer_id":
                    continue
                if entity_key and attr.name != entity_key and entity.canonical_id == self._root_entity_for_key(entity_key, entities):
                    if attr.name.endswith("_id") and attr.name not in {"account_id"}:
                        continue
                projected_field_names.add(attr.name)

        for entity in entities:
            for attr in entity.attributes:
                if attr.name in seen_fields:
                    continue
                target = dict(attr.params or {}).get("target")
                if target and str(target) not in entity_ids:
                    continue
                if entity_key and entity_key.lower() == "subscriber_id" and attr.name == "customer_id":
                    continue
                if entity_key and attr.name != entity_key and entity.canonical_id == self._root_entity_for_key(entity_key, entities):
                    if attr.name.endswith("_id") and attr.name not in {"account_id"}:
                        continue
                params = dict(attr.params or {})
                if attr.generator == "msisdn" and country:
                    params["country"] = str(country).upper()
                if country_currency and "currency" in params:
                    params["currency"] = country_currency
                if country_currency and attr.generator == "constant" and attr.name.lower() == "currency":
                    params["value"] = country_currency
                runtime_generator, params = self._runtime_generation_contract(attr, params, country)
                if runtime_generator == "dependent_choice":
                    params["depends_on_field"] = (attr.depends_on[0] if attr.depends_on else "")
                dependencies = self._normalize_dependencies(
                    entity, attr.name, attr.depends_on, entity_ids, projected_field_names
                )
                # At this point the field is known to be projected: its reference target,
                # identity policy, generator contract and dependencies have all been checked.
                seen_fields.add(attr.name)
                fields.append(GeneratedSchemaField(
                    name=attr.name,
                    dtype=attr.dtype,
                    description=attr.description,
                    gen=runtime_generator,
                    params=params,
                    depends_on=dependencies,
                    nullable=attr.nullable,
                    required=attr.required,
                    formula=attr.derived_formula,
                    provenance={
                        "canonical_entity": entity.canonical_id,
                        "standards": [dict(s) for s in entity.sources],
                        "required": attr.required,
                    },
                ))

        hard_constraints = [
            "Every entity must resolve to an approved registry entry.",
            "Every field must resolve to an approved entity attribute.",
            "Every relationship must come from an approved registry edge.",
            "No LLM-created enum, formula, generator or field is executable.",
            "Generated records must pass deterministic schema, reference, temporal and arithmetic validation.",
            "Business domain selection is deterministic and registry-backed; industryType selects the supported model family.",
            "Scenario type, data type, country, entity key and use case are backend-authoritative inputs to proposal compilation; scenarioId is identifier-only.",
        ]
        if not entities:
            unresolved.append("No supported telecom concepts could be resolved from the request")

        if "recharge" in entity_ids and "prepaid_account" in entity_ids:
            hard_constraints.append("Successful recharge affects prepaid balance; failed/reversed recharge must not be applied as a completed top-up.")
        if "usage_event" in entity_ids and "charging_event" in entity_ids:
            hard_constraints.append("Charging event must reference an existing usage event and occur at or after the usage event timestamp.")
        if type_of_data == "aggregational":
            hard_constraints.append("Aggregational output is selected using the requested data type while retaining only registry-backed attributes.")
        if type_of_data == "transactional":
            hard_constraints.append("Transactional output is compiled around the requested entity key and registry-backed identifiers/events.")
        if country:
            hard_constraints.append(f"Country context '{country}' is applied to country-sensitive generation parameters only.")
        if scenario_type:
            hard_constraints.append(f"Scenario type '{scenario_type}' is treated as a backend-authoritative scenario context; no unregistered semantics are created from it.")
        applicable_standards = self.registry.standards_for_entities([e.canonical_id for e in entities])
        warnings = []
        if entities:
            warnings.append("Runtime registry is built from versioned normalized standards artifacts plus an INGENII generation-policy overlay. Official source artifacts should be re-ingested before production certification.")
        if fallback_used:
            warnings.append("The requested domain did not contain enough registry-backed attributes to meet the 35-variable minimum; approved foundational telecom entities were added deterministically.")
        return ScenarioSchema(
            domain=domain_query or intent.domain or intent.subdomain,
            subdomain=intent.subdomain,
            applicable_standards=applicable_standards,
            entities=[ResolvedConcept(canonical_id=e.canonical_id, name=e.name, source_model="/".join(sorted({s['standard'] for s in e.sources})), source_references=[s.get("reference", "") for s in e.sources], selected_attributes=[a.name for a in e.attributes]) for e in entities],
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
