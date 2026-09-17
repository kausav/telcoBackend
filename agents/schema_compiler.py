"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship
from core.telecom_registry import EntityDef, TelecomRegistry


class SchemaCompiler:
    """Compiles a scenario schema only from the runtime standards registry."""

    def __init__(self, registry: TelecomRegistry | None = None):
        self.registry = registry or TelecomRegistry()

    def resolve(self, intent: ScenarioIntent, domain_query: str | None = None) -> tuple[list[EntityDef], list[str]]:
        resolved: list[EntityDef] = []
        unresolved: list[str] = []
        seen: set[str] = set()

        # `domain` is the primary selector inside the chosen industry. The registry
        # deterministically retrieves entities whose names/aliases match the business domain.
        domain_candidates = self.registry.search(domain_query or intent.domain, limit=12)
        domain_ids = {candidate["canonical_id"] for candidate in domain_candidates}
        for candidate in domain_candidates:
            entity = self.registry.resolve_entity(candidate["canonical_id"])
            if entity and entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)

        # The LLM may refine the request, but only concepts already present in the
        # approved domain catalog are eligible. This keeps domain selection deterministic.
        for requested in intent.requested_entities:
            entity = self.registry.resolve_entity(requested)
            if entity is None:
                unresolved.append(f"Unknown concept '{requested}' is not in the approved telecom registry")
                continue
            if entity.canonical_id not in domain_ids:
                continue
            if entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)

        # Prepaid is subscriber-centric in the platform contract. Add the requested
        # entity key anchor only from the approved registry; do not automatically pull
        # customer/account master-data entities just because a standards relationship is required.
        if intent.subdomain == "prepaid" and self.registry.entity_exists("subscriber"):
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

    def compile(
        self,
        intent: ScenarioIntent,
        selected_entities: list[str] | None = None,
        min_variables: int = 20,
        domain_query: str | None = None,
        entity_key: str | None = None,
    ) -> ScenarioSchema:
        requested = intent
        if selected_entities is not None:
            # HITL may select/add only entities already present in the authoritative registry.
            normalized: list[str] = []
            for value in selected_entities:
                entity = self.registry.resolve_entity(value)
                if entity is None:
                    raise ValueError(f"HITL selected unknown registry entity '{value}'")
                normalized.append(entity.canonical_id)
            requested = requested.model_copy(update={"requested_entities": normalized})
        entities, unresolved = self.resolve(requested, domain_query=domain_query)
        entity_ids = {e.canonical_id for e in entities}
        domain_ids = {item["canonical_id"] for item in self.registry.search(domain_query or intent.domain, limit=12)}
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
        allowed_expansion_ids = domain_ids | supporting_ids

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
                # Stable de-duplication keeps schema generation reproducible.
                for candidate_id in dict.fromkeys(candidate_ids):
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

        # Some valid telecom domains contain fewer than the platform minimum of 20
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
        fields: list[GeneratedSchemaField] = []
        seen_fields: set[str] = set()
        for entity in entities:
            for attr in entity.attributes:
                if attr.name in seen_fields:
                    continue
                params = dict(attr.params or {})
                # Do not expose dangling foreign-key/reference fields when the referenced
                # master entity is not part of the selected scenario. This prevents a
                # subscriber-level scenario from unnecessarily producing both subscriber_id
                # and unrelated customer/account identity columns.
                target = params.get("target")
                if target and str(target) not in entity_ids:
                    continue
                # Subscriber-centric prepaid responses keep subscriber_id and account_id
                # as the two logical identity fields. customer_id is intentionally omitted
                # because it is not required by the requested subscriber-level contract.
                if entity_key and entity_key.lower() == "subscriber_id" and attr.name == "customer_id":
                    continue
                if entity_key and attr.name != entity_key and entity.canonical_id == self._root_entity_for_key(entity_key, entities):
                    if attr.name.endswith("_id") and attr.name not in {"account_id"}:
                        continue
                seen_fields.add(attr.name)
                fields.append(GeneratedSchemaField(
                    name=attr.name,
                    dtype=attr.dtype,
                    description=attr.description,
                    gen=attr.generator,
                    params=params,
                    depends_on=list(attr.depends_on),
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
        ]
        if not entities:
            unresolved.append("No supported telecom concepts could be resolved from the request")

        if "recharge" in entity_ids and "prepaid_account" in entity_ids:
            hard_constraints.append("Successful recharge affects prepaid balance; failed/reversed recharge must not be applied as a completed top-up.")
        if "usage_event" in entity_ids and "charging_event" in entity_ids:
            hard_constraints.append("Charging event must reference an existing usage event and occur at or after the usage event timestamp.")
        applicable_standards = self.registry.standards_for_entities([e.canonical_id for e in entities])
        warnings = []
        if entities:
            warnings.append("Runtime registry is built from versioned normalized standards artifacts plus an INGENII generation-policy overlay. Official source artifacts should be re-ingested before production certification.")
        if fallback_used:
            warnings.append("The requested domain did not contain enough registry-backed attributes to meet the 20-variable minimum; approved foundational telecom entities were added deterministically.")
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
