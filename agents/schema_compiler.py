"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations


from core.agentic_models import GeneratedSchemaField, ResolvedConcept, ScenarioIntent, ScenarioSchema, SchemaRelationship
from core.telecom_registry import EntityDef, TelecomRegistry


class SchemaCompiler:
    """Compiles a scenario schema only from the runtime standards registry."""

    def __init__(self, registry: TelecomRegistry | None = None):
        self.registry = registry or TelecomRegistry()

    def resolve(self, intent: ScenarioIntent) -> tuple[list[EntityDef], list[str]]:
        resolved: list[EntityDef] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        for requested in intent.requested_entities:
            entity = self.registry.resolve_entity(requested)
            if entity is None:
                unresolved.append(f"Unknown concept '{requested}' is not in the approved telecom registry")
                continue
            if entity.canonical_id not in seen:
                resolved.append(entity)
                seen.add(entity.canonical_id)
        # Prepaid scenarios need a minimal anchor graph. Add anchors only when the
        # runtime registry contains them; never manufacture missing model concepts.
        if intent.subdomain == "prepaid":
            needs_anchors = not resolved or any(
                entity.canonical_id in {"recharge", "usage_event", "charging_event"}
                for entity in resolved
            )
            if needs_anchors:
                for anchor in ("subscriber", "customer_account", "prepaid_account"):
                    if anchor not in seen and self.registry.entity_exists(anchor):
                        resolved.append(self.registry.get_entity(anchor))
                        seen.add(anchor)

        return resolved, unresolved

    def compile(self, intent: ScenarioIntent, selected_entities: list[str] | None = None) -> ScenarioSchema:
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
        entities, unresolved = self.resolve(requested)
        entity_ids = {e.canonical_id for e in entities}

        # Add relationship endpoints needed by selected entities, but only from registry edges.
        changed = True
        while changed:
            changed = False
            for entity in list(entities):
                for rel in entity.relationships:
                    if rel.required and rel.target not in entity_ids and self.registry.entity_exists(rel.target):
                        entities.append(self.registry.get_entity(rel.target))
                        entity_ids.add(rel.target)
                        changed = True

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
                seen_fields.add(attr.name)
                fields.append(GeneratedSchemaField(
                    name=attr.name,
                    dtype=attr.dtype,
                    description=attr.description,
                    gen=attr.generator,
                    params=dict(attr.params or {}),
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
        ]
        if not entities:
            unresolved.append("No supported telecom concepts could be resolved from the request")

        if "recharge" in entity_ids and "prepaid_account" in entity_ids:
            hard_constraints.append("Successful recharge affects prepaid balance; failed/reversed recharge must not be applied as a completed top-up.")
        if "usage_event" in entity_ids and "charging_event" in entity_ids:
            hard_constraints.append("Charging event must reference an existing usage event and occur at or after the usage event timestamp.")
        applicable_standards = self.registry.standards_for_entities([e.canonical_id for e in entities])
        return ScenarioSchema(
            domain="telecom",
            subdomain=intent.subdomain,
            applicable_standards=applicable_standards,
            entities=[ResolvedConcept(canonical_id=e.canonical_id, name=e.name, source_model="/".join(sorted({s['standard'] for s in e.sources})), source_references=[s.get("reference", "") for s in e.sources], selected_attributes=[a.name for a in e.attributes]) for e in entities],
            relationships=relationships,
            fields=fields,
            hard_constraints=hard_constraints,
            unresolved_items=unresolved,
            warnings=["Runtime registry is built from versioned normalized standards artifacts plus an INGENII generation-policy overlay. Official source artifacts should be re-ingested before production certification."] if entities else [],
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
