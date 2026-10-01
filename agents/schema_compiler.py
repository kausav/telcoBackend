"""Deterministic schema compiler and HITL proposal builder."""
from __future__ import annotations

import logging
from typing import Any

from core.agentic_models import (
    GeneratedSchemaField,
    ResolvedConcept,
    ScenarioIntent,
    ScenarioSchema,
)
import math
import re
from core.json_domain_policy import is_json_grounded_domain
from core import lexicon
from core.industry_source_store import (
    _canonical_context_tokens,
    catalog_for_request,
    normalize_industry_key,
    normalize_lookup_key,
    semantic_exclusion_aliases,
    _catalog_role,
)
from core.variable_quality import VariableQualityEngine
from core.scenario_planner import owning_models, select_source_rows
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

    SOURCE_RUNTIME_NAME_MAX_LENGTH = 120
    _GENERIC_SOURCE_LEAF_TOKENS = {
        "id", "key", "name", "label", "type", "status", "state", "reason", "value",
        "amount", "unit", "date", "time", "datetime", "timestamp", "role", "code",
        "description", "href", "url", "duration", "action", "result", "category", "method",
        "channel", "price", "period", "count", "number", "reference", "identifier",
        "visible", "shared", "enabled", "active", "available", "valid",
    }
    _SOURCE_BOOLEAN_PREFIXES = {"is", "has", "can", "should"}


    @classmethod
    def _source_readable_names(
        cls,
        fields: list[GeneratedSchemaField],
        all_source_specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Preserve stable source field names in the public scenario contract.

        MongoDB source names are the authoritative client vocabulary. Earlier implementations
        rebuilt names from flattened semantic paths and added hashes on collisions, which caused
        stable source fields such as ``adjust_balance_amount_amount`` to drift between proposals.
        Keep the exact source field name whenever it is already a valid executable identifier.
        Ambiguous names fail closed; never expose a run-specific hash suffix.
        """
        source_fields = [
            field for field in fields
            if str((field.provenance or {}).get("generated_from") or "").strip().lower() == "mongodb_json_source"
        ]
        if not source_fields:
            return {}

        replacements: dict[str, str] = {}
        used: dict[str, str] = {}
        for field in sorted(
            source_fields,
            key=lambda item: str((item.provenance or {}).get("source_json_semantic_key") or item.name).casefold(),
        ):
            provenance = field.provenance or {}
            original = str(
                provenance.get("source_json_original_name")
                or provenance.get("source_json_name")
                or field.name
            ).strip()
            candidate = cls._normalize_variable_key(original)
            if not candidate:
                raise ValueError("Source field has no usable canonical output name")
            # Source contracts are authoritative and must not silently truncate public names.
            semantic = str(provenance.get("source_json_semantic_key") or original).strip()
            existing = used.get(candidate)
            if existing and existing != semantic.casefold():
                raise ValueError(
                    f"Source schema contains duplicate public field name '{candidate}' for semantic paths "
                    f"'{existing}' and '{semantic}'. Resolve the source semantic identity before proposing the scenario."
                )
            used[candidate] = semantic.casefold()
            replacements[cls._normalize_variable_key(field.name)] = candidate
        return replacements

    @classmethod
    def _compact_source_field_names(cls, fields: list[GeneratedSchemaField], all_source_specs: list[dict[str, Any]] | None = None) -> list[GeneratedSchemaField]:
        """Apply compact source-derived names while retaining the full original JSON identity."""
        if not fields:
            return fields
        replacements = cls._source_readable_names(fields, all_source_specs=all_source_specs)
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
        # important for other sources where a field named ``id`` may legally be a UUID/URI.
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
            source_examples = []
            raw_examples = spec.get("examples")
            if isinstance(raw_examples, (list, tuple)):
                source_examples.extend(value for value in raw_examples if isinstance(value, str))
            elif isinstance(raw_examples, str) and raw_examples.strip():
                source_examples.append(raw_examples.strip())
            if isinstance(spec.get("example"), str) and spec.get("example").strip():
                source_examples.append(spec.get("example").strip())
            if isinstance(spec.get("default"), str) and spec.get("default").strip():
                source_examples.append(spec.get("default").strip())
            source_examples = list(dict.fromkeys(source_examples))[:20]
            if source_examples:
                string_params["source_examples"] = source_examples
            # Every MongoDB-backed JSON string is source-contract-bound. There is no
            # filesystem/vocabulary fallback for an industry source field.
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
        # in their descriptions (for example RelatedOrder.role = parent/child).
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
        all_source_specs: list[dict[str, Any]] | None = None,
    ) -> list[GeneratedSchemaField]:
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
        # ``extra_charge_amount`` can intentionally coexist with their source-backed fields.
        # Fuzzy suppression previously removed legitimate scenario variables and narrowed the
        # generated schema below the requested business scope.
        for idea in raw_ideas:
            add_idea(idea)
        if entity_key:
            add_idea({
                "name": entity_key,
                "description": f"Stable identifier for the requested {intent.use_case or intent.domain or 'business'} entity.",
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
        return self._compact_source_field_names(fields, all_source_specs=all_source_specs)

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
        forced_names: set[str] | None = None,
    ) -> ScenarioSchema:
        """Compile any MongoDB-backed standards domain without assuming any industry's semantics."""
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
            requested_key = self._normalize_variable_key(entity_key) if str(entity_key or "").strip() else ""
            if requested_key:
                if requested_key not in {self._normalize_variable_key(row.get("name")) for row in catalog_rows} and requested_key not in external_keys:
                    raise ValueError(
                        f"Requested entity key '{entity_key}' is not present in the active JSON source catalog or MongoDB variables."
                    )
                resolved_entity_key = entity_key.strip()
            else:
                entity_candidates = [
                    row for row in catalog_rows
                    if normalize_lookup_key(row.get("name")) in set(all_identity_names)
                    and _is_entity_catalog_row(row)
                ]
                entity_candidates.sort(key=lambda row: (
                    0 if any(token in normalize_lookup_key(row.get("name")) for token in ("customer", "subscriber", "user", "member", "patient")) else 1,
                    0 if str(row.get("required")) == "True" else 1,
                    normalize_lookup_key(row.get("name")),
                ))
                if entity_candidates:
                    resolved_entity_key = str(entity_candidates[0].get("name"))
        if resolved_entity_key:
            preferred_names.add(normalize_lookup_key(resolved_entity_key))

        # Source relevance is intentionally recall-oriented. Give the quality policy a larger
        # candidate pool, then let deterministic scoring keep the best source-backed concepts up to
        # the requested budget. This preserves the variable ceiling while preferring useful fields
        # over filling the budget with low-information metadata.
        candidate_pool_budget = min(len(catalog_rows), max(variable_budget, 1)) if variable_budget else 0
        selected_rows, source_selection_report = select_source_rows(
            catalog_rows,
            context=_canonical_context_tokens(
                set(re.findall(r"[a-z0-9]+", business_context.casefold())) - {"and", "the", "of", "to", "in", "for", "with", "a", "an", "is", "are"},
                lexicon.load(normalize_industry_key(industry_type) if industry_type else None),
            ),
            preferred_names=preferred_names,
            preferred_models={str(value) for value in (intent.requested_entities or []) if str(value).strip()},
            excluded_names=excluded,
            excluded_semantic_keys=external_semantic_aliases,
            forced_names=forced_names,
            owner_models=owning_models(external_variable_definitions, catalog_rows),
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
                "_force_include": bool(spec.get("required")) or normalize_lookup_key(spec.get("name")) in {normalize_lookup_key(n) for n in (forced_names or ())} or (
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
            all_source_specs=catalog_rows,
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
        forced_names: set[str] | None = None,
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
            forced_names=forced_names,
        )

    def approval_questions(self, intent: ScenarioIntent, schema: ScenarioSchema) -> list[str]:
        questions = list(intent.ambiguities)
        if intent.record_count is None:
            questions.append("How many records/entities should be generated?")
        if intent.country is None:
            questions.append("Which country/market should be modeled?")
        return questions
