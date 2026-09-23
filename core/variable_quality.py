"""Scenario-variable quality scoring, semantic de-duplication and selection.

The schema compiler may discover a large standards-backed candidate set. This module keeps
that breadth while preventing API/transport metadata, duplicate semantic concepts, and
low-information fields from consuming the executable variable budget.

The policy is intentionally deterministic: no LLM is consulted during ranking or filtering.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable


def _norm(value: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(t) > 1}


@dataclass(frozen=True)
class CandidateScore:
    score: float
    semantic_key: str
    reasons: tuple[str, ...]


class VariableQualityEngine:
    """Deterministic quality policy shared by all schema-compilation paths."""

    DEFAULT_MAX_VARIABLES = 100
    DEFAULT_MIN_SCORE = 42.0
    SECONDARY_ENTITY_PENALTY = 40.0

    # Technical/API metadata is useful for API fidelity, but generally has low analytical value.
    LOW_VALUE_SUFFIXES = {
        "href", "referred_type", "schema_location", "base_type",
    }
    LOW_VALUE_TOKENS = {
        "href", "referred", "reference", "uri", "schema", "transport",
    }
    DISPLAY_TOKENS = {"description", "display", "label", "formatted", "friendly"}
    RELATION_METADATA_TOKENS = {
        "party_account", "engaged_party", "related_party", "requestor",
        "referred_type", "balance_topup",
    }

    # Cross-model synonyms that frequently produce duplicate columns after flattening.
    SYNONYMS = {
        "identifier": "id",
        "key": "id",
        "lifecycle_state": "status",
        "lifecycle_status": "status",
        "state": "status",
        "is_shared_flag": "shared",
        "shared_flag": "shared",
        "requested_date_time": "requested_timestamp",
        "requested_datetime": "requested_timestamp",
        "confirmation_date_time": "confirmation_timestamp",
        "confirmation_datetime": "confirmation_timestamp",
        "valid_for_start_date_time": "valid_from",
        "valid_for_end_date_time": "valid_to",
    }

    # Generic technical prefixes introduced by flattened API paths. They should not force two
    # fields to be considered different concepts when the business concept is identical.
    ENTITY_PREFIXES = {
        "bucket", "topupbalance", "topup_balance", "customer", "subscriber",
        "account", "party", "payment", "channel",
    }

    ENTITY_CONTEXT_ALIASES = {
        "topupbalance": {"topup", "top_up", "balance", "recharge"},
        "topup_balance": {"topup", "top_up", "balance", "recharge"},
        "bucket": {"bucket"},
        "customer": {"customer", "subscriber", "account", "party"},
        "subscriber": {"subscriber", "customer", "account"},
        "account": {"account", "subscriber", "customer"},
    }

    HIGH_VALUE_ROLES = {
        "identity": 12.0,
        "event": 13.0,
        "transaction": 12.0,
        "status": 14.0,
        "decision": 15.0,
        "measurement": 14.0,
        "metric": 14.0,
        "timing": 13.0,
        "configuration": 9.0,
        "derived": 15.0,
        "profile": 7.0,
        "other": 1.0,
    }

    IMPORTANT_NAME_TOKENS = {
        "amount", "balance", "quota", "threshold", "timestamp", "date", "time",
        "status", "state", "outcome", "decision", "result", "reason", "channel",
        "payment", "method", "plan", "validity", "latency", "duration", "usage",
        "eligibility", "suppression", "intervention", "segment", "region", "service",
        "event", "transaction", "request", "confirmation", "activation", "expiry",
    }

    def __init__(self, max_variables: int | None = None, min_score: float | None = None):
        self.max_variables = max(
            1,
            int(max_variables if max_variables is not None else self.DEFAULT_MAX_VARIABLES),
        )
        self.min_score = float(
            min_score if min_score is not None else self.DEFAULT_MIN_SCORE
        )

    @classmethod
    def semantic_key(cls, idea: dict[str, Any]) -> str:
        """Return a conservative business-concept key used to collapse true semantic aliases.

        The entity/object path is intentionally preserved. For example, ``bucket_id`` and
        ``topupbalance_channel_id`` are different concepts even though both end in ``_id``.
        Only well-known aliases in the same object path are normalized.
        """
        raw = _norm(str(idea.get("name") or "field"))
        source_spec = idea.get("_json_source_spec")
        if isinstance(source_spec, dict):
            source_path = _norm(str(source_spec.get("path") or source_spec.get("name") or ""))
            if source_path:
                raw = source_path

        replacements = (
            ("_lifecycle_state", "_status"),
            ("_lifecycle_status", "_status"),
            ("_is_shared_flag", "_is_shared"),
            ("_shared_flag", "_shared"),
            ("_requested_date_time", "_requested_timestamp"),
            ("_requested_datetime", "_requested_timestamp"),
            ("_confirmation_date_time", "_confirmation_timestamp"),
            ("_confirmation_datetime", "_confirmation_timestamp"),
            ("_valid_for_start_date_time", "_valid_from"),
            ("_valid_for_end_date_time", "_valid_to"),
            ("_identifier", "_id"),
        )
        for old, new in replacements:
            if raw.endswith(old):
                raw = raw[: -len(old)] + new
                break

        # A small set of exact application identity fields remain independent even when they
        # share common suffixes such as ``_id``.
        identity_name = _norm(str(idea.get("name") or ""))
        if identity_name in {"subscriber_id", "account_id", "customer_id", "msisdn", "user_id", "entity_id"}:
            return identity_name

        return raw

    @classmethod
    def _is_forced(cls, idea: dict[str, Any], entity_key: str | None) -> bool:
        name = _norm(str(idea.get("name") or ""))
        return bool(
            name == _norm(entity_key or "")
            or name in {"subscriber_id", "account_id", "msisdn"}
            or bool(idea.get("_force_include"))
        )

    @classmethod
    def _metadata_penalty(cls, name: str, description: str) -> tuple[float, list[str]]:
        n = _norm(name)
        d = str(description or "").lower()
        tokens = _tokens(f"{n} {d}")
        penalty = 0.0
        reasons: list[str] = []

        if n.endswith(tuple(f"_{suffix}" for suffix in cls.LOW_VALUE_SUFFIXES)):
            penalty += 23.0
            reasons.append("technical_reference_metadata")
        elif any(token in tokens for token in {"href", "uri", "referred"}):
            penalty += 18.0
            reasons.append("technical_reference_metadata")

        if any(token in n for token in cls.RELATION_METADATA_TOKENS):
            penalty += 14.0
            reasons.append("secondary_relationship_metadata")

        # Long API prose / display fields are less useful when they only reproduce descriptive
        # metadata already represented by a categorical, amount, state, or identifier column.
        if any(tok in tokens for tok in cls.DISPLAY_TOKENS):
            penalty += 7.0
            reasons.append("display_or_descriptive_field")
        if n.endswith("_description") or n.endswith("_formatted") or n.endswith("_remaining_value_name"):
            penalty += 8.0
            reasons.append("display_only_column")
        if n.endswith("_units") or n.endswith("_unit"):
            penalty += 7.0
            reasons.append("denomination_support_column")
        if "disambiguation" in d or "disambiguate" in d:
            penalty += 10.0
            reasons.append("api_disambiguation_field")

        return penalty, reasons

    @classmethod
    def _role_mismatch_penalty(cls, idea: dict[str, Any]) -> tuple[float, list[str]]:
        name = _norm(str(idea.get("name") or ""))
        description = str(idea.get("description") or "").lower()
        role = str(idea.get("role") or "other").lower()
        dtype = str(idea.get("dtype") or "string").lower()
        text = f"{name} {description}"
        penalty = 0.0
        reasons: list[str] = []

        if role in {"status", "decision", "derived"}:
            if not any(t in text for t in ("status", "state", "outcome", "decision", "result", "reason", "eligible", "suppression")):
                penalty += 12.0
                reasons.append("semantic_role_mismatch")
            if dtype in {"string", "object"} and not any(t in text for t in ("name", "description", "reason", "message")):
                penalty += 4.0
                reasons.append("weak_status_contract")
        elif role in {"measurement", "metric"}:
            if dtype not in {"integer", "float", "decimal", "number", "numeric"}:
                penalty += 15.0
                reasons.append("measurement_not_numeric")
        elif role == "timing":
            if dtype not in {"datetime", "date", "timestamp"}:
                penalty += 15.0
                reasons.append("timing_not_datetime")

        # A variable explicitly called an outcome/decision should not be satisfied by a generic
        # usage/unit/type field merely because their tokenization overlaps. This is a common
        # source of semantically incorrect enum reuse in flattened API models.
        if any(t in name.split("_") for t in ("outcome", "decision", "result")):
            if any(t in text for t in ("usage_type", "unit", "units", "href", "referred_type")):
                penalty += 20.0
                reasons.append("outcome_cross_domain_match")
        return penalty, reasons

    def score(self, idea: dict[str, Any], context_text: str, entity_key: str | None) -> CandidateScore:
        name = str(idea.get("name") or "")
        description = str(idea.get("description") or "")
        role = str(idea.get("role") or "other").lower()
        forced = self._is_forced(idea, entity_key)
        text_tokens = _tokens(f"{name} {description}")
        context_tokens = _tokens(context_text)
        score = 35.0
        reasons: list[str] = []

        if forced:
            score += 100.0
            reasons.append("forced_business_contract")

        score += self.HIGH_VALUE_ROLES.get(role, 0.0)
        if role in self.HIGH_VALUE_ROLES and self.HIGH_VALUE_ROLES[role] >= 12:
            reasons.append("analytical_role")

        important_overlap = len(text_tokens & context_tokens & self.IMPORTANT_NAME_TOKENS)
        general_overlap = len(text_tokens & context_tokens)
        score += min(18.0, important_overlap * 4.0)
        score += min(8.0, max(0, general_overlap - important_overlap) * 1.5)
        if general_overlap:
            reasons.append("scenario_relevance")

        registry_entity = _norm(str(idea.get("_registry_entity_name") or idea.get("_registry_entity") or ""))
        entity_tokens = set(registry_entity.split("_")) if registry_entity else set()
        entity_aliases = set(entity_tokens)
        for token in list(entity_tokens):
            entity_aliases.update(self.ENTITY_CONTEXT_ALIASES.get(token, set()))
        if entity_aliases:
            entity_overlap = len(entity_aliases & context_tokens)
            if entity_overlap:
                score += min(8.0, entity_overlap * 2.5)
                reasons.append("entity_context_relevance")
            elif idea.get("_registry_entity_name") and not forced:
                # Related official entities can be valid context, but they should not crowd out
                # scenario-critical fields merely because their fields are scalar and well-typed.
                score -= self.SECONDARY_ENTITY_PENALTY
                reasons.append("secondary_entity_penalty")

        meaningful_dependencies = [
            str(dep) for dep in (idea.get("depends_on") or [])
            if _norm(str(dep)) != _norm(entity_key or "")
        ]
        if meaningful_dependencies:
            score += 5.0
            reasons.append("relational_dependency")

        if idea.get("_registry_required"):
            score += 18.0
            reasons.append("officially_required")

        if idea.get("_json_source_spec"):
            score += 6.0
            reasons.append("official_grounding")
            depth = int(idea.get("_json_source_spec", {}).get("depth", 0) or 0) if isinstance(idea.get("_json_source_spec"), dict) else 0
            if depth >= 3:
                score -= 10.0
                reasons.append("deep_nested_source_penalty")
            elif depth >= 2:
                score -= 5.0
                reasons.append("nested_source_penalty")
        elif idea.get("_registry_exact"):
            score += 5.0
            reasons.append("registry_grounding")

        # Strongly favor variables that can be used in behavioral analysis.
        if any(t in text_tokens for t in {"amount", "balance", "threshold", "timestamp", "status", "outcome", "decision", "channel", "plan", "latency"}):
            score += 8.0
            reasons.append("behavioral_signal")

        metadata_penalty, metadata_reasons = self._metadata_penalty(name, description)
        score -= metadata_penalty
        reasons.extend(metadata_reasons)

        mismatch_penalty, mismatch_reasons = self._role_mismatch_penalty(idea)
        score -= mismatch_penalty
        reasons.extend(mismatch_reasons)

        # Open-ended semantic strings have less analytical value than concrete numeric/category/time
        # fields unless their role explicitly makes them meaningful.
        gen = str(idea.get("gen") or "").lower()
        if gen in {"semantic_string", "generic"} and role == "other":
            score -= 8.0
            reasons.append("low_information_string")

        return CandidateScore(score=score, semantic_key=self.semantic_key(idea), reasons=tuple(reasons))

    def select(
        self,
        ideas: Iterable[dict[str, Any]],
        *,
        context_text: str,
        entity_key: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select the widest high-quality set while preserving dependency closure."""
        ideas_list = list(ideas)
        scored: list[tuple[float, int, dict[str, Any], CandidateScore]] = []
        for index, raw in enumerate(ideas_list):
            details = self.score(raw, context_text, entity_key)
            scored.append((details.score, index, raw, details))

        grouped: dict[str, list[tuple[float, int, dict[str, Any], CandidateScore]]] = {}
        for item in scored:
            grouped.setdefault(item[3].semantic_key, []).append(item)

        representatives: list[tuple[float, int, dict[str, Any], CandidateScore]] = []
        representative_by_name: dict[str, tuple[float, int, dict[str, Any], CandidateScore]] = {}
        alias_to_representative: dict[str, str] = {}
        duplicate_count = 0
        def _representative_sort_key(item):
            _score, index, idea, _details = item
            explicit_force = 1 if idea.get("_force_include") else 0
            official = 1 if (idea.get("_json_source_spec") or idea.get("_registry_exact")) else 0
            return (-explicit_force, -official, -_score, index)

        for semantic_key, group_items in grouped.items():
            # When candidates are semantic aliases, prefer an authoritative registry/Swagger
            # representation over an LLM-created alias unless the alias was explicitly forced.
            group_items.sort(key=_representative_sort_key)
            best = group_items[0]
            representatives.append(best)
            representative_by_name[_norm(str(best[2].get("name") or ""))] = best
            canonical_name = str(best[2].get("name") or "")
            for _score, _index, idea, _details in group_items:
                alias_to_representative[_norm(str(idea.get("name") or ""))] = canonical_name
            duplicate_count += max(0, len(group_items) - 1)

        representatives.sort(key=lambda item: (-item[0], item[1]))
        selected_by_semantic: dict[str, tuple[float, int, dict[str, Any], CandidateScore]] = {}
        required_semantics: set[str] = set()
        rejected_low_score = 0
        truncated = 0

        for score, _index, idea, details in representatives:
            forced = self._is_forced(idea, entity_key)
            if forced:
                selected_by_semantic[details.semantic_key] = (score, _index, idea, details)
                required_semantics.add(details.semantic_key)
                continue
            if score < self.min_score:
                rejected_low_score += 1
                continue
            if len(selected_by_semantic) >= self.max_variables:
                truncated += 1
                continue
            selected_by_semantic[details.semantic_key] = (score, _index, idea, details)

        # Dependency closure: if a selected field references another candidate concept, retain
        # the best canonical representative for that dependency even if it is otherwise below
        # the quality floor. This prevents quality filtering from silently removing a prerequisite.
        name_to_candidate = {
            _norm(str(idea.get("name") or "")): (score, index, idea, details)
            for score, index, idea, details in scored
            if str(idea.get("name") or "").strip()
        }
        dependency_forced = 0
        changed = True
        while changed:
            changed = False
            current_items = list(selected_by_semantic.values())
            for _score, _index, idea, _details in current_items:
                for raw_dep in idea.get("depends_on") or []:
                    dep_key = _norm(str(raw_dep or ""))
                    if not dep_key:
                        continue
                    candidate_item = name_to_candidate.get(dep_key)
                    if candidate_item is None:
                        continue
                    canonical_name = alias_to_representative.get(dep_key, str(candidate_item[2].get("name") or ""))
                    rep = representative_by_name.get(_norm(canonical_name)) or candidate_item
                    semantic_key = rep[3].semantic_key
                    if semantic_key in selected_by_semantic:
                        required_semantics.add(semantic_key)
                        continue
                    selected_by_semantic[semantic_key] = rep
                    required_semantics.add(semantic_key)
                    dependency_forced += 1
                    changed = True

        # The maximum is a preference. If dependency closure pushed us over the budget, evict
        # only non-required, lowest-quality representatives; never break the dependency graph.
        if len(selected_by_semantic) > self.max_variables:
            removable = sorted(
                (
                    item for key, item in selected_by_semantic.items()
                    if key not in required_semantics and not self._is_forced(item[2], entity_key)
                ),
                key=lambda item: (item[0], item[1]),
            )
            while len(selected_by_semantic) > self.max_variables and removable:
                victim = removable.pop(0)
                selected_by_semantic.pop(victim[3].semantic_key, None)
                truncated += 1

        selected = [item[2] for item in selected_by_semantic.values()]
        selected_ids = {id(item) for item in selected}
        ordered = [idea for idea in ideas_list if id(idea) in selected_ids]
        selected_canonical_names = {str(item.get("name") or "") for item in ordered}

        # Keep aliases only when their canonical representative was actually selected. The
        # compiler uses this map to rewrite dependencies from a dropped alias (e.g. bucket_identifier)
        # to the retained canonical field (e.g. bucket_id).
        dependency_aliases = {
            alias: canonical
            for alias, canonical in alias_to_representative.items()
            if canonical in selected_canonical_names
        }

        report = {
            "candidate_count": len(scored),
            "selected_count": len(ordered),
            "maximum": self.max_variables,
            "minimum_quality_score": self.min_score,
            "semantic_duplicates_removed": duplicate_count,
            "low_quality_candidates_removed": rejected_low_score,
            "quality_budget_truncated": truncated,
            "dependency_closure_added": dependency_forced,
            "dependency_aliases": dependency_aliases,
        }
        return ordered, report
