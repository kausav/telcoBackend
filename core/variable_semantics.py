"""Shared deterministic variable-name semantics.

This module contains only domain-neutral normalization used to identify variables that are
structurally the same concept even when producers flatten or rename them differently.
It deliberately avoids generic suffix aliases (for example treating every ``*_id`` as the
same field), because those cause false-positive deduplication across unrelated entities.
"""
from __future__ import annotations

import re
from typing import Any


_SYNONYMS = (
    ("lifecycle_state", "status"),
    ("lifecycle_status", "status"),
    ("identifier", "id"),
    ("requested_date_time", "requested_timestamp"),
    ("requested_datetime", "requested_timestamp"),
    ("confirmation_date_time", "confirmation_timestamp"),
    ("confirmation_datetime", "confirmation_timestamp"),
    ("valid_for_start_date_time", "valid_from"),
    ("valid_for_end_date_time", "valid_to"),
    ("shared_flag", "shared"),
    ("is_shared_flag", "is_shared"),
    ("automatic", "auto"),
)


def normalize_variable_name(value: Any) -> str:
    """Normalize separators/casing and collapse harmless structural aliases.

    Examples:
      ``product_product_id`` -> ``product_id``
      ``product_identifier`` -> ``product_id``
      ``product_product_identifier`` -> ``product_id``
      ``requestedDateTime`` -> ``requested_timestamp``
    """
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        return ""

    # Apply longer aliases first so requested_date_time does not become requested_date + time.
    for old, new in sorted(_SYNONYMS, key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"(?:^|_)({re.escape(old)})(?=_|$)", lambda m: m.group(0).replace(m.group(1), new), text)
        text = re.sub(r"_+", "_", text).strip("_")


    # Treat a trailing state marker as an alias. Do not globally rewrite ``*_key`` to ``*_id``:
    # API/authentication/configuration keys are often deliberately different concepts.
# Source semantic metadata can still collapse exporter aliases when the authoritative source
    # says they represent the same field.
    text = re.sub(r"(?:^|_)state$", "_status", text).lstrip("_")

    # Flatteners and some model exporters duplicate an owning token while appending the scalar
    # property: product.product.id -> product_product_id. Collapse only adjacent identical tokens.
    parts = [part for part in text.split("_") if part]
    collapsed: list[str] = []
    for part in parts:
        if collapsed and collapsed[-1] == part:
            continue
        collapsed.append(part)
    return "_".join(collapsed)


def variable_semantic_aliases(value: Any) -> set[str]:
    """Return a small, conservative alias set for semantic duplicate suppression."""
    canonical = normalize_variable_name(value)
    if not canonical:
        return set()
    aliases = {canonical, re.sub(r"[^a-z0-9]", "", canonical)}

    # A second normalization pass protects against mixed producer orderings such as
    # order_order_total vs purchase_total without introducing broad suffix aliases.
    compact_canonical = aliases.copy()
    for alias in compact_canonical:
        if alias:
            aliases.add(normalize_variable_name(alias))
    return {item for item in aliases if item}

def variable_semantic_identities(value: Any) -> set[str]:
    """Return conservative semantic identities for a variable definition or name.

    Source-backed definitions may carry a stable source semantic key. Use that key in addition
    to the flattened display name so the proposal/persistence layers collapse exporter aliases
    such as ``product_id`` and ``product_product_id`` without collapsing unrelated references.
    """
    if isinstance(value, dict):
        identities: set[str] = set()
        name = value.get("name")
        identities.update(variable_semantic_aliases(name))
        direct = value.get("semantic_key") or value.get("source_semantic_key")
        if direct:
            identities.update(variable_semantic_aliases(direct))
        source_spec = value.get("_json_source_spec")
        if isinstance(source_spec, dict):
            source_key = source_spec.get("semantic_key") or source_spec.get("source_semantic_key") or source_spec.get("path")
            if source_key:
                identities.update(variable_semantic_aliases(source_key))
        return {item for item in identities if item}
    return variable_semantic_aliases(value)

