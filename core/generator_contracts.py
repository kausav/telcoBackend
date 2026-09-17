"""Executable generator contract names shared by schema-definition parsing and generation."""

SUPPORTED_GENERATORS = frozenset({
    "prefixed_int", "id_mirror", "e164_phone", "constant", "weighted_choice",
    "dependent_choice", "weighted_bucket", "uniform", "uniform_int", "lognormal",
    "lognormal_int", "beta", "segment_range", "uniform_bounded", "recent_datetime",
    "ts_offset", "ts_add_field", "date_offset", "date_offset_range", "prefixed_uuid",
    "tx_id", "formula", "generic", "semantic_event",
})
