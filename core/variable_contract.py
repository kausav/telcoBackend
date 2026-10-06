"""Structural contract check for persisted (DB-owned) variable definitions.

A definition can be valid against the ``GeneratedSchemaField`` model and still be unusable by a
generator. Failing early, before a large dataset is generated and repaired row by row, is the point.
The checks are industry-neutral analytical conventions (flag/count/timestamp suffixes, dtype versus
generator compatibility); they carry no domain vocabulary.
"""
from __future__ import annotations

import io
import re
import tokenize
from typing import Any

from core.agentic_models import GeneratedSchemaField


_QUOTED_NAME = re.compile(r"""(['"])([A-Za-z_][A-Za-z0-9_]*)\1""")


def _token_spans(text: str, kinds: set[int]) -> list[tuple[int, int, str]] | None:
    """``(start, end, token)`` for every token of ``kinds`` in ``text``, or None when the text cannot be tokenised."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    starts = [0]
    for line in text.splitlines(True):
        starts.append(starts[-1] + len(line))
    return [
        (starts[tok.start[0] - 1] + tok.start[1], starts[tok.end[0] - 1] + tok.end[1], tok.string)
        for tok in tokens
        if tok.type in kinds
    ]


def rewrite_formula_names(formula: str, mapping: dict[str, str]) -> str:
    """``formula`` with each referenced name in ``mapping`` replaced, leaving string literals untouched.

    A name is only a reference where the expression uses it as one; the same word inside a quoted literal
    (a JSON key, a status text) is data and must keep its spelling.
    """
    if not mapping:
        return formula
    spans = _token_spans(formula, {tokenize.NAME})
    if spans is None:
        return formula
    out: list[str] = []
    last = 0
    for start, end, token in spans:
        replacement = mapping.get(token)
        if replacement and replacement != token:
            out.append(formula[last:start])
            out.append(replacement)
            last = end
    out.append(formula[last:])
    return "".join(out)


def clean_formula(formula: Any, depends_on: Any = ()) -> Any:
    """Make a stored formula parseable without changing what it computes.

    CSV-style exports quote the referenced variables (`` 'a' - 'b'``) and pad the text with spaces. A name
    declared in ``depends_on`` is a reference, not a string literal, so its quotes are dropped; every other
    quoted token (a real literal such as ``'completed'``) is left alone. Only a whole quoted token is
    considered: the same name inside a longer literal (``'{"is_depleted":'``) is text and keeps its quotes.
    """
    if not isinstance(formula, str):
        return formula
    names = {str(d).strip() for d in (depends_on or ()) if str(d).strip()}
    text = formula.strip()
    if names:
        spans = _token_spans(text, {tokenize.STRING})
        if spans is None:
            text = _QUOTED_NAME.sub(lambda m: m.group(2) if m.group(2) in names else m.group(0), text)
        else:
            out: list[str] = []
            last = 0
            for start, end, token in spans:
                match = _QUOTED_NAME.fullmatch(token)
                if match and match.group(2) in names:
                    out.append(text[last:start])
                    out.append(match.group(2))
                    last = end
            out.append(text[last:])
            text = "".join(out)
    return text or None


def clean_definition(variable: dict[str, Any]) -> dict[str, Any]:
    """DB-owned definition with only its formula text normalised (name, params, contract untouched)."""
    item = dict(variable)
    if isinstance(item.get("formula"), str):
        item["formula"] = clean_formula(item["formula"], item.get("depends_on"))
    return item


def validate_db_definition(variable: dict[str, Any]) -> dict[str, Any]:
    """Validate a DB variable without changing its semantics or name.

    Pydantic validates the structural shape of ``GeneratedSchemaField`` but cannot detect a
    contract that is structurally valid yet semantically unusable by the generator, such as
    ``days_to_depletion`` declared as a string or ``offer_accepted_flag`` declared as text.
    Those errors are especially damaging because DB-owned definitions have higher precedence than
    source fields. Fail early here instead of generating a large dataset and repairing every row.
    """
    model = GeneratedSchemaField.model_validate(variable)
    name = str(model.name or "").strip()
    dtype = str(model.dtype or "string").strip().lower()
    gen = str(model.gen or "").strip().lower()
    params = dict(model.params or {})

    if not name:
        raise ValueError("MongoDB scenario variable is missing its name")
    if model.required and model.nullable:
        raise ValueError(f"MongoDB variable '{name}' cannot be both required and nullable")

    numeric_dtypes = {"integer", "int", "float", "decimal", "number", "numeric"}
    boolean_dtypes = {"boolean", "bool"}
    temporal_dtypes = {"datetime", "date", "timestamp"}

    # Semantic suffixes are intentionally narrow. They are universal analytical conventions,
    # unlike industry-specific vocabularies, and therefore apply safely across industries.
    normalized_name = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    bool_like = (
        normalized_name.endswith("_flag")
        or normalized_name.startswith(("is_", "has_", "can_", "should_"))
    )
    numeric_like = (
        normalized_name.endswith((
            "_count", "_days", "_hours", "_minutes", "_seconds", "_duration", "_amount", "_price",
            "_balance", "_velocity", "_rate", "_score", "_percentage", "_percent", "_ratio",
            "_quantity", "_limit", "_gb", "_mb", "_kb", "_gbps", "_mbps",
            "_per_day", "_per_hour", "_per_minute", "_per_second",
        ))
        or normalized_name.startswith(("days_to_", "hours_to_", "minutes_to_", "seconds_to_", "time_to_"))
        or normalized_name in {"count", "amount", "price", "balance", "score", "ratio", "limit"}
    )
    temporal_like = normalized_name.endswith((
        "_timestamp", "_datetime", "_date_time", "_at", "_date",
    ))

    if bool_like and dtype not in boolean_dtypes:
        raise ValueError(f"MongoDB variable '{name}' is boolean-like but dtype='{dtype}'")
    if numeric_like and dtype not in numeric_dtypes:
        raise ValueError(f"MongoDB variable '{name}' is numeric-like but dtype='{dtype}'")
    if temporal_like and dtype not in temporal_dtypes:
        raise ValueError(f"MongoDB variable '{name}' is temporal-like but dtype='{dtype}'")

    # Prevent a semantically numeric/boolean/temporal variable from using a string generator.
    string_generators = {"semantic_string", "generic", "semantic_event", "uuid_string", "email_string", "uri_string"}
    if dtype in numeric_dtypes | boolean_dtypes | temporal_dtypes and gen in string_generators:
        raise ValueError(f"MongoDB variable '{name}' uses incompatible generator '{gen}' for dtype='{dtype}'")

    declared_choices = params.get("choices", params.get("values"))
    if declared_choices is not None:
        choices = list(declared_choices) if isinstance(declared_choices, (list, tuple, set)) else [declared_choices]
        if dtype in numeric_dtypes and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in choices):
            raise ValueError(f"MongoDB variable '{name}' has non-numeric choices for dtype='{dtype}'")
        if dtype in boolean_dtypes and not all(isinstance(v, bool) for v in choices):
            raise ValueError(f"MongoDB variable '{name}' has non-boolean choices for dtype='{dtype}'")

    # Avoid static/vendor vocabulary encoded as a generic constant when the name clearly describes
    # a measured field. Constants are still allowed for categorical/status fields.
    if gen == "constant" and numeric_like and "value" in params:
        value = params.get("value")
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise ValueError(f"MongoDB variable '{name}' constant value must be numeric")

    if gen == "formula" and not str(model.formula or "").strip():
        raise ValueError(f"MongoDB variable '{name}' uses formula generator but has no formula")

    # Keep the original DB mapping intact. Callers use this only as validation feedback and
    # deliberately continue storing the source mapping unchanged.
    return model.model_dump()
