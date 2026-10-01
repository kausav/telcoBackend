"""Optional domain vocabulary, stored in MongoDB (collection ``domain_lexicon``), never in code.

Business wording rarely matches source-model wording ("top up" vs ``TopupBalance``, "claim" vs
``Loss``). Those bridges are data about an industry, so they live next to the source documents and
behaviour packs and the application ships with none. Every document looks like::

    {"industry_key": "telecom" | "*",            # "*" applies to every industry
     "aliases":          {"top_up": "topup"},     # token -> canonical token
     "expansions":       {"depletion": ["usage"]},# token -> extra tokens that count as the same evidence
     "branch_expansions":{"upsell": ["offer"]},   # token -> tokens enabling optional source branches
     "stopwords":        ["telecom"],             # tokens carrying no relevance evidence
     "industry_labels":  {"telecommunications": "telecom"}}   # spelling -> industry_key ("*" documents only)

Without a lexicon, matching falls back to exact tokens. A database failure is treated the same way.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


@dataclass(frozen=True)
class Lexicon:
    aliases: dict[str, str] = field(default_factory=dict)
    expansions: dict[str, frozenset[str]] = field(default_factory=dict)
    branch_expansions: dict[str, frozenset[str]] = field(default_factory=dict)
    stopwords: frozenset[str] = frozenset()
    industry_labels: dict[str, str] = field(default_factory=dict)


def _documents(industry_key: str) -> list[dict[str, Any]]:
    ttl = float(os.getenv("DOMAIN_LEXICON_CACHE_SECONDS", "60"))
    now = time.monotonic()
    hit = _CACHE.get(industry_key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    rows: list[dict[str, Any]] = []
    try:
        from models._helpers import collection_name
        from models.database import get_database

        wanted = ["*"] if industry_key == "*" else ["*", industry_key]
        cursor = get_database()[collection_name("MONGODB_DOMAIN_LEXICON_COLLECTION", "domain_lexicon")].find(
            {"industry_key": {"$in": wanted}}, {"_id": 0}
        )
        rows = [row for row in cursor if isinstance(row, dict)]
    except Exception as exc:
        logger.warning("domain lexicon not loaded from MongoDB: %s", exc)
        _CACHE[industry_key] = (now - max(ttl - 5.0, 0.0), rows)   # a failure is retried after ~5 s
        return rows
    _CACHE[industry_key] = (now, rows)
    return rows


def _lower(token: Any) -> str:
    return str(token or "").strip().casefold()


def load(industry_key: str | None = None) -> Lexicon:
    """Merged lexicon for one industry (``"*"`` documents first, industry documents override)."""
    key = _lower(industry_key) or "*"
    rows = sorted(_documents(key), key=lambda row: row.get("industry_key") != "*")
    aliases: dict[str, str] = {}
    expansions: dict[str, set[str]] = {}
    branches: dict[str, set[str]] = {}
    stop: set[str] = set()
    labels: dict[str, str] = {}
    for row in rows:
        for src, dst in (row.get("aliases") or {}).items():
            aliases[_lower(src)] = _lower(dst)
        for target, name in ((expansions, "expansions"), (branches, "branch_expansions")):
            for src, values in (row.get(name) or {}).items():
                target.setdefault(_lower(src), set()).update(_lower(v) for v in values or [])
        stop.update(_lower(t) for t in row.get("stopwords") or [])
        if row.get("industry_key") == "*":
            labels.update({_lower(k): _lower(v) for k, v in (row.get("industry_labels") or {}).items()})
    return Lexicon(
        aliases, {k: frozenset(v) for k, v in expansions.items()},
        {k: frozenset(v) for k, v in branches.items()}, frozenset(stop), labels,
    )


def clear_cache() -> None:
    _CACHE.clear()
