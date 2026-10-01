"""Run context: seeded RNG + reference clock.

Every stochastic decision goes through ``RunContext.rng`` and every "now" through
``RunContext.as_of`` so a (seed, as_of) pair reproduces a dataset byte-for-byte. The legacy
generator used ``datetime.now()`` in several places, which made ``seed`` only partially effective.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Any

from zoneinfo import ZoneInfo


def get_tz(name: str | None) -> tzinfo:
    """IANA zone by name. An unknown zone is an error: silently using another one would shift every timestamp."""
    try:
        return ZoneInfo(name or "UTC")
    except Exception as exc:
        raise ValueError(f"unknown time zone {name!r} (needs an IANA name and the tzdata package)") from exc


def parse_as_of(value: Any) -> datetime | None:
    """Accept datetime / ISO-8601 string / None; always return an aware UTC datetime (or None)."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class RunContext:
    seed: int | None = None
    as_of: datetime | None = None
    tz_name: str = "UTC"
    rng: random.Random = field(init=False)
    registries: dict[str, set] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.as_of = parse_as_of(self.as_of) or datetime.now(timezone.utc).replace(microsecond=0)
        self.tz = get_tz(self.tz_name)

    # ---- distributions (stdlib only) ---------------------------------------------------------
    def weighted(self, items: list[Any], weights: list[float]) -> Any:
        return self.rng.choices(items, weights=weights, k=1)[0]

    def lognormal_median(self, median: float, sigma: float) -> float:
        return self.rng.lognormvariate(math.log(max(median, 1e-9)), sigma)

    def bernoulli(self, p: float) -> bool:
        return self.rng.random() < min(max(p, 0.0), 1.0)

    def unique(self, registry: str, factory) -> Any:
        """Draw from ``factory()`` until the value is unseen in ``registry`` (bounded retries)."""
        seen = self.registries.setdefault(registry, set())
        for _ in range(10_000):
            value = factory()
            if value not in seen:
                seen.add(value)
                return value
        raise RuntimeError(f"could not draw a unique value for registry '{registry}'")
