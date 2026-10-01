"""How often each business fact actually carries a value for a given scenario mode.

A fact that applies only to a rare branch of the journey (a recurring-top-up period, a suspension reason)
is empty on most rows. Offering it as a variable just hands the caller a mostly-null column, so the proposal
step measures it - on a small, fixed-seed simulation of the behaviour pack itself, never on assumed numbers -
and leaves out the optional columns that are mostly empty for the requested scenario type.
"""
from __future__ import annotations

from synth.clock import RunContext
from synth.engines import get_engine
from synth.pack import BehaviorPack

_ENTITIES = 240
_PER_ENTITY = 4
_AS_OF = "2026-01-01T00:00:00+00:00"        # fixed: a fill share never depends on the wall clock
_cache: dict[tuple[str, int, str], dict[str, float]] = {}


def fill_rates(pack: BehaviorPack, mode: str) -> dict[str, float]:
    """concept id -> share of simulated rows (0..1) in which the concept has a value, for ``mode``."""
    mode = mode if mode in pack.modes else "mixed"
    key = (pack.pack_id, int(pack.version), mode)
    if key not in _cache:
        ctx = RunContext(seed=0, as_of=_AS_OF, tz_name=pack.timezone)
        rows = get_engine(pack.engine).simulate(
            pack, pack.mode_params(mode), ctx, entities=_ENTITIES, per_entity=_PER_ENTITY, hints={})
        total = max(1, len(rows))
        _cache[key] = {
            cid: sum(1 for row in rows if row.get(cid) is not None) / total for cid in pack.concepts
        }
    return _cache[key]
