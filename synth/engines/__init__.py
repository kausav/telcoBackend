"""Simulation engines. An engine turns a validated behaviour pack into rows of ``concept -> value``.

Engines are selected by ``pack.engine`` (``"<name>/v<major>"``). Adding a new behaviour for another
industry means writing one engine (or reusing one with a different pack) and registering it here;
the canonicaliser, projection, scorer and API wiring are engine-agnostic.
"""
from __future__ import annotations

from typing import Any, Protocol

from synth.clock import RunContext
from synth.pack import BehaviorPack


class Engine(Protocol):
    engine_id: str

    def reference_view(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        """The ``REF`` dictionary that pack invariants/targets are evaluated against."""

    def validate(self, pack: BehaviorPack) -> None:
        """Optional. Raise ``ValueError`` when the pack cannot run on this engine (checked when a pack is loaded)."""

    def simulate(self, pack: BehaviorPack, params: dict[str, Any], ctx: RunContext, *,
                 entities: int, per_entity: int, hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """``entities * per_entity`` rows of ``concept -> python value`` (datetimes are aware UTC)."""


def get_engine(engine_id: str) -> Engine:
    from synth.engines.renewal import RenewalJourney
    from synth.engines.rows import EventRows

    registry: dict[str, Engine] = {e.engine_id: e for e in (RenewalJourney(), EventRows())}
    try:
        return registry[engine_id]
    except KeyError as exc:
        raise KeyError(f"Unknown simulation engine '{engine_id}'. Registered: {sorted(registry)}") from exc
