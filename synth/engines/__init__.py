"""Simulation engines. An engine turns a validated generation spec into rows of ``column -> value``.

Engines are selected by ``spec.engine`` (``"<name>/v<major>"``). The same engine serves every industry: what differs
between industries and scenarios is the spec, never the mechanism. A new kind of simulation (for example one
that is not made of per-entity event histories) is added by writing one engine and registering it here; the
spec, projection, scorer and API wiring are engine-agnostic.
"""
from __future__ import annotations

from typing import Any, Protocol

from synth.clock import RunContext
from synth.spec import GenerationSpec


class Engine(Protocol):
    engine_id: str

    def reference_view(self, spec: GenerationSpec, ctx: RunContext) -> dict[str, Any]:
        """The ``REF`` dictionary that spec invariants/targets are evaluated against."""

    def validate(self, spec: GenerationSpec) -> None:
        """Optional. Raise ``ValueError`` when the spec cannot run on this engine (checked when a spec is loaded)."""

    def simulate(self, spec: GenerationSpec, ctx: RunContext, *,
                 entities: int, per_entity: int, hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """``entities * per_entity`` rows of ``column -> python value`` (datetimes are aware UTC)."""


def get_engine(engine_id: str) -> Engine:
    from synth.engines.rows import EventRows

    registry: dict[str, Engine] = {e.engine_id: e for e in (EventRows(),)}
    try:
        return registry[engine_id]
    except KeyError as exc:
        raise KeyError(f"Unknown simulation engine '{engine_id}'. Registered: {sorted(registry)}") from exc
