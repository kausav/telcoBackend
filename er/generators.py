"""Column generators: a small closed catalogue, one per ``gen.type`` (see ``er.model.GEN_SPECS``).

A generator is given the random stream of its entity and the row built so far, and returns one value. Dates are
``datetime.date`` objects while rows are being built and become ISO text only in the delivered tables.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from typing import Any


class GenerationError(Exception):
    """A generation run could not be completed (the message says why, in plain words)."""


def stream(seed: int, entity: str, stage: str) -> random.Random:
    """An independent, reproducible random stream per entity and stage, so changing one never shifts another."""
    digest = hashlib.sha256(f"{seed}|{entity}|{stage}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


class Gen:
    """One column's value source."""

    def __init__(self, spec: dict[str, Any], label: str = ""):
        self.spec = spec
        self.kind = spec["type"]
        self.label = label
        self.null_share = float(spec.get("null_share") or 0)
        self._n = int(spec.get("start", 1)) if self.kind == "id" else 0
        self._seen: set[str] = set()
        if self.kind == "choice":
            if spec.get("weights"):
                self._labels = list(spec["weights"])
                self._weights = [float(spec["weights"][k]) for k in self._labels]
            else:
                self._labels, self._weights = list(spec["values"]), None
        elif self.kind == "date":
            self._lo, self._hi = date.fromisoformat(spec["start"]).toordinal(), date.fromisoformat(spec["end"]).toordinal()

    def value(self, rng: random.Random, row: dict[str, Any]) -> Any:
        if self.null_share and self.kind != "id" and rng.random() < self.null_share:
            return None
        s = self.spec
        kind = self.kind
        if kind == "id":
            if self._n >= 10 ** s["digits"]:
                raise GenerationError(f"{self.label}: more than {10 ** s['digits'] - 1:,} ids needed; raise 'digits'")
            out = f"{s['prefix']}{self._n:0{s['digits']}d}"
            self._n += 1
            return out
        if kind == "choice":
            return rng.choices(self._labels, weights=self._weights)[0]
        if kind == "int":
            return rng.randint(s["min"], s["max"])
        if kind == "float":
            return round(rng.uniform(s["min"], s["max"]), s["round"])
        if kind == "date":
            return date.fromordinal(rng.randint(self._lo, self._hi))
        if kind == "bool":
            return rng.random() < s["p"]
        if kind == "const":
            return s["value"]
        if kind == "date_after":
            base = row.get(s["column"])
            if base is None:
                return None
            return base + timedelta(days=rng.randint(s["min_days"], s["max_days"]))
        if kind == "digits":
            length, prefix = s["length"], s["prefix"]
            for _ in range(500):
                body = "".join(rng.choice("0123456789") for _ in range(length - len(prefix)))
                if not prefix and body[0] == "0":
                    body = str(rng.randint(1, 9)) + body[1:]
                out = prefix + body
                if not s["unique"]:
                    return out
                if out not in self._seen:
                    self._seen.add(out)
                    return out
            raise GenerationError(f"{self.label}: could not find a new unique number; raise 'length'")
        raise GenerationError(f"{self.label}: unknown generator '{kind}'")
