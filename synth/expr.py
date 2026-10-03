"""A tiny, safe expression language for the rules and formulas of a generation spec.

Only a whitelist of AST nodes is evaluated (no attribute access, no imports, no comprehension),
so a spec - which a language model writes and a database stores - is never a code-execution vector.
Short-circuit semantics of ``and`` / ``or`` / ``if-else`` are preserved, which lets an author guard ``None`` values:

    status != 'completed' or confirmed_at is not None
"""
from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime, timedelta
from typing import Any, Callable

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}


def _secs(a: datetime, b: datetime) -> float:
    """seconds(a - b)"""
    return (a - b).total_seconds()


def _days(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 86400.0


def _matches(pattern: str, value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _distinct(values: Any) -> int:
    """Number of distinct non-null values in a list (entity/dataset level invariants)."""
    return len({v for v in values if v is not None and (not isinstance(v, tuple) or None not in v)})


def _pairs(a: Any, b: Any) -> list[tuple]:
    return list(zip(a, b))


def _strictly_increasing(values: Any) -> bool:
    seq = [v for v in values if v is not None]
    return all(x < y for x, y in zip(seq, seq[1:]))


def _non_decreasing(values: Any) -> bool:
    seq = [v for v in values if v is not None]
    return all(x <= y for x, y in zip(seq, seq[1:]))


def _all_after(starts: Any, ends: Any, applies: Any = None) -> bool:
    """Entity-level: every event starts after the previous event's ``end`` (only where ``applies`` holds for that previous event)."""
    flags = applies if isinstance(applies, (list, tuple)) else [True if applies is None else bool(applies)] * len(ends)
    return all(nxt >= end for nxt, end, on in zip(starts[1:], ends, flags) if on and end is not None and nxt is not None)


def _min_gap_secs(values: Any) -> float:
    seq = sorted(v for v in values if v is not None)
    gaps = [(y - x).total_seconds() for x, y in zip(seq, seq[1:])]
    return min(gaps) if gaps else float("inf")


def _hour_of(dt: datetime, offset_minutes: float) -> int:
    return (dt + timedelta(minutes=offset_minutes)).hour


def _add_days(dt: datetime, n: float) -> datetime:
    return dt + timedelta(days=n)


def _add_secs(dt: datetime, n: float) -> datetime:
    return dt + timedelta(seconds=n)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)                      # x << 0: exp(-x) would overflow
    return e / (1.0 + e)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "add_days": _add_days, "add_secs": _add_secs, "text": lambda x: None if x is None else str(x),
    "lower": lambda x: None if x is None else str(x).lower(), "upper": lambda x: None if x is None else str(x).upper(),
    "int": int, "float": float, "sigmoid": _sigmoid,
    "abs": abs, "min": min, "max": max, "len": len, "round": round,
    "secs": _secs, "days": _days, "matches": _matches,
    "is_none": lambda x: x is None,
    "distinct": _distinct, "pairs": _pairs, "strictly_increasing": _strictly_increasing, "non_decreasing": _non_decreasing, "all_after": _all_after,
    "min_gap_secs": _min_gap_secs, "hour_of": _hour_of,
}


class ExprError(ValueError):
    pass


_VALID_ESCAPES = frozenset("\\'\"abfnrtvxNuU01234567\n")


def _literal_escapes(source: str) -> str:
    """``source`` with every backslash that does not start a Python string escape doubled (``'\\+91'`` keeps meaning a backslash and a plus).

    A pattern such as ``matches('\\+91[0-9]{10}', x)`` is written that way by authors; Python only warns about it today and
    will refuse it in a future version, and the value is the same either way.
    """
    out: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == "\\":
            nxt = source[i + 1] if i + 1 < len(source) else ""
            if nxt in _VALID_ESCAPES and nxt != "":
                out.append(ch + nxt)
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class Expr:
    """Compiled expression. ``names`` are the free variables it reads (concept ids)."""

    def __init__(self, source: str):
        self.source = source
        try:
            self._tree = ast.parse(_literal_escapes(source.strip()), mode="eval").body
        except SyntaxError as exc:
            raise ExprError(f"invalid expression {source!r}: {exc}") from exc
        self.names: set[str] = set()
        self._validate(self._tree)

    # ---- validation -------------------------------------------------------------------------
    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, (ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.IfExp,
                             ast.Tuple, ast.List, ast.Set, ast.Load, ast.And, ast.Or, ast.Not,
                             ast.USub, ast.UAdd, *_BIN, *_CMP)):
            pass
        elif isinstance(node, ast.Constant):
            return
        elif isinstance(node, ast.Name):
            if node.id not in FUNCTIONS and node.id not in {"True", "False", "None", "REF", "P"}:
                self.names.add(node.id)
            return
        elif isinstance(node, ast.Subscript):
            pass                      # any validated expression may be a key: REF['t'][text(days)]
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExprError(f"call not allowed in {self.source!r}")
        else:
            raise ExprError(f"{type(node).__name__} not allowed in {self.source!r}")
        for child in ast.iter_child_nodes(node):
            self._validate(child)

    # ---- evaluation -------------------------------------------------------------------------
    def __call__(self, env: dict[str, Any]) -> Any:
        return self._eval(self._tree, env)

    def _eval(self, n: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id == "True":
                return True
            if n.id == "False":
                return False
            if n.id == "None":
                return None
            if n.id in FUNCTIONS:
                return FUNCTIONS[n.id]
            return env[n.id]
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                result: Any = True
                for v in n.values:
                    result = self._eval(v, env)
                    if not result:
                        return result
                return result
            result = False
            for v in n.values:
                result = self._eval(v, env)
                if result:
                    return result
            return result
        if isinstance(n, ast.UnaryOp):
            v = self._eval(n.operand, env)
            return (not v) if isinstance(n.op, ast.Not) else (-v if isinstance(n.op, ast.USub) else +v)
        if isinstance(n, ast.BinOp):
            return _BIN[type(n.op)](self._eval(n.left, env), self._eval(n.right, env))
        if isinstance(n, ast.Compare):
            left = self._eval(n.left, env)
            for op, comp in zip(n.ops, n.comparators):
                right = self._eval(comp, env)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self._eval(n.body, env) if self._eval(n.test, env) else self._eval(n.orelse, env)
        if isinstance(n, (ast.Tuple, ast.List)):
            return [self._eval(e, env) for e in n.elts]
        if isinstance(n, ast.Set):
            return {self._eval(e, env) for e in n.elts}
        if isinstance(n, ast.Subscript):
            return self._eval(n.value, env)[self._eval(n.slice, env)]
        if isinstance(n, ast.Call):
            return FUNCTIONS[n.func.id](*[self._eval(a, env) for a in n.args])  # type: ignore[attr-defined]
        raise ExprError(f"cannot evaluate {type(n).__name__}")

