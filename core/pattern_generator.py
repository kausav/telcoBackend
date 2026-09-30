"""Small, dependency-free regex sampler for source-defined string patterns.

The sampler intentionally supports the practical subset used by JSON Schema/OpenAPI data
contracts. It validates every sample with Python's regex engine before returning it, so an
unsupported construct fails closed instead of emitting a value that merely looks plausible.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import sre_parse  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    sre_parse = None


class PatternGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class _Context:
    rng: Any
    min_length: int = 0
    max_length: int = 256


def _category_chars(category: Any) -> str:
    if sre_parse is None:
        return "A"
    mapping = {
        sre_parse.CATEGORY_DIGIT: "0123456789",
        sre_parse.CATEGORY_NOT_DIGIT: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        sre_parse.CATEGORY_WORD: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
        sre_parse.CATEGORY_NOT_WORD: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        sre_parse.CATEGORY_SPACE: " ",
        sre_parse.CATEGORY_NOT_SPACE: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    }
    return mapping.get(category, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _in_chars(items: list[tuple[Any, Any]]) -> str:
    chars: list[str] = []
    negated = False
    if sre_parse is None:
        return "A"
    for op, arg in items:
        if op is sre_parse.NEGATE:
            negated = True
        elif op is sre_parse.LITERAL:
            chars.append(chr(arg))
        elif op is sre_parse.RANGE:
            lo, hi = int(arg[0]), int(arg[1])
            if hi - lo > 1024:
                hi = min(hi, lo + 1024)
            chars.extend(chr(i) for i in range(lo, hi + 1))
        elif op is sre_parse.CATEGORY:
            chars.extend(_category_chars(arg))
    if not negated:
        return "".join(dict.fromkeys(chars)) or "A"
    blocked = set(chars)
    safe = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" if c not in blocked]
    if not safe:
        raise PatternGenerationError("negative character class has no safe sample character")
    return "".join(safe)


def _choose_repeat(lo: int, hi: int, ctx: _Context, current_length: int) -> int:
    lo = max(0, int(lo))
    hi = max(lo, int(hi))
    remaining = max(0, ctx.max_length - current_length)
    hi = min(hi, remaining)
    if hi < lo:
        raise PatternGenerationError("pattern minimum length exceeds configured maximum length")
    return ctx.rng.randint(lo, hi)


def _walk(subpattern: Any, ctx: _Context, current_length: int = 0) -> str:
    if sre_parse is None:
        raise PatternGenerationError("regex parser unavailable")
    output: list[str] = []
    for op, arg in subpattern:
        if op is sre_parse.LITERAL:
            output.append(chr(arg))
        elif op is sre_parse.NOT_LITERAL:
            candidates = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            output.append(ctx.rng.choice(candidates))
        elif op is sre_parse.IN:
            output.append(ctx.rng.choice(_in_chars(list(arg))))
        elif op is sre_parse.ANY:
            output.append("A")
        elif op is sre_parse.CATEGORY:
            output.append(ctx.rng.choice(_category_chars(arg)))
        elif op is sre_parse.SUBPATTERN:
            output.append(_walk(arg[-1], ctx, current_length + sum(map(len, output))))
        elif op is sre_parse.BRANCH:
            branches = arg[1]
            output.append(_walk(ctx.rng.choice(branches), ctx, current_length + sum(map(len, output))))
        elif op in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT, getattr(sre_parse, "POSSESSIVE_REPEAT", object())}:
            lo, hi, body = arg
            count = _choose_repeat(lo, hi, ctx, current_length + sum(map(len, output)))
            for _ in range(count):
                output.append(_walk(body, ctx, current_length + sum(map(len, output))))
        elif op is sre_parse.AT:
            continue
        elif op in {sre_parse.ASSERT, sre_parse.ASSERT_NOT, sre_parse.GROUPREF, sre_parse.GROUPREF_EXISTS}:
            raise PatternGenerationError(f"unsupported regex construct: {op}")
        else:
            raise PatternGenerationError(f"unsupported regex construct: {op}")
    return "".join(output)


def generate_regex_sample(pattern: str, rng: Any, *, min_length: int = 0, max_length: int = 256, attempts: int = 40) -> str:
    text = str(pattern or "")
    if not text:
        raise PatternGenerationError("pattern is empty")
    try:
        compiled = re.compile(text)
    except re.error as exc:
        raise PatternGenerationError(f"unsupported or invalid regex: {exc}") from exc
    max_length = max(1, min(int(max_length), 4096))
    min_length = max(0, min(int(min_length), max_length))
    ctx = _Context(rng=rng, min_length=min_length, max_length=max_length)
    for _ in range(max(1, int(attempts))):
        try:
            candidate = _walk(sre_parse.parse(text), ctx)
        except PatternGenerationError:
            raise
        if min_length <= len(candidate) <= max_length and compiled.fullmatch(candidate) is not None:
            return candidate
    raise PatternGenerationError("unable to synthesize a value satisfying the regex contract")
