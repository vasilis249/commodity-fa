"""Deterministic check that every number an agent writes comes from a cited tool result.

Agents cite tool results inline, e.g. "Cushing stocks are 7.4% below the 5-year
average [T3]." For each sentence with a checkable number:
- it must cite at least one known, non-error result id, and
- each number must equal a numeric value in the cited results once that value is
  rounded to the precision written ("7.4%" matches 0.0742; "426k" matches 426398;
  "$100" matches only 99.5..100.5). Percent claims match fractions x100, or raw values
  only in fields that are already percentages (units "%", *pctile/*percentile).
  Sign is ignored (direction is carried by words; the LLM validator checks it).

Extraction is deliberately broad: every digit run is a claim unless it is a whitelisted
identifier: result ids [T3], quantile/contract labels (P10, M1, CLZ26), the 3-2-1 crack,
ISO dates and clock times, horizon tokens (20d, 5y), durations ("20 days", "200-day"),
ordinals (3rd), small counts (< 13 with no unit or currency) and years written in a
date context ("in 2025", "since 2019"). Numbers written as words ("roughly double")
are allowed by design: the prompts ask for words when no tool number exists.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CITE = re.compile(r"\[(T\d+(?:\s*,\s*T\d+)*)\]")
SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")
# identifiers blanked out before extraction
MASKS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"),  # ISO dates/times
    re.compile(r"\b\d{1,2}:\d{2}\b"),  # clock times
    re.compile(r"\b3-2-1\b"),  # crack spread name
]
NUMBER = re.compile(
    r"(?<![\w.])"  # not glued to a preceding letter/digit (P10, T3, CLZ26, 1.5.2)
    r"(?P<sign>[-+\u2212]?)(?P<cur>\$?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<suffix>\s?%|[A-Za-z]*)"
)
SCALE = {"k": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12, "tn": 1e12}
BASIS_POINTS = {"bp", "bps"}
# suffixes that make the token an identifier or a period, not a quantity
SKIP_SUFFIX = {"d", "w", "y", "wk", "mo", "yr", "yrs", "h", "q"}
ORDINAL = {"st", "nd", "rd", "th"}  # "55th percentile" is checked as 55
DURATION = re.compile(
    r"\s?-?\s?(?:trading\s)?(?:days?|weeks?|months?|years?|sessions?|bars?|quarters?)\b",
    re.IGNORECASE,
)
YEAR_CONTEXT = re.compile(
    r"(?:\bin|since|from|until|through|during|of|by|early|mid|late|year|fy|to|and)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumberClaim:
    raw: str
    value: float
    decimals: int
    percent: bool
    scale: float


@dataclass(frozen=True)
class Violation:
    sentence: str
    problem: str

    def __str__(self) -> str:
        return f"{self.problem} — in: {self.sentence!r}"


def _masked(text: str) -> str:
    for rx in MASKS:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def numbers_in(text: str) -> list[NumberClaim]:
    text = _masked(text)
    out = []
    for m in NUMBER.finditer(text):
        num = m.group("num").replace(",", "")
        suffix = m.group("suffix").strip()
        low = suffix.lower()
        if low in SKIP_SUFFIX:
            continue  # 20d, 5y, 3rd, Q3-style tokens
        mantissa = num.split("e")[0].split("E")[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        value = float(num)
        percent = suffix == "%" or low in BASIS_POINTS
        if low in BASIS_POINTS:
            value, decimals = value / 100, decimals + 2  # 150bp -> 1.50%
        scale = SCALE.get(low, 1.0)
        has_unit = percent or scale != 1.0 or bool(m.group("cur"))
        plain_int = decimals == 0 and "e" not in num.lower() and not has_unit
        if low in ORDINAL:
            low, has_unit = "", bool(m.group("cur"))
            plain_int = decimals == 0 and not has_unit
        if plain_int and value < 13 and not low:
            continue  # small counts: "3 analysts", "2 rounds", "3rd"
        if plain_int and DURATION.match(text, m.end()):
            continue  # window lengths: "20 days", "50-day"
        if plain_int and 1900 <= value <= 2100 and YEAR_CONTEXT.search(text[: m.start()]):
            continue  # "in 2025", "since 2019"
        out.append(NumberClaim(m.group(0).strip(), value, decimals, percent, scale))
    return out


@dataclass(frozen=True)
class _Value:
    v: float
    raw_percent: bool  # this field is already a percentage (e.g. units "%", a percentile)


PERCENT_KEYS = ("pctile", "percentile")


def _values(payload: Any, raw_percent: bool = False) -> Iterable[_Value]:
    if isinstance(payload, bool) or payload is None:
        return
    if isinstance(payload, int | float):
        if math.isfinite(payload):
            yield _Value(float(payload), raw_percent)
    elif isinstance(payload, str):
        for n in numbers_in(payload):  # numbers quoted inside strings (e.g. verdicts)
            yield _Value(n.value * n.scale / (100 if n.percent else 1), False)
    elif isinstance(payload, dict):
        if "error" in payload:
            return  # error payloads are never evidence (they can echo agent input)
        units_pct = payload.get("units") == "%"
        for k, v in payload.items():
            if k == "result_id":
                continue
            key_pct = any(t in str(k).lower() for t in PERCENT_KEYS)
            yield from _values(v, raw_percent or units_pct or key_pct)
    elif isinstance(payload, list | tuple):
        for v in payload:
            yield from _values(v, raw_percent)


def matches(claim: NumberClaim, values: Iterable[_Value]) -> bool:
    tol = 0.5 * 10 ** (-claim.decimals) + 1e-9  # the claim's own precision, nothing more
    for val in values:
        a = abs(val.v)
        if claim.percent:
            candidates = [a * 100] + ([a] if val.raw_percent else [])
        else:
            candidates = [a / claim.scale]
        if any(abs(c - claim.value) <= tol for c in candidates):
            return True
    return False


def check_text(text: str, results: dict[str, Any]) -> list[Violation]:
    violations = []
    for sentence in (s.strip() for s in SENTENCE.split(text or "")):
        if not sentence:
            continue
        cited = [rid.strip() for group in CITE.findall(sentence) for rid in group.split(",")]
        body = CITE.sub(" ", sentence)
        claims = numbers_in(body)
        if not claims:
            continue
        if not cited:
            violations.append(
                Violation(sentence, "states a number without citing a tool result [T#]")
            )
            continue
        unknown = [r for r in cited if r not in results]
        if unknown:
            violations.append(Violation(sentence, f"cites unknown result ids {unknown}"))
            continue
        values = [v for r in cited for v in _values(results[r])]
        for c in claims:
            if not matches(c, values):
                violations.append(
                    Violation(sentence, f"number {c.raw} not found in cited results {cited}")
                )
    return violations


def check_fields(obj: Any, results: dict[str, Any]) -> list[Violation]:
    """Check every string inside a (Pydantic-dumped) agent output."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    found: list[Violation] = []
    if isinstance(obj, str):
        found += check_text(obj, results)
    elif isinstance(obj, dict):
        for v in obj.values():
            found += check_fields(v, results)
    elif isinstance(obj, list):
        for v in obj:
            found += check_fields(v, results)
    return found
