"""Deterministic check that every number an agent writes comes from a cited tool result.

Agents cite tool results inline, e.g. "Cushing stocks are 7.4% below the 5-year
average [T3]." For each sentence with a checkable number:
- it must cite at least one known result id, and
- each number must match a numeric value in the cited results, allowing for
  rounding (to the number of decimals written), percent scaling (0.074 -> 7.4%),
  k/M/bn suffixes and sign (direction is carried by words).

Not checked (to avoid false alarms): small counts (< 13), years, durations ("20 days",
"200-day"), and numbers that are part of identifiers such as P10, T3, M1, 20d, 3-2-1
or dates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CITE = re.compile(r"\[(T\d+(?:\s*,\s*T\d+)*)\]")
SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")
NUMBER = re.compile(
    r"(?<![\w.\-/])"  # not part of an identifier, date or decimal
    r"(?P<sign>[-+\u2212]?)(?P<cur>\$?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<unit>\s?%|\s?(?:k|K|bn|mn|M|B)\b)?"
    r"(?![\w\-/]|\.\d)"
)
SCALE = {"k": 1e3, "K": 1e3, "M": 1e6, "mn": 1e6, "B": 1e9, "bn": 1e9}
DURATION = re.compile(
    r"\s?-?\s?(?:trading\s)?(?:days?|weeks?|months?|years?|sessions?|bars?|quarters?)\b",
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


def numbers_in(text: str) -> list[NumberClaim]:
    out = []
    for m in NUMBER.finditer(text):
        num = m.group("num").replace(",", "")
        unit = (m.group("unit") or "").strip()
        decimals = len(num.split(".")[1]) if "." in num else 0
        value = float(num)
        percent = unit == "%"
        scale = SCALE.get(unit, 1.0)
        plain_int = decimals == 0 and not percent and scale == 1.0 and not m.group("cur")
        if plain_int and (value < 13 or 1900 <= value <= 2100):
            continue  # counts, horizons and years
        if plain_int and DURATION.match(text, m.end()):
            continue  # window lengths: "20 days", "50-day", "3 years"
        out.append(NumberClaim(m.group(0).strip(), value, decimals, percent, scale))
    return out


def _numbers_from_payload(payload: Any) -> Iterable[float]:
    if isinstance(payload, bool) or payload is None:
        return
    if isinstance(payload, int | float):
        if math.isfinite(payload):
            yield float(payload)
    elif isinstance(payload, str):
        for n in numbers_in(payload):  # numbers quoted inside strings (e.g. verdicts)
            yield n.value * n.scale / (100 if n.percent else 1)
    elif isinstance(payload, dict):
        for v in payload.values():
            yield from _numbers_from_payload(v)
    elif isinstance(payload, list | tuple):
        for v in payload:
            yield from _numbers_from_payload(v)


def _tolerance(claim: NumberClaim) -> float:
    if claim.decimals:
        return 0.5 * 10 ** (-claim.decimals) + 1e-9
    digits = claim.raw.replace(",", "").rstrip("%kKMBbnm $").lstrip("-+\u2212$")
    zeros = len(digits) - len(digits.rstrip("0"))
    return 0.5 * 10**zeros + 1e-9


def matches(claim: NumberClaim, values: Iterable[float]) -> bool:
    tol = _tolerance(claim)
    for v in values:
        candidates = [abs(v) * 100, abs(v)] if claim.percent else [abs(v) / claim.scale]
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
        values = [v for r in cited for v in _numbers_from_payload(results[r])]
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
