"""Deterministic provenance for every number the model feeds into a calculation.

A threshold the model reads from the policy ("24 hours") must actually appear in the policy
section it cites, *with the same unit*. A figure it takes from the application ("$1,200")
must appear in the cited document. A derived figure must equal an earlier calculation's
result. This closes a specific failure: a valid-looking calculation built on a number
the model hallucinated or mis-assigned (for example treating "$5,000" as "5000 hours").

Scope is deliberately narrow: only calculation inputs, not every number in a finding.
Limits: it checks that a (value, unit) pair exists in the cited source, not that it is the
*right* clause (the claim-support judge covers that), and unit detection is pattern-based
(digits and common number words; no free-form units).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from adjudicator.calculations import Unit
from adjudicator.schemas import CalculationRecord, Case, DocumentKind, PolicyIndex, PolicySection

type Source = Literal["policy", "evidence", "derived"]

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_SMALL_ONES = {k: v for k, v in _ONES.items() if v < 10}
_WORD_NUMBER = rf"(?:(?:{'|'.join(_TENS)})(?:[\s-](?:{'|'.join(_SMALL_ONES)}))?|{'|'.join(_ONES)})"
_DURATION_UNITS = {"hour": "hours", "hr": "hours", "day": "days", "week": "weeks",
                   "month": "months", "year": "years"}  # fmt: skip

_MONEY = re.compile(
    r"(?:(?:US|AU|A)?\$\s?(?P<a>\d[\d,]*(?:\.\d+)?))"
    r"|(?:(?P<b>\d[\d,]*(?:\.\d+)?)\s?(?:USD|AUD|dollars?)\b)",
    re.IGNORECASE,
)
_DURATION = re.compile(
    rf"(?P<n>\d+(?:\.\d+)?|{_WORD_NUMBER})[\s-]*(?P<u>hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s?(?:%|per\s?cent\b|percent\b)", re.IGNORECASE)
_BARE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_MONEY_KEYS = ("amount", "cost", "price", "total", "paid", "fee", "premium", "value")
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit
    text: str


class InputRejected(ValueError):
    """The input's value and unit are not supported by the source it cites."""


def _words_to_number(words: str) -> float:
    return float(sum(_TENS.get(w, _ONES.get(w, 0)) for w in re.split(r"[\s-]+", words.lower())))


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_quantities(text: str) -> list[Quantity]:
    found: list[Quantity] = []
    for m in _MONEY.finditer(text):
        found.append(Quantity(_to_float(m["a"] or m["b"]), "usd", m.group().strip()))
    for m in _DURATION.finditer(text):
        raw = m["n"]
        value = float(raw) if raw[0].isdigit() else _words_to_number(raw)
        unit = _DURATION_UNITS[m["u"].lower().rstrip("s")]
        found.append(Quantity(value, unit, m.group().strip()))  # pyright: ignore[reportArgumentType]
    for m in _PERCENT.finditer(text):
        found.append(Quantity(float(m["n"]), "percent", m.group().strip()))
    return found


def bare_numbers(text: str) -> list[float]:
    return [_to_float(m.group()) for m in _BARE.finditer(text)]


def form_quantities(value: JsonValue, key: str = "") -> list[Quantity]:
    """Numeric form fields, with money-like keys treated as dollars."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        unit: Unit = "usd" if any(k in key.lower() for k in _MONEY_KEYS) else "count"
        return [Quantity(float(value), unit, f"{key}: {value}")]
    if isinstance(value, dict):
        return [q for k, v in value.items() for q in form_quantities(v, k)]
    if isinstance(value, list):
        return [q for item in value for q in form_quantities(item, key)]
    return []


def match_quantity(
    text: str, value: float, unit: Unit, extra: list[Quantity] | None = None
) -> str | None:
    """Return the matching snippet if ``(value, unit)`` is supported by ``text``."""
    quantities = [*extract_quantities(text), *(extra or [])]
    if unit in ("count", "other"):
        for q in quantities:
            if abs(q.value - value) <= _TOLERANCE:
                return q.text
        for n in bare_numbers(text):
            if abs(n - value) <= _TOLERANCE:
                return f"{n:g}"
        return None
    for q in quantities:
        if q.unit == unit and abs(q.value - value) <= _TOLERANCE:
            return q.text
    return None


def section_source_text(index: PolicyIndex, section: PolicySection) -> str:
    """Section text plus transcriptions of diagrams on the pages it spans."""
    parts = [f"{section.number} {section.title}", section.text]
    for page in index.all_pages():
        if (
            page.document_id == section.document_id
            and section.page_start <= page.page_number <= section.page_end
            and page.diagram is not None
            and page.diagram.has_visual_content
        ):
            parts.append(page.searchable_text())
    return "\n".join(parts)


@dataclass(frozen=True)
class VerifiedValue:
    value: float
    unit: Unit
    source: Source
    ref: str
    matched_text: str


class InputVerifier:
    """Verifies calculation inputs against the sources the model says they came from."""

    def __init__(
        self,
        index: PolicyIndex,
        case: Case,
        *,
        sections_read: set[str],
        all_policy_read: bool,
        calculations: dict[str, CalculationRecord],
    ) -> None:
        self._index = index
        self._case = case
        self._read = sections_read
        self._all_read = all_policy_read
        self._calculations = calculations

    def verify(self, value: float, unit: Unit, source: Source, ref: str) -> VerifiedValue:
        match source:
            case "policy":
                snippet = self._policy(value, unit, ref)
            case "evidence":
                snippet = self._evidence(value, unit, ref)
            case "derived":
                snippet = self._derived(value, unit, ref)
        return VerifiedValue(value, unit, source, ref, snippet)

    def _policy(self, value: float, unit: Unit, ref: str) -> str:
        section = self._index.section(ref)
        if section is None:
            raise InputRejected(f"unknown policy section {ref!r}; use list_policy_sections")
        if not self._all_read and ref not in self._read:
            raise InputRejected(
                f"read {ref} with get_policy_section (or view its page) before using its numbers"
            )
        snippet = match_quantity(section_source_text(self._index, section), value, unit)
        if snippet is not None:
            return snippet
        elsewhere = [
            s.section_id
            for s in self._index.all_sections()
            if match_quantity(section_source_text(self._index, s), value, unit) is not None
        ][:3]
        hint = f"; that value and unit appear in {elsewhere}" if elsewhere else ""
        raise InputRejected(
            f"{value:g} {unit} does not appear in policy section {ref}{hint}. "
            "Policy numbers must be quoted from the policy with their unit."
        )

    def _evidence(self, value: float, unit: Unit, ref: str) -> str:
        document = self._case.document(ref)
        if document is None or document.load_status == "failed":
            raise InputRejected(f"no loaded application document {ref!r}")
        extra = form_quantities(self._case.form) if document.kind is DocumentKind.FORM else None
        snippet = match_quantity(document.text, value, unit, extra)
        if snippet is None:
            raise InputRejected(
                f"{value:g} {unit} does not appear in application document {ref}. "
                "Use a figure stated in the document, with its unit."
            )
        return snippet

    def _derived(self, value: float, unit: Unit, ref: str) -> str:
        record = self._calculations.get(ref)
        if record is None:
            raise InputRejected(f"unknown calculation {ref!r}")
        if record.unit != unit or abs(record.result - value) > 0.005:
            raise InputRejected(
                f"{ref} produced {record.result:g} {record.unit}, not {value:g} {unit}"
            )
        return f"{ref} = {record.result:g} {record.unit}"
