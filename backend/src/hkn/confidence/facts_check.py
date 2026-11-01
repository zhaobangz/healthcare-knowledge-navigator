"""Deterministic facts check: every number and drug name in a claim must appear in the
text it cites.

The single most dangerous error in a clinical answer is a dose, interval or threshold
that drifted between the source and the answer ("500 mg" → "250 mg"). An entailment
model can miss a one-digit change; string matching cannot. The check is deliberately
literal and conservative — normalization covers only whitespace, case, thousands
separators and unicode dashes, never unit conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")
_DRUG = re.compile(
    r"\b(cef\w+|\w+(?:cillin|cycline|mycin|micin|floxacin|azole|vir|mab|nib|platin|taxel|"
    r"rubicin|parib|ciclib|trozole|tamoxifen|morphine|codone|orphine|fentanyl|naloxone|"
    r"methadone|tramadol|gentamicin|capecitabine|fluorouracil|oxaliplatin|leucovorin))\b",
    re.IGNORECASE,
)
_YEARS_LIKE = re.compile(r"^(19|20)\d\d$")


def _norm(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text)


def _num(token: str) -> str:
    return token.replace(",", "")


@dataclass
class FactsCheck:
    checked: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of checked entities found; 1.0 when nothing was checkable."""
        return 1.0 if self.checked == 0 else (self.checked - len(self.missing)) / self.checked


def extract_entities(claim: str) -> tuple[set[str], set[str]]:
    numbers = {_num(m) for m in _NUMBER.findall(claim)}
    # Citation-style list markers ("1.", "2.") and bare years add noise, not facts.
    numbers = {n for n in numbers if not _YEARS_LIKE.match(n)}
    drugs = {m.lower() for m in _DRUG.findall(claim)}
    return numbers, drugs


def check_claim(claim: str, cited_text: str) -> FactsCheck:
    numbers, drugs = extract_entities(claim)
    haystack = _norm(cited_text)
    hay_numbers = {_num(m) for m in _NUMBER.findall(haystack)}
    result = FactsCheck()
    for n in sorted(numbers):
        result.checked += 1
        if n not in hay_numbers:
            result.missing.append(n)
    for d in sorted(drugs):
        result.checked += 1
        if d not in haystack:
            result.missing.append(d)
    return result
