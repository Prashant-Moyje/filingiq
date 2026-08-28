"""Claim verification: the hallucination gate.

THE PROBLEM WITH GENERATED ANALYSIS
-----------------------------------
An LLM writing an analyst memo will produce fluent, plausible, correctly
formatted prose containing numbers it invented. This is far more dangerous than
a refusal, because nothing in the output signals which figures are real.

WHAT THIS MODULE DOES
---------------------
Extracts every numeric claim from generated text, and checks each against the
set of figures that were independently verified against XBRL. Claims that
cannot be matched are flagged or removed before the memo is shown to anyone.

The check is arithmetic, not semantic. No second LLM grades the first -- a
model asked "is this correct?" agrees with itself, which is not verification.

WHAT IT CANNOT DO
-----------------
Only NUMERIC claims are checkable this way. "Management's tone on supply chain
risk has become more cautious" is unverifiable by arithmetic, and this module
makes no attempt to judge it. Those claims are instead required to carry a
chunk citation, so a human can check them in one click. Stating that boundary
plainly is more useful than pretending to verify everything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers with an optional scale word, a currency symbol, or a percent sign.
# Bare integers under 1000 are ignored: years, counts and list indices are not
# financial claims and flagging them produces noise that trains people to
# ignore the flags.
_NUMBER_RE = re.compile(
    r"""(?<![\w.])
    (?P<currency>[$€£]\s?)?
    (?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,}|\d{1,3})
    \s*
    (?P<scale>billion|bn|million|mn|thousand|trillion|%|percent)?
    (?![\w])""",
    re.IGNORECASE | re.VERBOSE,
)

SCALE_WORDS = {
    "trillion": 1e12, "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mn": 1e6, "thousand": 1e3,
}

# Fractional tolerance when matching a claim to a verified figure. Generous,
# because memos legitimately round: "$391.0 billion" for 391,035,000,000.
MATCH_TOLERANCE = 0.01


# Bare four-digit numbers in this range are years, not financial claims.
# Flagging "fiscal 2024" as an unverified figure is noise, and a gate that
# cries wolf gets ignored -- which costs more than the claims it catches.
YEAR_MIN, YEAR_MAX = 1900, 2100


def _is_probably_a_year(value: float, currency: str | None,
                        scale: str | None) -> bool:
    if currency or scale:
        return False               # "$2,024 million" is a figure, not a year
    return value == int(value) and YEAR_MIN <= value <= YEAR_MAX


def _is_uncheckable_small_integer(raw: str, currency: str | None,
                                  scale: str | None) -> bool:
    """Bare small integers ("12 segments", "3 regions") are not claims.

    But a CURRENCY-MARKED small number is: per-share figures live there.
    An earlier version excluded everything under 1000 outright, so a memo
    stating "earnings per share were $6" against a true $6.08 passed the gate
    untouched -- the exact class of error the gate exists to catch. A dollar
    sign is the signal that a number is financial regardless of magnitude.
    """
    if currency or scale:
        return False
    return "." not in raw and "," not in raw and len(raw) <= 3


@dataclass
class Claim:
    raw: str
    value: float | None
    is_percentage: bool
    start: int
    end: int
    sentence: str
    supported: bool = False
    matched_metric: str | None = None


@dataclass
class ClaimReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def numeric_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.value is not None
                and not c.is_percentage]

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.numeric_claims if not c.supported]

    def summary(self) -> dict:
        n = len(self.numeric_claims)
        return {
            "numeric_claims": n,
            "supported": n - len(self.unsupported),
            "unsupported": len(self.unsupported),
            "support_rate": round((n - len(self.unsupported)) / n, 3) if n else None,
        }


# A sentence boundary is a .!? followed by whitespace or end-of-string.
# Splitting on a bare "." cuts "$47.2 billion" in half, producing the fragment
# "Charges were $47." -- which then failed to match any real sentence, so
# strip_unsupported_sentences() removed nothing while reporting success. A
# safety gate that silently does nothing is worse than an absent one.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _sentence_around(text: str, pos: int) -> str:
    start = 0
    for m in _SENTENCE_END.finditer(text[:pos]):
        start = m.end()
    nl = text.rfind("\n", 0, pos)
    start = max(start, nl + 1)

    end = len(text)
    m = _SENTENCE_END.search(text, pos)
    if m:
        end = m.end()
    return text[start:end].strip()


def extract_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for m in _NUMBER_RE.finditer(text):
        raw_value = m.group("value").replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        scale = (m.group("scale") or "").lower()
        currency = m.group("currency")
        if _is_probably_a_year(value, currency, scale):
            continue
        if _is_uncheckable_small_integer(raw_value, currency, scale):
            continue
        is_pct = scale in ("%", "percent")
        if scale in SCALE_WORDS:
            value *= SCALE_WORDS[scale]
        claims.append(Claim(
            raw=m.group(0).strip(), value=value, is_percentage=is_pct,
            start=m.start(), end=m.end(),
            sentence=_sentence_around(text, m.start()),
        ))
    return claims


def verify_claims(text: str, verified_values: dict[str, float],
                  tolerance: float = MATCH_TOLERANCE) -> ClaimReport:
    """Match numeric claims against XBRL-verified figures.

    A claim counts as supported if it is within tolerance of any verified
    value, at any of the scales a memo might use. The scale sweep matters:
    a memo saying "391,035" (millions, as printed) and one saying
    "$391.0 billion" both refer to the same verified figure.
    """
    report = ClaimReport(claims=extract_claims(text))
    if not verified_values:
        return report

    for claim in report.numeric_claims:
        for metric, truth in verified_values.items():
            if truth is None or truth == 0:
                continue
            for factor in (1.0, 1e3, 1e6, 1e9):
                scaled = claim.value * factor
                if abs(scaled - truth) / abs(truth) <= tolerance:
                    claim.supported = True
                    claim.matched_metric = metric
                    break
            if claim.supported:
                break
    return report


def annotate_unsupported(text: str, report: ClaimReport,
                         marker: str = " [UNVERIFIED]") -> str:
    """Mark unsupported figures in place, working backwards so offsets hold."""
    out = text
    for claim in sorted(report.unsupported, key=lambda c: -c.end):
        out = out[:claim.end] + marker + out[claim.end:]
    return out


def strip_unsupported_sentences(text: str, report: ClaimReport) -> str:
    """Remove whole sentences containing unsupported figures.

    Deleting the number alone would leave a mutilated sentence that still reads
    as an assertion. If a claim cannot be supported, the claim goes.
    """
    bad = {c.sentence for c in report.unsupported if c.sentence}
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip() not in bad]
    return " ".join(kept)
