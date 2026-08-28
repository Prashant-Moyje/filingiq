"""Build per-filing ground truth from SEC XBRL company facts.

WHY THIS FILE MATTERS
---------------------
This is the differentiator of the whole project. Every fact in the SEC's
companyfacts payload carries an `accn` field identifying the exact accession
number (filing) that reported it. That means for any given 10-K we can build an
authoritative, machine-readable table of what the financials actually were --
independent of anything an LLM says.

Later, when the extraction agent reads Item 8 and claims "revenue was $383.3B",
we join on (cik, accession, metric) and check it. That gives us a real accuracy
metric instead of vibes, and it turns hallucination from an anecdote into a
number we can put on a dashboard.

SUBTLETIES WORTH KNOWING (and worth mentioning in an interview)
---------------------------------------------------------------
1. A 10-K reports 2-3 years of comparatives. We keep only the fact whose period
   matches the filing's fiscal period -- otherwise you "verify" FY2022 revenue
   against an FY2021 number and conclude your LLM is hallucinating when it isn't.
2. The same concept appears under different us-gaap tags across companies and
   years (ASC 606 changed revenue tagging in 2018). Hence the fallback chains
   in GROUND_TRUTH_METRICS.
3. Facts come in flavours: duration facts (revenue, over a period: has `start`
   and `end`) vs instant facts (assets, at a point: `end` only). We handle both.
4. Some facts are restated in later filings. We key on accession, so we capture
   what THAT filing said -- which is what the LLM read.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from filingiq.config import GROUND_TRUTH_METRICS

log = logging.getLogger(__name__)

# Duration facts for an annual period should span roughly a year.
_MIN_ANNUAL_DAYS = 300
_MAX_ANNUAL_DAYS = 430


@dataclass
class GroundTruthFact:
    cik: int
    ticker: str
    accession: str
    metric: str
    xbrl_tag: str
    unit: str
    value: float
    period_start: str | None
    period_end: str
    fiscal_year: int
    fiscal_period: str
    is_instant: bool


def _parse(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_annual_duration(fact: dict) -> bool:
    start, end = _parse(fact.get("start")), _parse(fact.get("end"))
    if not start or not end:
        return False
    days = (end - start).days
    return _MIN_ANNUAL_DAYS <= days <= _MAX_ANNUAL_DAYS


def _close_to(a: str | None, b: str | None, tolerance_days: int = 10) -> bool:
    """Fiscal period ends can differ by a few days (52/53-week retail calendars)."""
    da, db = _parse(a), _parse(b)
    if not da or not db:
        return False
    return abs((da - db).days) <= tolerance_days


def extract_ground_truth(
    company_facts: dict,
    cik: int,
    ticker: str,
    accession: str,
    period_of_report: str,
    metrics: dict[str, list[str]] | None = None,
) -> list[GroundTruthFact]:
    """Pull the ground-truth financials that *this specific filing* reported.

    Args:
        company_facts: parsed companyfacts JSON for the company.
        accession: dashed accession number of the filing.
        period_of_report: the filing's fiscal period end (YYYY-MM-DD).
    """
    metrics = metrics or GROUND_TRUTH_METRICS
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    dei = company_facts.get("facts", {}).get("dei", {})
    namespaces = {**gaap, **dei}

    results: list[GroundTruthFact] = []

    for metric, tag_chain in metrics.items():
        fact = _first_matching_fact(
            namespaces, tag_chain, accession, period_of_report
        )
        if fact is None:
            continue
        tag, unit, f = fact
        results.append(
            GroundTruthFact(
                cik=int(cik),
                ticker=ticker.upper(),
                accession=accession,
                metric=metric,
                xbrl_tag=tag,
                unit=unit,
                value=float(f["val"]),
                period_start=f.get("start"),
                period_end=f["end"],
                fiscal_year=int(f.get("fy") or period_of_report[:4] or 0),
                fiscal_period=str(f.get("fp") or "FY"),
                is_instant="start" not in f,
            )
        )

    missing = set(metrics) - {r.metric for r in results}
    if missing:
        log.debug("%s %s: no ground truth for %s", ticker, accession, sorted(missing))

    return results


def _first_matching_fact(
    namespaces: dict,
    tag_chain: list[str],
    accession: str,
    period_of_report: str,
) -> tuple[str, str, dict] | None:
    """Walk the fallback chain and return the first fact reported by this filing
    for the filing's own fiscal period."""
    for tag in tag_chain:
        entry = namespaces.get(tag)
        if not entry:
            continue
        for unit, facts in entry.get("units", {}).items():
            candidates = [
                f
                for f in facts
                if f.get("accn") == accession
                and _close_to(f.get("end"), period_of_report)
            ]
            if not candidates:
                continue

            # Duration facts: keep only the ~annual one (drops Q4-only and
            # comparative-period rows). Instant facts pass through.
            duration = [f for f in candidates if "start" in f]
            instant = [f for f in candidates if "start" not in f]

            chosen: dict | None = None
            if duration:
                annual = [f for f in duration if _is_annual_duration(f)]
                if annual:
                    # Prefer the one whose `form` is the 10-K itself.
                    tenk = [f for f in annual if str(f.get("form", "")).upper() == "10-K"]
                    chosen = (tenk or annual)[0]
            elif instant:
                chosen = instant[0]

            if chosen is not None:
                return tag, unit, chosen
    return None


def coverage_report(facts: list[GroundTruthFact], metrics: dict | None = None) -> dict:
    """How complete is our ground truth? Report this honestly in EVALUATION.md --
    you can only measure extraction accuracy on metrics you actually have."""
    metrics = metrics or GROUND_TRUTH_METRICS
    found = {f.metric for f in facts}
    return {
        "metrics_expected": len(metrics),
        "metrics_found": len(found),
        "coverage_pct": round(100 * len(found) / max(len(metrics), 1), 1),
        "missing": sorted(set(metrics) - found),
    }
