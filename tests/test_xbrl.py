"""Tests for ground-truth extraction.

These run offline against a hand-built fixture that reproduces the three
failure modes that actually bite you on real data:

  1. Comparative-year facts from the same filing (a 10-K reports 3 years).
  2. Quarterly facts mixed in with annual ones.
  3. Facts from OTHER filings that mention the same period.

If extract_ground_truth() picks the wrong one, your entire accuracy metric is
silently wrong. Hence: tests first.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.ingestion.xbrl import (  # noqa: E402
    extract_ground_truth,
    coverage_report,
    _is_annual_duration,
    _close_to,
)

TARGET_ACCN = "0000320193-23-000106"
OTHER_ACCN = "0000320193-22-000108"

FIXTURE = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        # correct: annual, this filing, matching period end
                        {"start": "2022-09-25", "end": "2023-09-30", "val": 383285000000,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "FY", "form": "10-K"},
                        # trap 1: comparative prior year, same filing
                        {"start": "2021-09-26", "end": "2022-09-24", "val": 394328000000,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "FY", "form": "10-K"},
                        # trap 2: a quarter within the filing
                        {"start": "2023-07-02", "end": "2023-09-30", "val": 89498000000,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "Q4", "form": "10-K"},
                        # trap 3: different filing entirely
                        {"start": "2022-09-25", "end": "2023-09-30", "val": 999999999999,
                         "accn": OTHER_ACCN, "fy": 2023, "fp": "FY", "form": "10-Q"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        # instant fact -- no 'start' key
                        {"end": "2023-09-30", "val": 352583000000,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "FY", "form": "10-K"},
                        {"end": "2022-09-24", "val": 352755000000,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "FY", "form": "10-K"},
                    ]
                }
            },
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {"start": "2022-09-25", "end": "2023-09-30", "val": 6.13,
                         "accn": TARGET_ACCN, "fy": 2023, "fp": "FY", "form": "10-K"},
                    ]
                }
            },
            # fallback-chain test: primary tag absent, older tag present
            "Revenues": {"units": {"USD": []}},
            "SalesRevenueNet": {"units": {"USD": []}},
        }
    },
}


@pytest.fixture
def facts():
    return extract_ground_truth(
        FIXTURE, cik=320193, ticker="AAPL",
        accession=TARGET_ACCN, period_of_report="2023-09-30",
    )


def test_picks_the_annual_fact_not_the_comparative(facts):
    rev = next(f for f in facts if f.metric == "revenue")
    assert rev.value == 383285000000, "picked a comparative or quarterly period"


def test_ignores_facts_from_other_filings(facts):
    rev = next(f for f in facts if f.metric == "revenue")
    assert rev.value != 999999999999


def test_handles_instant_facts(facts):
    assets = next(f for f in facts if f.metric == "total_assets")
    assert assets.value == 352583000000
    assert assets.is_instant is True
    assert assets.period_start is None


def test_non_usd_units_preserved(facts):
    eps = next(f for f in facts if f.metric == "eps_diluted")
    assert eps.unit == "USD/shares"
    assert eps.value == pytest.approx(6.13)


def test_missing_metrics_are_omitted_not_guessed(facts):
    found = {f.metric for f in facts}
    assert "operating_cash_flow" not in found, "invented a fact that isn't in the data"


def test_coverage_report_is_honest(facts):
    cov = coverage_report(facts)
    assert cov["metrics_found"] == len(facts)
    assert 0 < cov["coverage_pct"] < 100
    assert "operating_cash_flow" in cov["missing"]


@pytest.mark.parametrize(
    "fact,expected",
    [
        ({"start": "2022-09-25", "end": "2023-09-30"}, True),   # 370 days
        ({"start": "2023-07-02", "end": "2023-09-30"}, False),  # quarter
        ({"start": "2021-09-26", "end": "2023-09-30"}, False),  # two years
        ({"end": "2023-09-30"}, False),                          # instant
    ],
)
def test_annual_duration_detection(fact, expected):
    assert _is_annual_duration(fact) is expected


def test_fiscal_period_tolerance():
    # 52/53-week retail calendars shift the year-end by a few days.
    assert _close_to("2023-01-28", "2023-01-31") is True
    assert _close_to("2023-01-28", "2023-03-31") is False
