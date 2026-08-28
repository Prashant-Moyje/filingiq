"""Feature store tests -- mostly guarding against leakage."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from filingiq.features.build import (  # noqa: E402
    _forward_return, add_market_target, build_disclosure_features,
)


def row(ticker="AAPL", fy=2024, drift=0.22, new=0, modified=14, removed=8,
        total=63, baseline=True, themes=None):
    return {
        "ticker": ticker, "fiscal_year": fy, "accession": f"{ticker}-{fy}",
        "diff_summary": {"new": new, "modified": modified, "removed": removed,
                         "unchanged": total - new - modified,
                         "total_current": total,
                         "drift_score": drift if baseline else None,
                         "has_baseline": baseline},
        "risk_themes": {}, "modified_themes": themes or {"supply_chain": 7},
    }


# --- baseline handling -----------------------------------------------------

def test_rows_without_a_baseline_are_dropped_not_zero_filled():
    """Zero-filling would assert 'no change' where the truth is 'unknown' --
    a fabricated observation fed to the model as fact."""
    df = build_disclosure_features([row(baseline=True), row(fy=2021, baseline=False)])
    assert len(df) == 1
    assert df.iloc[0]["fiscal_year"] == 2024


def test_all_rows_lacking_baselines_yields_empty_frame():
    assert build_disclosure_features([row(baseline=False)]).empty


# --- feature construction --------------------------------------------------

def test_theme_features_are_rates_not_raw_counts():
    """Rates make large and small filers comparable; counts would encode
    filing length instead of risk emphasis."""
    df = build_disclosure_features([row(new=0, modified=10,
                                        themes={"supply_chain": 5})])
    assert df.iloc[0]["theme_supply_chain"] == pytest.approx(0.5)
    assert df.iloc[0]["theme_supply_chain_count"] == 5


def test_rates_are_bounded_and_consistent():
    df = build_disclosure_features([row(new=3, modified=12, removed=5, total=60)])
    r = df.iloc[0]
    assert 0 <= r["new_rate"] <= 1 and 0 <= r["modified_rate"] <= 1
    assert r["n_total_risks"] == 60


def test_missing_themes_default_to_zero_not_missing():
    df = build_disclosure_features([row(themes={})])
    assert df.iloc[0]["theme_cybersecurity"] == 0.0


# --- leakage ---------------------------------------------------------------

def _prices(start="2024-01-01", days=200, drift=0.001):
    idx = pd.bdate_range(start, periods=days)
    return pd.DataFrame({"Close": [100 * (1 + drift) ** i for i in range(days)]},
                        index=idx)


def test_forward_return_starts_at_the_given_date():
    px = _prices()
    r = _forward_return(px, pd.Timestamp("2024-01-01"), 90)
    assert r > 0


def test_forward_return_ignores_prices_before_the_date():
    """The target must not see anything at or before the information date."""
    px = _prices()
    early = _forward_return(px, pd.Timestamp("2024-01-01"), 30)
    late = _forward_return(px, pd.Timestamp("2024-04-01"), 30)
    assert early != late


def test_forward_return_is_na_without_enough_history():
    px = _prices(days=2)
    assert pd.isna(_forward_return(px, pd.Timestamp("2025-01-01"), 90))


def test_excess_return_subtracts_the_market():
    """A raw forward return in a rising market mostly measures the calendar."""
    df = pd.DataFrame([{"ticker": "AAPL", "filing_date": pd.Timestamp("2024-01-02")}])
    out = add_market_target(df, {"AAPL": _prices(drift=0.002),
                                 "SPY": _prices(drift=0.001)})
    r = out.iloc[0]
    assert r["fwd_return"] > r["excess_return"] > 0


def test_excess_return_can_be_negative_against_a_strong_market():
    df = pd.DataFrame([{"ticker": "AAPL", "filing_date": pd.Timestamp("2024-01-02")}])
    out = add_market_target(df, {"AAPL": _prices(drift=0.0005),
                                 "SPY": _prices(drift=0.002)})
    assert out.iloc[0]["excess_return"] < 0


def test_missing_price_series_yields_na_not_zero():
    """A zero target is a claim that the stock did not move. Absent data must
    stay absent."""
    df = pd.DataFrame([{"ticker": "NOPE", "filing_date": pd.Timestamp("2024-01-02")}])
    out = add_market_target(df, {"SPY": _prices()})
    assert pd.isna(out.iloc[0]["fwd_return"])


# --- parser-failure contamination ------------------------------------------

def bad_row(ticker="BAC", fy=2023, removed=100, total=0, has_current=False):
    return {"ticker": ticker, "fiscal_year": fy, "accession": f"{ticker}-{fy}",
            "diff_summary": {"new": 0, "modified": 0, "unchanged": 0,
                             "removed": removed, "total_current": total,
                             "drift_score": None if not has_current else 0.0,
                             "has_baseline": True, "has_current": has_current},
            "risk_themes": {}, "modified_themes": {}}


def test_parser_failure_is_not_a_disclosure_event():
    """BAC FY2023 parsed with no Item 1A. The diff reported '0 new, 0 modified,
    100 removed, drift 0.0' -- which reads as a company deleting its entire
    risk section, and would be the strongest signal in the feature store."""
    df = build_disclosure_features([bad_row(), row()])
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"


def test_rejection_reasons_are_reported():
    df = build_disclosure_features([bad_row(), row()])
    reasons = df.attrs["rejected"]
    assert any("Item 1A" in r for r in reasons)


def test_partial_parse_is_dropped():
    """A section that yielded 3 chunks gives rates computed from noise."""
    df = build_disclosure_features([row(new=1, modified=1, total=3)])
    assert df.empty


def test_implausible_removal_count_is_dropped():
    """More removed than plausibly existed means one of the two years failed
    to parse, so the comparison is between a document and a fragment."""
    df = build_disclosure_features([row(new=0, modified=2, removed=90, total=10)])
    assert df.empty


def test_duplicate_company_years_are_collapsed():
    """JNJ filed FY2023 twice (an amendment). Two rows would double-weight it
    and break the one-observation-per-period assumption of walk-forward."""
    df = build_disclosure_features([row(ticker="JNJ", fy=2023),
                                    row(ticker="JNJ", fy=2023)])
    assert len(df) == 1


def test_valid_rows_survive_all_filters():
    df = build_disclosure_features([row(new=2, modified=14, removed=8, total=63)])
    assert len(df) == 1 and df.iloc[0]["drift_score"] == 0.22


# --- degenerate diffs must not become observations -------------------------

def test_unparsed_current_year_is_dropped_not_scored_as_zero_drift():
    """The nastiest of the three exclusions. When the CURRENT year fails to
    parse, every prior chunk counts as 'removed', total_current is 0, and
    drift computes to 0.0 -- which reads as 'the company changed nothing'.
    BAC FY2023, GS FY2020, MCD FY2019 and NKE FY2024 all produced that."""
    r = row()
    r["diff_summary"] = {"new": 0, "modified": 0, "unchanged": 0, "removed": 100,
                         "total_current": 0, "drift_score": None,
                         "has_baseline": True, "has_current": False}
    assert build_disclosure_features([r]).empty


def test_size_mismatch_is_dropped():
    r = row()
    r["diff_summary"] = {"new": 21, "modified": 12, "unchanged": 0, "removed": 1,
                         "total_current": 33, "drift_score": None,
                         "has_baseline": True, "has_current": True,
                         "size_mismatch": True}
    assert build_disclosure_features([r]).empty


def test_duplicate_company_year_is_kept_once():
    """JNJ filed twice for FY2023. Two rows double-weight the observation and
    put the same data on both sides of a walk-forward split."""
    df = build_disclosure_features([row(ticker="JNJ", fy=2023),
                                    row(ticker="JNJ", fy=2023)])
    assert len(df) == 1


def test_distinct_years_are_both_kept():
    df = build_disclosure_features([row(fy=2023), row(fy=2024)])
    assert len(df) == 2
