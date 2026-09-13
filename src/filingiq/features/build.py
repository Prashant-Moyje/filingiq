"""Feature store: one row per company-year.

THE POINT OF THIS LAYER
-----------------------
Everything before this produced text analysis. This turns it into numbers a
model can use, and answers the question the whole project exists to test:

    do LLM-derived disclosure features add anything beyond fundamentals?

That is an ablation, and it has to be able to come out negative. A feature set
that cannot lose is not being tested.

LEAKAGE
-------
Two rules, both easy to break and fatal if broken:

1. Features must use only information available AT THE FILING DATE. A 10-K for
   FY2024 is filed months after the fiscal year ends; using the fiscal year end
   as the information date leaks months of market data into the features.
2. Targets must be measured strictly AFTER the filing date. Forward returns are
   computed from the filing date, not the period end.

The gap between period_of_report and filing_date is typically 30-90 days. Get
this wrong and the model looks excellent and is worthless.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

THEME_COLUMNS = [
    "supply_chain", "cybersecurity", "regulatory", "litigation", "macro",
    "talent", "climate", "concentration", "liquidity", "ai_technology",
]


@dataclass
class FeatureConfig:
    forward_days: int = 90          # horizon for the target
    # NOTE: there is deliberately no `winsorize` setting. One existed,
    # documented as "clip extreme returns before modelling", and was read by
    # nothing -- returns reaching the model were never clipped. Rather than
    # switch a preprocessing step on retrospectively, which would change every
    # published number in EVALUATION.md section 7, the field is removed and the
    # absence stated plainly: THIS PIPELINE DOES NOT WINSORIZE.
    #
    # Measured, so the choice is informed rather than assumed. Clipping
    # excess_return at the 1st/99th percentile touches 4 of 155 rows and moves
    # disclosure IC from +0.0912 to +0.0883 (p 0.343 -> 0.388); no conclusion
    # changes. To enable it, add it here AND regenerate the feature store, so
    # that code and data agree.


def build_disclosure_features(analysis_rows: list[dict]) -> pd.DataFrame:
    """Turn analysis_results.json into numeric features.

    Rows without a baseline (a company's first year in the corpus) are DROPPED,
    not zero-filled. Zero-filling would assert 'no change' where the truth is
    'unknown', which is a fabricated observation.
    """
    records = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for r in analysis_rows:
        diff = r.get("diff_summary") or {}
        if not diff:
            reject("no diff computed")
            continue
        if diff.get("has_current") is False:
            reject("current year has no Item 1A (parser failure)")
            continue
        if diff.get("has_baseline") is False:
            reject("no prior-year baseline")
            continue
        if diff.get("drift_score") is None:
            reject("drift undefined")
            continue

        total = diff.get("total_current", 0)
        if total < 5:
            # A handful of chunks means the section was partially parsed;
            # rates computed from it are noise with a plausible-looking value.
            reject("too few risk chunks (partial parse)")
            continue
        if diff.get("removed", 0) > 3 * total:
            # More removed than plausibly existed -- the prior year parsed and
            # this one did not, or vice versa.
            reject("implausible removal count (asymmetric parse)")
            continue

        total = max(diff.get("total_current", 0), 1)
        new_themes = r.get("risk_themes") or {}
        mod_themes = r.get("modified_themes") or {}

        rec = {
            "ticker": r["ticker"],
            "fiscal_year": r["fiscal_year"],
            "accession": r["accession"],
            # --- disclosure churn ---
            "drift_score": diff.get("drift_score"),
            "n_new": diff.get("new", 0),
            "n_modified": diff.get("modified", 0),
            "n_removed": diff.get("removed", 0),
            "n_total_risks": total,
            "removal_rate": diff.get("removed", 0) / total,
            "new_rate": diff.get("new", 0) / total,
            "modified_rate": diff.get("modified", 0) / total,
        }
        # Theme intensity: share of changed risks touching each theme. A rate
        # rather than a count, so large and small filers are comparable.
        n_changed = max(diff.get("new", 0) + diff.get("modified", 0), 1)
        for theme in THEME_COLUMNS:
            count = new_themes.get(theme, 0) + mod_themes.get(theme, 0)
            rec[f"theme_{theme}"] = count / n_changed
            rec[f"theme_{theme}_count"] = count
        records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        # One filing per company-year. JNJ appears twice for FY2023 in EDGAR
        # (an amended filing); keeping both would double-weight it and break
        # the walk-forward split's assumption of one observation per period.
        before = len(df)
        df = df.sort_values(["ticker", "fiscal_year"]).drop_duplicates(
            subset=["ticker", "fiscal_year"], keep="last")
        if len(df) < before:
            reject(f"duplicate company-year ({before - len(df)} rows)")

    df.attrs["rejected"] = rejected
    return df


def add_fundamentals(df: pd.DataFrame, con) -> pd.DataFrame:
    """Join XBRL fundamentals -- the baseline feature set the LLM must beat."""
    gt = con.execute(
        "SELECT accession, metric, value FROM ground_truth"
    ).fetchdf()
    if gt.empty:
        return df
    wide = gt.pivot_table(index="accession", columns="metric", values="value",
                          aggfunc="first").reset_index()
    out = df.merge(wide, on="accession", how="left")

    # Ratios generalise across company size; raw magnitudes do not. A model fed
    # raw revenue learns "large companies" rather than anything about risk.
    def safe_div(a, b):
        return out[a] / out[b].replace(0, pd.NA) if a in out and b in out else pd.NA

    out["net_margin"] = safe_div("net_income", "revenue")
    out["operating_margin"] = safe_div("operating_income", "revenue")
    out["leverage"] = safe_div("total_liabilities", "total_assets")
    out["roe"] = safe_div("net_income", "stockholders_equity")
    out["asset_turnover"] = safe_div("revenue", "total_assets")
    out["cash_ratio"] = safe_div("cash_and_equivalents", "total_assets")
    out["rnd_intensity"] = safe_div("rnd_expense", "revenue")
    return out


def add_filing_dates(df: pd.DataFrame, con) -> pd.DataFrame:
    """Attach the FILING date -- the only valid information date.

    Using period_of_report instead would leak the 30-90 days between fiscal
    year end and publication into every feature and target.
    """
    dates = con.execute(
        "SELECT accession, filing_date, period_of_report FROM filings"
    ).fetchdf()
    out = df.merge(dates, on="accession", how="left")
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    out["period_of_report"] = pd.to_datetime(out["period_of_report"])
    out["reporting_lag_days"] = (out["filing_date"]
                                 - out["period_of_report"]).dt.days
    return out


def add_market_target(df: pd.DataFrame, prices: dict[str, pd.DataFrame],
                      config: FeatureConfig | None = None) -> pd.DataFrame:
    """Forward return measured from the FILING date, plus a market-relative version.

    Raw forward return is dominated by market direction: in a rising market
    almost every stock is up 90 days later, and a model 'predicting' that has
    learned the calendar. Subtracting the market return over the identical
    window leaves the company-specific component, which is the only part
    disclosure language could plausibly explain.
    """
    config = config or FeatureConfig()
    spy = prices.get("SPY")
    rows = []

    for _, r in df.iterrows():
        px = prices.get(r["ticker"])
        fwd = market = pd.NA
        if px is not None and pd.notna(r.get("filing_date")):
            fwd = _forward_return(px, r["filing_date"], config.forward_days)
            if spy is not None:
                market = _forward_return(spy, r["filing_date"], config.forward_days)
        rows.append({
            "fwd_return": fwd,
            "market_return": market,
            "excess_return": (fwd - market
                              if pd.notna(fwd) and pd.notna(market) else pd.NA),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _forward_return(px: pd.DataFrame, start, days: int):
    """Return from the first trading day on/after `start` to `days` later."""
    try:
        after = px[px.index >= start]
        if len(after) < 2:
            return pd.NA
        p0 = float(after["Close"].iloc[0])
        window = after[after.index <= start + pd.Timedelta(days=days)]
        if len(window) < 2:
            return pd.NA
        p1 = float(window["Close"].iloc[-1])
        return (p1 - p0) / p0 if p0 else pd.NA
    except Exception:  # noqa: BLE001
        return pd.NA


def fetch_prices(tickers: list[str], start: str = "2017-01-01") -> dict:
    """Daily prices via yfinance. SPY is always included as the market proxy."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for t in sorted(set(tickers) | {"SPY"}):
        try:
            df = yf.Ticker(t).history(start=start, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                out[t] = df
        except Exception as exc:  # noqa: BLE001
            log.warning("price fetch failed for %s: %s", t, exc)
    return out
