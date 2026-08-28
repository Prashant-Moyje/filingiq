#!/usr/bin/env python
"""Week 6a: assemble the feature store.

Combines disclosure features (drift, theme intensities) with XBRL fundamentals
and forward market returns into one row per company-year.

Usage:
    python scripts/14_build_features.py
    python scripts/14_build_features.py --no-prices    # skip yfinance
    python scripts/14_build_features.py --forward-days 60
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from filingiq.config import settings              # noqa: E402
from filingiq.features.build import (             # noqa: E402
    FeatureConfig, add_filing_dates, add_fundamentals, add_market_target,
    build_disclosure_features, fetch_prices,
)
from filingiq.storage import db                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--forward-days", type=int, default=90)
    ap.add_argument("--no-prices", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    path = Path(args.analysis) if args.analysis else settings.data_dir / "analysis_results.json"
    if not path.exists():
        print("No analysis results. Run: python scripts/13_analyze.py --dry-run")
        return 1

    rows = json.loads(path.read_text(encoding="utf-8"))
    df = build_disclosure_features(rows)
    dropped = len(rows) - len(df)
    print(f"Disclosure features: {len(df)} usable rows of {len(rows)} "
          f"({dropped} dropped)")
    for reason, n in sorted(df.attrs.get("rejected", {}).items(),
                            key=lambda kv: -kv[1]):
        print(f"    {n:>3}  {reason}")
    if df.empty:
        print("Nothing usable. Widen the corpus (config/universe.yaml) and "
              "re-run ingestion.")
        return 1

    with db.connect(read_only=True) as con:
        df = add_fundamentals(df, con)
        df = add_filing_dates(df, con)

    print(f"Reporting lag: median {df['reporting_lag_days'].median():.0f} days "
          f"between fiscal year end and filing")
    print("  (features and targets both anchor on FILING date, not period end)")

    if not args.no_prices:
        print("\nFetching prices...")
        prices = fetch_prices(df["ticker"].unique().tolist())
        print(f"  got {len(prices)} price series")
        df = add_market_target(df, prices, FeatureConfig(forward_days=args.forward_days))

    out = settings.data_dir / "features.parquet"
    df.to_parquet(out, index=False)
    csv = settings.data_dir / "features.csv"
    df.to_csv(csv, index=False)

    print(f"\n{'=' * 70}\nFEATURE STORE\n{'=' * 70}")
    print(f"  rows      : {len(df)}")
    print(f"  companies : {df['ticker'].nunique()}")
    print(f"  years     : {int(df['fiscal_year'].min())}-{int(df['fiscal_year'].max())}")
    print(f"  columns   : {len(df.columns)}")

    if "excess_return" in df:
        valid = df["excess_return"].notna().sum()
        print(f"\n  Target (excess return over {args.forward_days}d): "
              f"{valid}/{len(df)} rows have it")
        if valid:
            s = pd.to_numeric(df["excess_return"], errors="coerce").dropna()
            print(f"    mean={s.mean():+.3f}  median={s.median():+.3f}  "
                  f"std={s.std():.3f}")
            print(f"    positive: {(s > 0).mean():.1%}")

    print("\n  Rows per year (walk-forward needs enough in each):")
    print(df.groupby("fiscal_year").size().to_string())

    n = len(df)
    print(f"\n{'=' * 70}")
    if n < 60:
        print(f"  WARNING: {n} rows is too few to fit and validate a model.")
        print("  Widen config/universe.yaml and re-run:")
        print("    python scripts/01_ingest.py")
        print("    python scripts/03_parse.py && python scripts/06_chunk.py")
        print("    python scripts/13_analyze.py --dry-run")
        print("  All of those are free -- no LLM calls involved.")
    else:
        print(f"  {n} rows. Enough for a regularised model with walk-forward")
        print("  validation, though still small: report intervals, not points.")
    print(f"\nSaved -> {out}\n        {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
