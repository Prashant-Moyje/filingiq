#!/usr/bin/env python
"""Inspect what you ingested and sanity-check the ground truth.

Run this immediately after ingestion. Do NOT move on to parsing until the
numbers here look right -- a silent ground-truth bug will poison every accuracy
metric you report for the next seven weeks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.storage import db  # noqa: E402


QUERIES = {
    "Filings per company": """
        SELECT ticker, count(*) AS filings,
               min(fiscal_year) AS from_fy, max(fiscal_year) AS to_fy,
               sum(CASE WHEN local_path IS NOT NULL THEN 1 ELSE 0 END) AS downloaded
        FROM filings GROUP BY ticker ORDER BY ticker
    """,
    "Ground-truth coverage by metric": """
        SELECT metric,
               count(*) AS n_filings,
               count(DISTINCT ticker) AS n_companies,
               count(DISTINCT xbrl_tag) AS n_tags_used
        FROM ground_truth GROUP BY metric ORDER BY n_filings DESC
    """,
    "Companies with WEAK coverage (investigate these)": """
        SELECT f.ticker, f.fiscal_year, count(g.metric) AS metrics_found
        FROM filings f LEFT JOIN ground_truth g ON f.accession = g.accession
        GROUP BY f.ticker, f.fiscal_year
        HAVING count(g.metric) < 6
        ORDER BY metrics_found, f.ticker
    """,
    "Revenue sanity check (values should be plausible)": """
        SELECT ticker, fiscal_year, xbrl_tag,
               round(value / 1e9, 2) AS revenue_usd_bn,
               period_start, period_end
        FROM ground_truth WHERE metric = 'revenue'
        ORDER BY ticker, fiscal_year
    """,
    "Fiscal period length check (should be ~365 days)": """
        SELECT ticker, fiscal_year, metric,
               date_diff('day', period_start, period_end) AS days
        FROM ground_truth
        WHERE is_instant = FALSE AND period_start IS NOT NULL
          AND (date_diff('day', period_start, period_end) < 300
               OR date_diff('day', period_start, period_end) > 430)
        ORDER BY ticker
    """,
}


def main() -> int:
    if not db.settings.db_path.exists():
        print("No database found. Run scripts/01_ingest.py first.")
        return 1

    with db.connect(read_only=True) as con:
        for title, sql in QUERIES.items():
            print("\n" + "=" * 74)
            print(title)
            print("=" * 74)
            try:
                rows = con.execute(sql).fetchdf()
                if rows.empty:
                    print("  (no rows -- for the checks above, empty is GOOD)")
                else:
                    print(rows.to_string(index=False, max_rows=40))
            except Exception as exc:  # noqa: BLE001
                print(f"  query failed: {exc}")

    print("\n" + "=" * 74)
    print("WHAT TO LOOK FOR")
    print("=" * 74)
    print("""
  1. Revenue figures should match what you'd find on any finance site.
     Spot-check three of them manually. If AAPL FY2023 isn't ~$383B,
     your period-matching logic is wrong.

  2. 'Weak coverage' rows are expected for banks (JPM, GS) -- they use
     different us-gaap tags. That's a real finding, not a bug. Extend the
     fallback chains in config.py and write it up in FAILURE_MODES.md.

  3. The fiscal-period-length check should return NOTHING. Any row means a
     comparative-year fact leaked into your ground truth.

  4. Retailers (WMT, COST, NKE) have non-calendar year-ends. Verify their
     period_end dates look like Jan/May/Aug, not December.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
