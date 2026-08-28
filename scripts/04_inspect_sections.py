#!/usr/bin/env python
"""Verify section quality before building anything on top of it.

A sectioniser fails silently. It always produces *something*, and that
something looks fine in a summary count. The checks below are designed to
surface the specific ways it goes wrong.

Usage:
    python scripts/04_inspect_sections.py
    python scripts/04_inspect_sections.py --show AAPL 2024 1A   # eyeball one
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.storage import db  # noqa: E402

QUERIES = {
    "Section coverage per filing": """
        SELECT f.ticker, f.fiscal_year,
               string_agg(s.item, ', ' ORDER BY s.char_start) AS items_found,
               count(s.item) AS n
        FROM filings f LEFT JOIN sections s ON f.accession = s.accession
        WHERE f.local_path IS NOT NULL
        GROUP BY f.ticker, f.fiscal_year ORDER BY n, f.ticker
    """,
    "Section sizes (words) -- look for outliers": """
        SELECT item,
               count(*) AS n,
               min(n_tokens) AS min_words,
               round(median(n_tokens)) AS median_words,
               max(n_tokens) AS max_words
        FROM sections GROUP BY item ORDER BY item
    """,
    "SUSPICIOUSLY SHORT sections (likely TOC rows)": """
        SELECT f.ticker, f.fiscal_year, s.item, s.n_tokens,
               substr(s.text, 1, 90) AS preview
        FROM sections s JOIN filings f ON s.accession = f.accession
        WHERE (s.item IN ('1','1A','7','8') AND s.n_tokens < 500)
        ORDER BY s.n_tokens
    """,
    "SUSPICIOUSLY LONG sections (boundary was missed)": """
        SELECT f.ticker, f.fiscal_year, s.item, s.n_tokens
        FROM sections s JOIN filings f ON s.accession = f.accession
        WHERE s.n_tokens > 60000 ORDER BY s.n_tokens DESC
    """,
    "Risk Factors word count by year (should be stable-ish)": """
        SELECT f.ticker,
               max(CASE WHEN f.fiscal_year = 2021 THEN s.n_tokens END) AS fy2021,
               max(CASE WHEN f.fiscal_year = 2022 THEN s.n_tokens END) AS fy2022,
               max(CASE WHEN f.fiscal_year = 2023 THEN s.n_tokens END) AS fy2023,
               max(CASE WHEN f.fiscal_year = 2024 THEN s.n_tokens END) AS fy2024
        FROM sections s JOIN filings f ON s.accession = f.accession
        WHERE s.item = '1A' GROUP BY f.ticker ORDER BY f.ticker
    """,
    "Sections found by TITLE fallback (verify these by eye)": """
        SELECT f.ticker, f.fiscal_year, s.item, s.n_tokens,
               substr(s.text, 1, 60) AS preview
        FROM sections s JOIN filings f ON s.accession = f.accession
        WHERE s.detect_method = 'title-fallback'
        ORDER BY f.ticker, f.fiscal_year, s.item
    """,
    "Item 8 table retention (financial data must survive)": """
        SELECT f.ticker, f.fiscal_year,
               length(s.text) - length(replace(s.text, '[TABLE]', '')) AS table_marker_chars
        FROM sections s JOIN filings f ON s.accession = f.accession
        WHERE s.item = '8' ORDER BY table_marker_chars
    """,
}


def show_section(ticker: str, fy: int, item: str, n: int = 2000) -> int:
    with db.connect(read_only=True) as con:
        row = con.execute(
            """SELECT s.text, s.n_tokens FROM sections s
               JOIN filings f ON s.accession = f.accession
               WHERE f.ticker = ? AND f.fiscal_year = ? AND s.item = ?""",
            [ticker.upper(), int(fy), item.upper()],
        ).fetchone()
    if not row:
        print(f"No section found for {ticker} FY{fy} Item {item}")
        return 1
    print(f"--- {ticker} FY{fy} Item {item} ({row[1]:,} words) ---\n")
    print(row[0][:n])
    print(f"\n--- [showing first {n} chars] ---")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs=3, metavar=("TICKER", "FY", "ITEM"))
    args = ap.parse_args()

    if not db.settings.db_path.exists():
        print("No database. Run scripts/01_ingest.py then scripts/03_parse.py.")
        return 1

    if args.show:
        return show_section(*args.show)

    with db.connect(read_only=True) as con:
        n = con.execute("SELECT count(*) FROM sections").fetchone()[0]
        if n == 0:
            print("No sections yet. Run: python scripts/03_parse.py")
            return 1
        for title, sql in QUERIES.items():
            print("\n" + "=" * 78)
            print(title)
            print("=" * 78)
            try:
                df = con.execute(sql).fetchdf()
                print("  (no rows -- for the SUSPICIOUS checks, empty is GOOD)"
                      if df.empty else df.to_string(index=False, max_rows=40))
            except Exception as exc:  # noqa: BLE001
                print(f"  query failed: {exc}")

    print("\n" + "=" * 78)
    print("WHAT TO LOOK FOR")
    print("=" * 78)
    print("""
  1. Item 1A (Risk Factors) should be 8,000-25,000 words for a large filer.
     Under ~500 words means you captured a table-of-contents row.

  2. Item 7 (MD&A) should be 5,000-20,000 words.

  3. Any section over 60,000 words means the NEXT item heading was never
     matched, so one section absorbed everything after it. Find the missing
     heading and look at how it's punctuated.

  4. Item 8 should contain many [TABLE] markers. Zero means lxml isn't
     installed and you fell back to regex extraction -- financial tables
     will have been shredded.

  5. Year-over-year Risk Factors word counts should be broadly similar per
     company. A 10x jump between years is a parsing bug, not a disclosure
     change.

  Then eyeball one directly:
     python scripts/04_inspect_sections.py --show AAPL 2024 1A
  It must begin with the real heading and read as prose, not page numbers.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
