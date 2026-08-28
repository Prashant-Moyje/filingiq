#!/usr/bin/env python
"""Week 3a: split sections into retrieval units.

Usage:
    python scripts/06_chunk.py
    python scripts/06_chunk.py --tickers AAPL
    python scripts/06_chunk.py --stats-only     # report without rewriting
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.parsing.chunker import chunk_section  # noqa: E402
from filingiq.parsing.sectioniser import ITEM_SEQUENCE, TARGET_ITEMS  # noqa: E402
from filingiq.storage import db  # noqa: E402

log = logging.getLogger(__name__)
TITLES = dict(ITEM_SEQUENCE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--items", nargs="*", default=TARGET_ITEMS)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )

    db.init_db()
    total = 0
    with db.connect() as con:
        sql = """SELECT s.section_id, s.accession, s.item, s.text,
                        f.ticker, f.fiscal_year
                 FROM sections s JOIN filings f ON s.accession = f.accession
                 WHERE s.item IN ({})""".format(
            ",".join("?" for _ in args.items))
        params = list(args.items)
        if args.tickers:
            sql += " AND f.ticker IN ({})".format(
                ",".join("?" for _ in args.tickers))
            params += [t.upper() for t in args.tickers]
        sql += " ORDER BY f.ticker, f.fiscal_year, s.char_start"

        rows = con.execute(sql, params).fetchall()
        if not rows:
            print("No sections found. Run scripts/03_parse.py first.")
            return 1

        if not args.stats_only:
            con.execute("DELETE FROM chunks")

        for section_id, accession, item, text, ticker, fy in rows:
            meta = {"ticker": ticker, "fiscal_year": fy, "item": item,
                    "item_title": TITLES.get(item, "")}
            chunks = chunk_section(text, accession, item, meta)
            total += len(chunks)
            if args.stats_only:
                continue
            for c in chunks:
                con.execute(
                    """INSERT INTO chunks
                       (chunk_id, accession, ticker, fiscal_year, item, chunk_index,
                        char_start, char_end, n_tokens, has_table, text, raw_text)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [c.chunk_id, accession, ticker, fy, item, c.chunk_index,
                     c.char_start, c.char_end, c.n_tokens, c.has_table,
                     c.text, c.raw_text])

        print("\n" + "=" * 70)
        print(f"CHUNKING COMPLETE -- {total:,} chunks from {len(rows)} sections")
        print("=" * 70)
        if not args.stats_only:
            print(con.execute("""
                SELECT item, count(*) AS chunks,
                       round(avg(n_tokens)) AS avg_tokens,
                       min(n_tokens) AS min_t, max(n_tokens) AS max_t,
                       sum(CASE WHEN has_table THEN 1 ELSE 0 END) AS with_table
                FROM chunks GROUP BY item ORDER BY item
            """).fetchdf().to_string(index=False))
            print("\nPer company:")
            print(con.execute("""
                SELECT ticker, count(*) AS chunks, count(DISTINCT accession) AS filings
                FROM chunks GROUP BY ticker ORDER BY ticker
            """).fetchdf().to_string(index=False))
            bad = con.execute("""
                SELECT count(*) FROM chunks
                WHERE (length(raw_text) - length(replace(raw_text, '[TABLE]', '')))
                    / 7 <> (length(raw_text) - length(replace(raw_text, '[/TABLE]', ''))) / 8
            """).fetchone()[0]
            print(f"\nChunks with unbalanced table markers: {bad}  (must be 0)")
    print("\nNext: embeddings + vector index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
