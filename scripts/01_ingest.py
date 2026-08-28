#!/usr/bin/env python
"""Week 1 entry point: build the corpus and the ground-truth table.

Usage:
    python scripts/01_ingest.py                    # full universe
    python scripts/01_ingest.py --tickers AAPL MSFT
    python scripts/01_ingest.py --no-download      # metadata + ground truth only
    python scripts/01_ingest.py --limit 2          # 2 most recent filings each

Start with:  python scripts/01_ingest.py --tickers AAPL --limit 1
Confirm it works on one company before running all 20.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.ingestion.pipeline import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest SEC filings and XBRL ground truth")
    p.add_argument("--tickers", nargs="*", help="Override the universe file")
    p.add_argument("--no-download", action="store_true",
                   help="Skip downloading filing documents")
    p.add_argument("--limit", type=int, default=None,
                   help="Max filings per ticker (most recent first)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = run(
        tickers=[t.upper() for t in args.tickers] if args.tickers else None,
        download=not args.no_download,
        limit_per_ticker=args.limit,
    )

    print("\n" + "=" * 62)
    print("INGESTION COMPLETE")
    print("=" * 62)
    print(f"  Companies processed : {stats['tickers']}")
    print(f"  Filings indexed     : {stats['filings']}")
    print(f"  Documents downloaded: {stats['downloaded']}")
    print(f"  Ground-truth facts  : {stats['gt_facts']}")
    print("\n  Database state:")
    for k, v in stats["db_summary"].items():
        print(f"    {k:.<22} {v}")
    if stats["errors"]:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for e in stats["errors"][:15]:
            print(f"    - {e}")
    print("=" * 62)
    print("\nNext: python scripts/02_inspect.py")
    return 1 if stats["tickers"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
