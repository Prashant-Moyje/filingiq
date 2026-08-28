#!/usr/bin/env python
"""Week 2 entry point: parse downloaded filings into sections.

Usage:
    python scripts/03_parse.py                       # everything downloaded
    python scripts/03_parse.py --tickers AAPL MSFT
    python scripts/03_parse.py -v                    # show rejected candidates
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.parsing.pipeline import run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Parse filings into Items")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = run(tickers=[t.upper() for t in args.tickers] if args.tickers else None)

    print("\n" + "=" * 62)
    print("PARSING COMPLETE")
    print("=" * 62)
    print(f"  Filings parsed        : {stats['filings']}")
    print(f"  Sections extracted    : {stats['sections']}")
    print(f"  Full target coverage  : {stats['perfect']}/{stats['filings']}")
    if stats["problems"]:
        print(f"\n  Needs attention ({len(stats['problems'])}):")
        for pr in stats["problems"][:20]:
            print(f"    - {pr}")
        print("\n  Each of these is a FAILURE_MODES.md entry waiting to be written.")
    print("=" * 62)
    print("\nNext: python scripts/04_inspect_sections.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
