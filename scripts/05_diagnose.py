#!/usr/bin/env python
"""Diagnose why a filing didn't sectionise properly.

Dumps the raw evidence -- where every item heading and every candidate title
actually sits in the document -- so a fix can be based on the document's real
structure rather than a guess about it.

Usage:
    python scripts/05_diagnose.py JPM 2024
    python scripts/05_diagnose.py JPM 2024 --context 200
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.parsing.html_text import load_filing_text  # noqa: E402
from filingiq.parsing.sectioniser import (  # noqa: E402
    detect_toc_span,
    find_candidates,
    sectionise,
)
from filingiq.storage import db  # noqa: E402

# Loose patterns -- no end-of-line anchor, no line-start anchor. We want to see
# EVERY occurrence so we can judge which formatting variant the filer uses.
LOOSE_TITLES = {
    "7":  r"management[’'`]?s\s+discussion\s+and\s+analysis",
    "7A": r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk",
    "8":  r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm",
    "8b": r"consolidated\s+(?:statements?\s+of\s+income|balance\s+sheets?)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("fiscal_year", type=int)
    ap.add_argument("--context", type=int, default=110)
    ap.add_argument("--max-hits", type=int, default=12)
    args = ap.parse_args()

    with db.connect(read_only=True) as con:
        row = con.execute(
            "SELECT local_path FROM filings WHERE ticker = ? AND fiscal_year = ?",
            [args.ticker.upper(), args.fiscal_year],
        ).fetchone()
    if not row or not row[0]:
        print(f"No downloaded filing for {args.ticker} FY{args.fiscal_year}")
        return 1

    doc = load_filing_text(row[0])
    text = doc.text
    print(f"\n{args.ticker} FY{args.fiscal_year} | {doc.parser_used} | "
          f"{doc.n_chars:,} chars | {doc.n_tables} tables\n")

    # --- where the item headings landed ---------------------------------
    cands = find_candidates(text)
    toc = detect_toc_span(cands)
    print("=" * 78)
    print(f"TOC span detected: {toc}")
    print("=" * 78)

    res = sectionise(text)
    print("\nSections built (position as % through document):")
    for item in res.found_items:
        s = res.sections[item]
        pct = 100 * s.start / len(text)
        print(f"  Item {item:<3} {pct:5.1f}%  start={s.start:>9,}  "
              f"{s.n_words:>7,} words  [{s.source}]")

    # --- every item-number candidate, in position order ------------------
    print("\n" + "=" * 78)
    print("ALL item-number candidates (pos% | item | in_toc | context)")
    print("=" * 78)
    for c in cands:
        in_toc = "TOC" if toc and toc[0] <= c.start <= toc[1] else "   "
        pct = 100 * c.start / len(text)
        ctx = re.sub(r"\s+", " ", text[c.start:c.start + 70])
        print(f"  {pct:5.1f}% | {c.item:<3} | {in_toc} | {ctx}")

    # --- where the titles actually appear --------------------------------
    print("\n" + "=" * 78)
    print("TITLE occurrences (loose match -- shows the real formatting)")
    print("=" * 78)
    for item, pat in LOOSE_TITLES.items():
        hits = list(re.compile(pat, re.I).finditer(text))
        print(f"\n--- Item {item}: {len(hits)} occurrence(s) ---")
        if not hits:
            print("    NONE -- this title never appears; the filer uses different wording")
            continue
        for m in hits[: args.max_hits]:
            pct = 100 * m.start() / len(text)
            before = text[max(0, m.start() - args.context):m.start()]
            after = text[m.end():m.end() + args.context]
            before = re.sub(r"\s+", " ", before)[-args.context:]
            after = re.sub(r"\s+", " ", after)
            at_line_start = text[:m.start()].rsplit("\n", 1)[-1].strip() == ""
            rest_of_line = after.split("\n")[0].strip()
            alone = at_line_start and len(rest_of_line) < 5
            print(f"\n  [{pct:5.1f}%] line_start={at_line_start} alone_on_line={alone}")
            print(f"    ...{before}")
            print(f"    >>> {m.group(0)}")
            print(f"    {after}...")
        if len(hits) > args.max_hits:
            print(f"\n    ({len(hits) - args.max_hits} more not shown)")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""
  Compare the pos% of the title occurrences against the pos% of the sections
  that WERE found.

  - If the real MD&A text sits AFTER Item 15, the filing is a wrapper: the
    numbered items are an index and the actual content is appended behind
    them. The fallback's search window is then wrong by construction.

  - If 'alone_on_line' is False everywhere, the heading shares its line with
    other text (a page header, a page number), so a line-anchored regex can
    never match it.

  - If the title never appears at all, the filer uses different wording and
    the pattern itself needs widening.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
