"""Week 2 pipeline: filings on disk -> sections table in DuckDB."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from filingiq.parsing.html_text import load_filing_text
from filingiq.parsing.sectioniser import TARGET_ITEMS, sectionise
from filingiq.storage import db

log = logging.getLogger(__name__)


def _section_id(accession: str, item: str) -> str:
    return hashlib.sha1(f"{accession}:{item}".encode()).hexdigest()[:16]


def parse_one(accession: str, local_path: str) -> dict:
    path = Path(local_path)
    if not path.exists():
        return {"accession": accession, "error": f"file missing: {local_path}"}

    doc = load_filing_text(path)
    res = sectionise(doc.text)

    return {
        "accession": accession,
        "n_chars": doc.n_chars,
        "n_tables": doc.n_tables,
        "parser": doc.parser_used,
        "toc_span": res.toc_span,
        "found": res.found_items,
        "coverage": res.coverage(),
        "rejected": res.rejected,
        "warnings": res.warnings,
        "sections": res.sections,
    }


def run(tickers: list[str] | None = None, targets: list[str] | None = None) -> dict:
    targets = targets or TARGET_ITEMS
    db.init_db()

    stats = {"filings": 0, "sections": 0, "perfect": 0, "problems": []}

    with db.connect() as con:
        sql = "SELECT accession, ticker, fiscal_year, local_path FROM filings WHERE local_path IS NOT NULL"
        params: list = []
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            sql += f" AND ticker IN ({placeholders})"
            params = [t.upper() for t in tickers]
        sql += " ORDER BY ticker, fiscal_year"

        rows = con.execute(sql, params).fetchall()
        if not rows:
            log.warning("No downloaded filings found. Run scripts/01_ingest.py first.")
            return stats

        for accession, ticker, fy, local_path in rows:
            out = parse_one(accession, local_path)
            stats["filings"] += 1

            if "error" in out:
                log.error("%s FY%s: %s", ticker, fy, out["error"])
                stats["problems"].append(f"{ticker} FY{fy}: {out['error']}")
                continue

            con.execute("DELETE FROM sections WHERE accession = ?", [accession])
            for item, sec in out["sections"].items():
                con.execute(
                    """INSERT INTO sections
                       (section_id, accession, item, title, char_start, char_end,
                        n_tokens, detect_method, text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        _section_id(accession, item), accession, item, sec.title,
                        sec.start, sec.end, sec.n_words, sec.source, sec.text,
                    ],
                )
                stats["sections"] += 1

            missing = [t for t in targets if t not in out["sections"]]
            if not missing:
                stats["perfect"] += 1
                level = log.info
            else:
                stats["problems"].append(
                    f"{ticker} FY{fy}: missing Item(s) {', '.join(missing)}"
                )
                level = log.warning

            level(
                "%s FY%s | %s | %.0f%% coverage | %s chars | %s tables | items: %s",
                ticker, fy, out["parser"], out["coverage"], f"{out['n_chars']:,}",
                out["n_tables"], ", ".join(out["found"]) or "none",
            )
            for r in out["rejected"]:
                log.debug("  rejected -> %s", r)

    return stats
