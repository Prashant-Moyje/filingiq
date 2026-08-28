"""End-to-end Week 1 pipeline: universe -> filings -> documents -> ground truth."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from filingiq.config import settings
from filingiq.ingestion.edgar import EdgarClient, Filing
from filingiq.ingestion.xbrl import extract_ground_truth, coverage_report
from filingiq.storage import db

log = logging.getLogger(__name__)


def load_universe(path: Path | None = None) -> list[str]:
    path = path or (settings.data_dir.parent / "config" / "universe.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [t.upper() for t in data["tickers"]]


def run(
    tickers: list[str] | None = None,
    download: bool = True,
    limit_per_ticker: int | None = None,
) -> dict:
    settings.ensure_dirs()
    db.init_db()

    tickers = tickers or load_universe()
    client = EdgarClient()

    log.info("Resolving ticker -> CIK map ...")
    cik_map = client.ticker_to_cik()

    stats = {"tickers": 0, "filings": 0, "downloaded": 0, "gt_facts": 0, "errors": []}

    with db.connect() as con:
        for ticker in tickers:
            cik = cik_map.get(ticker)
            if cik is None:
                log.error("Ticker %s not found in SEC map -- skipping", ticker)
                stats["errors"].append(f"{ticker}: unknown ticker")
                continue

            try:
                db.upsert_company(con, cik, ticker)

                filings = client.list_filings(cik, ticker, forms=settings.forms)
                if limit_per_ticker:
                    filings = filings[-limit_per_ticker:]

                if not filings:
                    log.warning("%s: no %s filings in %s-%s", ticker,
                                settings.forms, settings.start_year, settings.end_year)
                    continue

                if download:
                    for f in filings:
                        try:
                            client.download_filing(f)
                            stats["downloaded"] += 1
                        except Exception as exc:  # noqa: BLE001
                            log.error("%s %s download failed: %s", ticker, f.accession, exc)
                            stats["errors"].append(f"{ticker}/{f.accession}: {exc}")

                db.upsert_filings(con, filings)
                stats["filings"] += len(filings)

                # --- ground truth -------------------------------------------
                facts_path = client.cache_company_facts(cik, ticker)
                company_facts = json.loads(facts_path.read_text(encoding="utf-8"))

                for f in filings:
                    gt = extract_ground_truth(
                        company_facts,
                        cik=cik,
                        ticker=ticker,
                        accession=f.accession,
                        period_of_report=f.period_of_report,
                    )
                    db.upsert_ground_truth(con, gt)
                    stats["gt_facts"] += len(gt)
                    cov = coverage_report(gt)
                    log.info(
                        "%s FY%s: %s filings-facts, coverage %.0f%% (missing: %s)",
                        ticker, f.fiscal_year, len(gt), cov["coverage_pct"],
                        ", ".join(cov["missing"]) or "none",
                    )

                stats["tickers"] += 1

            except Exception as exc:  # noqa: BLE001
                log.exception("%s failed", ticker)
                stats["errors"].append(f"{ticker}: {exc}")

    stats["db_summary"] = db.summary()
    return stats
