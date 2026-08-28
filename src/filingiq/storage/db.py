"""DuckDB storage layer.

DuckDB is the right choice here: zero-config, columnar, reads Parquet natively,
and handles the analytical joins (ground truth vs extraction vs prices) that
this project lives on. One file, no server, ships in your repo.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import duckdb

from filingiq.config import settings
from filingiq.ingestion.edgar import Filing
from filingiq.ingestion.xbrl import GroundTruthFact

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik        BIGINT PRIMARY KEY,
    ticker     VARCHAR NOT NULL,
    name       VARCHAR,
    sic        VARCHAR,
    sector     VARCHAR
);

CREATE TABLE IF NOT EXISTS filings (
    accession         VARCHAR PRIMARY KEY,
    cik               BIGINT NOT NULL,
    ticker            VARCHAR NOT NULL,
    form              VARCHAR NOT NULL,
    filing_date       DATE,
    period_of_report  DATE,
    fiscal_year       INTEGER,
    primary_document  VARCHAR,
    primary_doc_url   VARCHAR,
    local_path        VARCHAR,
    downloaded_at     TIMESTAMP DEFAULT current_timestamp
);

-- Authoritative financials, keyed to the filing that reported them.
CREATE TABLE IF NOT EXISTS ground_truth (
    cik            BIGINT,
    ticker         VARCHAR,
    accession      VARCHAR,
    metric         VARCHAR,
    xbrl_tag       VARCHAR,
    unit           VARCHAR,
    value          DOUBLE,
    period_start   DATE,
    period_end     DATE,
    fiscal_year    INTEGER,
    fiscal_period  VARCHAR,
    is_instant     BOOLEAN,
    PRIMARY KEY (accession, metric)
);

-- Retrieval units, produced in Week 3.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     VARCHAR PRIMARY KEY,
    accession    VARCHAR,
    ticker       VARCHAR,
    fiscal_year  INTEGER,
    item         VARCHAR,
    chunk_index  INTEGER,
    char_start   INTEGER,
    char_end     INTEGER,
    n_tokens     INTEGER,
    has_table    BOOLEAN,
    text         VARCHAR,      -- context-prefixed; this is what gets embedded
    raw_text     VARCHAR       -- original passage; this is what the user sees
);

-- Populated from Week 4: what the LLM claimed, with its citation.
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id  VARCHAR PRIMARY KEY,
    accession      VARCHAR,
    metric         VARCHAR,
    value          DOUBLE,
    unit           VARCHAR,
    raw_text       VARCHAR,      -- the sentence the model read
    chunk_id       VARCHAR,      -- provenance: which chunk it came from
    page_hint      VARCHAR,
    confidence     DOUBLE,
    model          VARCHAR,
    prompt_version VARCHAR,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    latency_ms     INTEGER,
    created_at     TIMESTAMP DEFAULT current_timestamp
);

-- Parsed document sections, produced in Week 2.
CREATE TABLE IF NOT EXISTS sections (
    section_id  VARCHAR PRIMARY KEY,
    accession   VARCHAR,
    item        VARCHAR,          -- '1A', '7', '7A', '8'
    title       VARCHAR,
    char_start  INTEGER,
    char_end    INTEGER,
    n_tokens    INTEGER,
    detect_method VARCHAR,      -- 'item-heading' or 'title-fallback'
    text        VARCHAR
);
"""


@contextmanager
def connect(read_only: bool = False, db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path=db_path) as con:
        con.execute(SCHEMA)
        # Migration for databases created before detect_method existed.
        for table, col, coltype in [
            ("sections", "detect_method", "VARCHAR"),
            ("extractions", "value_as_stated", "DOUBLE"),
            ("extractions", "scale", "VARCHAR"),
            ("extractions", "found", "BOOLEAN"),
            ("extractions", "band", "VARCHAR"),
            ("extractions", "error_type", "VARCHAR"),
        ]:
            try:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                log.info("Migrated %s: added %s", table, col)
            except Exception:
                pass  # column already present
    log.info("Initialised database at %s", db_path or settings.db_path)


def upsert_company(con, cik: int, ticker: str, name: str = "", sic: str = "") -> None:
    con.execute("DELETE FROM companies WHERE cik = ?", [int(cik)])
    con.execute(
        "INSERT INTO companies (cik, ticker, name, sic) VALUES (?, ?, ?, ?)",
        [int(cik), ticker.upper(), name, sic],
    )


def upsert_filings(con, filings: Iterable[Filing]) -> int:
    rows = [
        (
            f.accession, f.cik, f.ticker, f.form,
            f.filing_date or None, f.period_of_report or None,
            f.fiscal_year, f.primary_document, f.primary_doc_url, f.local_path,
        )
        for f in filings
    ]
    if not rows:
        return 0
    con.executemany("DELETE FROM filings WHERE accession = ?", [(r[0],) for r in rows])
    con.executemany(
        """INSERT INTO filings
           (accession, cik, ticker, form, filing_date, period_of_report,
            fiscal_year, primary_document, primary_doc_url, local_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def upsert_ground_truth(con, facts: Iterable[GroundTruthFact]) -> int:
    rows = [
        (
            f.cik, f.ticker, f.accession, f.metric, f.xbrl_tag, f.unit, f.value,
            f.period_start or None, f.period_end or None,
            f.fiscal_year, f.fiscal_period, f.is_instant,
        )
        for f in facts
    ]
    if not rows:
        return 0
    con.executemany(
        "DELETE FROM ground_truth WHERE accession = ? AND metric = ?",
        [(r[2], r[3]) for r in rows],
    )
    con.executemany(
        """INSERT INTO ground_truth
           (cik, ticker, accession, metric, xbrl_tag, unit, value,
            period_start, period_end, fiscal_year, fiscal_period, is_instant)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def summary(db_path: Path | None = None) -> dict:
    with connect(read_only=True, db_path=db_path) as con:
        return {
            "companies": con.execute("SELECT count(*) FROM companies").fetchone()[0],
            "filings": con.execute("SELECT count(*) FROM filings").fetchone()[0],
            "downloaded": con.execute(
                "SELECT count(*) FROM filings WHERE local_path IS NOT NULL"
            ).fetchone()[0],
            "ground_truth_facts": con.execute(
                "SELECT count(*) FROM ground_truth"
            ).fetchone()[0],
            "metrics_covered": con.execute(
                "SELECT count(DISTINCT metric) FROM ground_truth"
            ).fetchone()[0],
        }
