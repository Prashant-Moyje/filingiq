"""SEC EDGAR client.

Endpoints used (all free, no API key required -- only a User-Agent header):

  Ticker -> CIK map : https://www.sec.gov/files/company_tickers.json
  Filing index      : https://data.sec.gov/submissions/CIK##########.json
  XBRL company facts: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  Filing document   : https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}

The companyfacts endpoint is the heart of this project: every fact it returns
carries an `accn` field naming the exact filing that reported it. That lets us
build per-filing ground truth and measure LLM extraction accuracy objectively.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from filingiq.config import settings
from filingiq.utils.rate_limit import RateLimiter

log = logging.getLogger(__name__)


@dataclass
class Filing:
    cik: int
    ticker: str
    accession: str          # dashed form, e.g. 0000320193-23-000106
    form: str
    filing_date: str
    period_of_report: str
    primary_document: str
    primary_doc_url: str
    local_path: str | None = None

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def fiscal_year(self) -> int:
        # period_of_report is the fiscal period END date. A company with a
        # Sept year-end filing in Nov 2023 has FY2023. Using the report period
        # (not the filing date) is what keeps this correct.
        return int(self.period_of_report[:4]) if self.period_of_report else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or settings.sec_user_agent
        if "example.com" in ua:
            log.warning(
                "You are using the placeholder SEC_USER_AGENT. Set a real "
                "name and email in .env -- the SEC blocks generic agents."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.limiter = RateLimiter(settings.sec_requests_per_second)

    # -- low level ---------------------------------------------------------
    def _get(self, url: str, host: str) -> requests.Response:
        self.limiter.acquire()
        headers = {"Host": host}
        resp = self.session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp

    def _get_json(self, url: str, host: str = "data.sec.gov") -> dict:
        return self._get(url, host).json()

    # -- ticker -> CIK -----------------------------------------------------
    def ticker_to_cik(self) -> dict[str, int]:
        url = f"{settings.edgar_www_base}/files/company_tickers.json"
        raw = self._get_json(url, host="www.sec.gov")
        return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}

    @staticmethod
    def pad_cik(cik: int | str) -> str:
        return str(int(cik)).zfill(10)

    # -- filings -----------------------------------------------------------
    def get_submissions(self, cik: int) -> dict:
        url = f"{settings.edgar_data_base}/submissions/CIK{self.pad_cik(cik)}.json"
        return self._get_json(url)

    def list_filings(
        self,
        cik: int,
        ticker: str,
        forms: Iterable[str] = ("10-K",),
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[Filing]:
        """Return filings of the requested form types within a year range.

        The submissions JSON stores the most recent ~1000 filings inline under
        `filings.recent` as *parallel arrays*, with older ones paginated into
        `filings.files`. We walk both so a 10-year backfill works.
        """
        forms = {f.upper() for f in forms}
        start_year = start_year or settings.start_year
        end_year = end_year or settings.end_year

        sub = self.get_submissions(cik)
        chunks: list[dict] = [sub["filings"]["recent"]]

        for extra in sub["filings"].get("files", []):
            url = f"{settings.edgar_data_base}/submissions/{extra['name']}"
            try:
                chunks.append(self._get_json(url))
            except requests.HTTPError as exc:
                log.warning("Could not fetch older submissions %s: %s", extra["name"], exc)

        out: list[Filing] = []
        for chunk in chunks:
            n = len(chunk.get("accessionNumber", []))
            for i in range(n):
                form = chunk["form"][i].upper()
                if form not in forms:
                    continue
                period = chunk.get("reportDate", [""] * n)[i] or ""
                year = int(period[:4]) if period[:4].isdigit() else 0
                if not (start_year <= year <= end_year):
                    continue
                accession = chunk["accessionNumber"][i]
                doc = chunk.get("primaryDocument", [""] * n)[i]
                if not doc:
                    continue
                url = (
                    f"{settings.edgar_www_base}/Archives/edgar/data/"
                    f"{int(cik)}/{accession.replace('-', '')}/{doc}"
                )
                out.append(
                    Filing(
                        cik=int(cik),
                        ticker=ticker.upper(),
                        accession=accession,
                        form=form,
                        filing_date=chunk["filingDate"][i],
                        period_of_report=period,
                        primary_document=doc,
                        primary_doc_url=url,
                    )
                )

        out.sort(key=lambda f: f.period_of_report)
        return out

    def download_filing(self, filing: Filing, raw_dir: Path | None = None) -> Path:
        """Download the primary document. Idempotent -- skips if already on disk."""
        raw_dir = raw_dir or settings.raw_dir
        dest_dir = raw_dir / filing.ticker / str(filing.fiscal_year)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{filing.accession}_{filing.primary_document}"

        if dest.exists() and dest.stat().st_size > 0:
            filing.local_path = str(dest)
            return dest

        resp = self._get(filing.primary_doc_url, host="www.sec.gov")
        dest.write_bytes(resp.content)
        filing.local_path = str(dest)
        log.info("Downloaded %s %s (%.1f MB)", filing.ticker, filing.fiscal_year,
                 len(resp.content) / 1e6)
        return dest

    # -- XBRL --------------------------------------------------------------
    def get_company_facts(self, cik: int) -> dict:
        url = (
            f"{settings.edgar_data_base}/api/xbrl/companyfacts/"
            f"CIK{self.pad_cik(cik)}.json"
        )
        return self._get_json(url)

    def cache_company_facts(self, cik: int, ticker: str) -> Path:
        dest = settings.raw_dir / ticker.upper() / "companyfacts.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        facts = self.get_company_facts(cik)
        dest.write_text(json.dumps(facts), encoding="utf-8")
        return dest
