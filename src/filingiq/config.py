"""Central configuration. Everything reads from here, nothing hardcodes paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # --- SEC EDGAR -------------------------------------------------------
    # The SEC REQUIRES a descriptive User-Agent with a real contact email.
    # Requests without one get 403'd. Fair-access limit is 10 req/sec; we
    # stay at 6 to be a good citizen.
    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT", "FilingIQ research your.email@example.com"
    )
    sec_requests_per_second: float = float(os.getenv("SEC_RPS", "6"))

    edgar_data_base: str = "https://data.sec.gov"
    edgar_www_base: str = "https://www.sec.gov"

    # --- Storage ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "db" / "filingiq.duckdb"

    # --- Scope (keep v1 small and honest) --------------------------------
    forms: tuple[str, ...] = ("10-K",)
    start_year: int = int(os.getenv("START_YEAR", "2018"))
    end_year: int = int(os.getenv("END_YEAR", "2024"))

    # --- LLM (used from Week 4 onward; router lets you swap providers) ----
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama | groq
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # NOTE: no `groq_api_key` here. One existed and was read by nothing --
    # llm/router.py calls os.getenv("GROQ_API_KEY") directly at each use, so
    # the setting was a second, unused copy of the same value that could
    # silently disagree with the one actually sent.

    # Consumed by retrieval/embedder.py. Every number in EVALUATION.md section 4
    # was produced with this model; overriding it invalidates them.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # NOTE: there is deliberately no `reranker_model` setting. An earlier one
    # defaulted to bge-reranker-v2-m3 and was read by nothing -- the reranker is
    # chosen from hybrid.RERANKER_MODELS via `--reranker {fast,base,large}`, and
    # defaults to 'fast'. A config value the code never loads is worse than no
    # config value: it documents a system that does not exist.

    # NOTE: no `qdrant_url`. retrieval/store.py runs Qdrant in local
    # library mode -- QdrantClient(path=...) -- so no URL is ever dialled.
    # The setting described a server deployment this project does not use.

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()


# ---------------------------------------------------------------------------
# The XBRL tags we treat as ground truth. Note the fallback chains: companies
# report the same economic concept under different us-gaap tags depending on
# their accounting policy and filing year. Handling this correctly is one of
# the first real engineering problems in this project.
# ---------------------------------------------------------------------------
GROUND_TRUTH_METRICS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "rnd_expense": [
        "ResearchAndDevelopmentExpense",
    ],
}
