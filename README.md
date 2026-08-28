# FilingIQ



![tests](https://github.com/Prashant-Moyje/filingiq/actions/workflows/tests.yml/badge.svg)

**Multi-agent SEC filing intelligence with XBRL-verified extraction and an ML risk layer.**

Analysts can't read every 10-K, and the highest-signal content is usually *what changed* since last year. FilingIQ extracts structured risk and financial features from SEC filings, verifies every number against authoritative XBRL data, and feeds the results into a gradient-boosted model.

> **The design decision this project is built around:** the SEC publishes XBRL data alongside every filing, and each XBRL fact names the exact filing that reported it. That gives us objective ground truth, so extraction accuracy is a *measured number*, not a claim.

---

## Status

| Week | Stage | Status |
|---|---|---|
| 1 | EDGAR ingestion + XBRL ground truth | ✅ Done |
| 2 | Document parsing + sectioniser | ✅ Done |
| 3 | Chunking, hybrid retrieval, eval harness | ✅ Done |
| 4 | Extraction agent + XBRL verification | ✅ Done |
| 5 | LangGraph orchestration + hallucination gate | ✅ Done |
| 6 | Feature store + ML layer + walk-forward CV | ✅ Done |
| 7 | Evaluation harness in CI | ⬜ |
| 8 | UI, Docker, deploy | ⬜ |

## Results

*(Fill this in as you go. Recruiters read this table and nothing else on the first pass — so it belongs at the top, and every number must be reproducible via `make eval`.)*

| Metric | Value | Notes |
|---|---|---|
| Retrieval hit@5 (hybrid, filtered) | **0.852** [0.797, 0.901] | 182 queries, bootstrap CI |
| Retrieval MRR (hybrid, filtered) | **0.678** [0.624, 0.730] | |
| Effect of metadata routing | **+70% MRR** | 0.399 → 0.678, same eval set |
| Reranking on numeric queries | **+0.111 MRR** (p=0.017) | pooled effect is null — see §4d |
| Section coverage | 16/16 filings | Items 1, 1A, 7, 7A, 8 |
| Ground-truth facts (XBRL) | 156 across 11 metrics | 100% metric coverage |
| **Financial extraction accuracy vs XBRL** | **95.5% exact, 98.5% within 2%** | 134 figures, 16 filings |
| Statement-scope guard (FM-015) | 91.4% → 95.5% | JPM 71% → 100% |
| Extraction cost | **$0.0023 / filing** | grouped calls, 62% cheaper |
| Hallucination gate | numeric claims verified vs XBRL | fabricated figures stripped pre-display |
| Disclosure features → 90d excess return | **null (p=0.34–0.95)** | permutation-tested; see §7 |

---

## Quickstart

```bash
git clone <your-repo> && cd filingiq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# EDIT .env — put your real name and email in SEC_USER_AGENT.
# The SEC returns 403 for generic user agents. This is enforced, not advisory.

# Verify the ground-truth logic offline (no network needed)
pytest -q

# Ingest, then verify before scaling up
python scripts/01_ingest.py --tickers AAPL --limit 1
python scripts/02_inspect.py

# Parse filings into Items 1 / 1A / 7 / 7A / 8, then verify
python scripts/03_parse.py
python scripts/04_inspect_sections.py

# Chunk, embed, and evaluate retrieval
python scripts/06_chunk.py
python scripts/07_embed.py
python scripts/08_build_evalset.py
python scripts/09_eval_retrieval.py

# Then the full universe (~20 companies × 4 years, roughly 15 minutes at 6 req/s)
python scripts/01_ingest.py
```

## What Week 1 gives you

```
data/
├── raw/AAPL/2023/0000320193-23-000106_aapl-20230930.htm   # the filing
├── raw/AAPL/companyfacts.json                              # cached XBRL
└── db/filingiq.duckdb
     ├── companies
     ├── filings           # metadata + local paths
     ├── ground_truth      # ← the asset that makes this project credible
     ├── sections          # (Week 2)
     └── extractions       # (Week 4)
```

Query it directly:

```sql
-- The join that becomes your headline accuracy metric in Week 4
SELECT g.ticker, g.fiscal_year, g.metric,
       g.value AS truth, e.value AS extracted,
       abs(e.value - g.value) / nullif(abs(g.value), 0) AS rel_error
FROM ground_truth g
JOIN extractions e USING (accession, metric);
```

## Architecture

```
EDGAR API ──▶ Docling parse ──▶ hierarchical chunks ──▶ Qdrant (dense+sparse)
    │                                                          │
    └──▶ XBRL companyfacts ──▶ ground_truth ◀── verify ──── extraction agent
                                    │                          │
                                    ▼                          ▼
                            LangGraph supervisor ──▶ analyst memo (cited)
                                    │
                                    ▼
                       feature store ──▶ LightGBM ──▶ walk-forward backtest
```

Full design notes in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Design decisions

| Decision | Why |
|---|---|
| DuckDB over Postgres | Zero-config, columnar, reads Parquet natively, ships as one file in the repo |
| Ground truth keyed on `accn` | Captures what *that filing* said, not later restatements — the LLM read the original |
| Period matching with ±10 day tolerance | 52/53-week retail fiscal calendars shift year-ends |
| Tag fallback chains | ASC 606 changed revenue tagging in 2018; banks use different tag families entirely |
| Statistical verification, not LLM self-check | An LLM grading its own output is not evaluation |

## Known limitations

Stated honestly, because an interviewer will find them anyway and it's better if you name them first:

- 20 companies × 4 years is enough to demonstrate the method, not enough for statistically strong financial conclusions.
- Ground-truth coverage is weaker for financials (JPM, GS) — different us-gaap tag families.
- Only the primary document is downloaded; exhibits and some tables live in separate files.

## Docs

- [`EVALUATION.md`](EVALUATION.md) — metrics, methodology, ablations
- [`FAILURE_MODES.md`](FAILURE_MODES.md) — what broke and what I did about it

## Data source

All data from [SEC EDGAR](https://www.sec.gov/edgar), public domain. No API key required. Usage complies with the SEC's [fair access policy](https://www.sec.gov/os/webmaster-faq#developers) (descriptive User-Agent, ≤10 req/s).
