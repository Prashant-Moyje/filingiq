"""Build a retrieval evaluation set with objectively-derived labels.

THE LABELLING PROBLEM
---------------------
Retrieval metrics need relevance judgments: for each query, which chunks are
correct answers? Hand-labelling 100 queries against 6,000 chunks is hours of
work, and worse, the labels reflect one person's opinion of relevance.

THE SHORTCUT THIS PROJECT HAS
-----------------------------
XBRL gives us the true value of every financial metric for every filing. So for
"What was Apple's total revenue in fiscal 2024?", a chunk is relevant if it
contains 391,035 (or 391.0 billion, or 391035000000) AND comes from that
filing. That is a fact, not a judgment -- reproducible, auditable, and free.

TWO RELEVANCE MODES, REPORTED SEPARATELY
----------------------------------------
value-anchored (strict): the chunk contains the actual figure. Precise, but
    only applicable to numeric questions.
section-anchored (loose): the chunk comes from the right filing and item.
    Weaker, but works for qualitative questions about risk factors and MD&A.

Reporting both, and being explicit that the loose mode is a proxy, is the
honest thing to do. An interviewer who probes your evaluation methodology --
and a good one will -- is far more impressed by "here is exactly what my labels
do and do not prove" than by a single unqualified number.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

log = logging.getLogger(__name__)

METRIC_PHRASES = {
    "revenue": "total revenue",
    "net_income": "net income",
    "operating_income": "operating income",
    "eps_diluted": "diluted earnings per share",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "stockholders_equity": "stockholders equity",
    "cash_and_equivalents": "cash and cash equivalents",
    "operating_cash_flow": "net cash provided by operating activities",
    "rnd_expense": "research and development expense",
}


@dataclass
class EvalQuery:
    query_id: str
    question: str
    ticker: str
    fiscal_year: int
    accession: str
    kind: str                       # 'numeric' | 'qualitative'
    relevance_mode: str             # 'value-anchored' | 'section-anchored'
    expected_items: list[str] = field(default_factory=list)
    expected_value: float | None = None
    metric: str | None = None
    relevant_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def value_variants(value: float, metric: str) -> list[str]:
    """Every plausible textual rendering of an XBRL figure.

    Filings state the same number as 391,035 (millions), 391.0 billion, or
    occasionally in thousands. Missing a variant understates recall by
    mislabelling correct chunks as irrelevant -- a bug that would make your
    retriever look worse than it is.
    """
    out: set[str] = set()
    if value is None:
        return []
    av = abs(value)

    if metric.startswith("eps"):
        out.add(f"{value:,.2f}")
        out.add(f"{value:.2f}")
        return sorted(out)

    for scale, _label in ((1e6, "millions"), (1e3, "thousands"), (1.0, "units")):
        scaled = av / scale
        if scaled >= 1:
            out.add(f"{scaled:,.0f}")
            if scaled < 10000:
                out.add(f"{scaled:,.1f}")
    if av >= 1e9:
        out.add(f"{av / 1e9:,.1f}")
        out.add(f"{av / 1e9:,.2f}")
    return sorted(v for v in out if len(v.replace(",", "")) >= 3)


def find_value_anchored_chunks(con, accession: str, value: float,
                               metric: str) -> list[str]:
    """Chunks from this filing containing any rendering of the figure."""
    variants = value_variants(value, metric)
    if not variants:
        return []
    clauses = " OR ".join("raw_text LIKE ?" for _ in variants)
    params = [accession] + [f"%{v}%" for v in variants]
    rows = con.execute(
        f"SELECT chunk_id FROM chunks WHERE accession = ? AND ({clauses})",
        params,
    ).fetchall()
    return [r[0] for r in rows]


def build_numeric_queries(con, max_per_filing: int = 4) -> list[EvalQuery]:
    """Generate questions whose answers XBRL already knows."""
    rows = con.execute("""
        SELECT g.accession, g.ticker, g.fiscal_year, g.metric, g.value
        FROM ground_truth g
        JOIN filings f ON g.accession = f.accession
        WHERE g.metric IN ({})
        ORDER BY g.ticker, g.fiscal_year, g.metric
    """.format(",".join("?" for _ in METRIC_PHRASES)),
        list(METRIC_PHRASES)).fetchall()

    per_filing: dict[str, int] = {}
    queries: list[EvalQuery] = []

    for accession, ticker, fy, metric, value in rows:
        if per_filing.get(accession, 0) >= max_per_filing:
            continue
        chunks = find_value_anchored_chunks(con, accession, value, metric)
        if not chunks:
            # No chunk contains the figure. Usually means the number lives in a
            # table that was scaled differently. Skip rather than emit a query
            # with no possible correct answer -- that would drag recall to zero
            # for reasons unrelated to the retriever.
            continue
        per_filing[accession] = per_filing.get(accession, 0) + 1
        phrase = METRIC_PHRASES[metric]
        queries.append(EvalQuery(
            query_id=f"num_{ticker}_{fy}_{metric}",
            question=f"What was {ticker}'s {phrase} in fiscal year {fy}?",
            ticker=ticker, fiscal_year=fy, accession=accession,
            kind="numeric", relevance_mode="value-anchored",
            expected_items=["7", "8"], expected_value=float(value),
            metric=metric, relevant_chunks=chunks,
        ))
    return queries


# Each template carries KEYWORDS used to narrow the relevant set. Marking every
# chunk in Item 1A as relevant produces label sets of 70+ chunks, which caps
# Recall@5 at 0.07 and makes the metric unable to move regardless of retriever
# quality. Requiring a topical keyword shrinks label sets to a handful and
# restores the metric's discriminating power. See FM-008.
QUALITATIVE_TEMPLATES = [
    ("supply_chain", "What supply chain risks does {ticker} disclose?", ["1A"],
     ["supply chain", "supplier", "component shortage", "manufacturing partner",
      "single source", "sole source"]),
    ("cybersecurity", "What cybersecurity risks does {ticker} identify?", ["1A", "1C"],
     ["cybersecurity", "cyberattack", "data breach", "information security",
      "unauthorized access", "malicious"]),
    ("competition", "How does {ticker} describe competition in its industry?", ["1", "1A"],
     ["competit", "competitors", "market share"]),
    ("regulation", "What regulatory or legal risks does {ticker} face?", ["1A"],
     ["regulat", "antitrust", "compliance", "litigation", "governmental"]),
    ("gross_margin", "What does {ticker} say about gross margin drivers?", ["7"],
     ["gross margin", "cost of sales", "cost of revenue"]),
    ("segments", "What are {ticker}'s reportable business segments?", ["1", "7"],
     ["reportable segment", "segment", "operating segment"]),
    ("liquidity", "How does {ticker} describe its liquidity and capital resources?", ["7"],
     ["liquidity", "capital resources", "credit facility", "commercial paper"]),
    ("fx_risk", "What foreign currency exchange risk does {ticker} report?", ["7A", "1A"],
     ["foreign currency", "exchange rate", "foreign exchange", "currency fluctuation"]),
]

# A label set this large means the query is trivially satisfiable and will
# flatter the metrics. Flagged in the build report rather than silently kept.
LABEL_SET_WARN_THRESHOLD = 40


def build_qualitative_queries(con) -> list[EvalQuery]:
    """Section-anchored queries: relevance is 'right filing, right item'.

    This is a PROXY for relevance, not relevance itself -- a chunk can be from
    Item 1A and still not discuss supply chains. It is stated as a proxy in
    EVALUATION.md and used for comparing retrieval MODES against each other,
    where the bias applies equally to every mode.
    """
    filings = con.execute("""
        SELECT DISTINCT accession, ticker, fiscal_year FROM filings
        WHERE local_path IS NOT NULL ORDER BY ticker, fiscal_year
    """).fetchall()

    queries: list[EvalQuery] = []
    for accession, ticker, fy in filings:
        for slug, template, items, keywords in QUALITATIVE_TEMPLATES:
            placeholders = ",".join("?" for _ in items)
            kw_clause = " OR ".join("lower(raw_text) LIKE ?" for _ in keywords)
            rows = con.execute(
                f"SELECT chunk_id FROM chunks WHERE accession = ? "
                f"AND item IN ({placeholders}) AND ({kw_clause})",
                [accession] + items + [f"%{k.lower()}%" for k in keywords],
            ).fetchall()
            # Fewer than 2 matches usually means the topic genuinely is not
            # discussed; a query with no findable answer measures nothing.
            if len(rows) < 2:
                continue
            queries.append(EvalQuery(
                query_id=f"qual_{ticker}_{fy}_{slug}",
                question=template.format(ticker=ticker) + f" (fiscal year {fy})",
                ticker=ticker, fiscal_year=fy, accession=accession,
                kind="qualitative", relevance_mode="section-anchored",
                expected_items=items, relevant_chunks=[r[0] for r in rows],
            ))
    return queries


def save(queries: list[EvalQuery], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([q.to_dict() for q in queries], fh, indent=2)


def load(path: Path | str) -> list[EvalQuery]:
    with open(path, encoding="utf-8") as fh:
        return [EvalQuery(**d) for d in json.load(fh)]
