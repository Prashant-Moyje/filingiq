# FilingIQ

**Multi-agent SEC filing intelligence with XBRL-verified extraction.**

Every financial figure this system extracts is checked arithmetically against
the SEC's own XBRL data for that exact filing. Extraction accuracy is therefore
a **measured number, not a claim**.

```
216 filings · 28 companies · FY2018–2024 · 65,000 chunks · 204 tests · 19 documented failure modes
```

---

## Results

| Metric | Result | Notes |
|---|---|---|
| **Financial extraction accuracy vs XBRL** | **95.5% exact**, 98.5% within 2% | 134 figures, verified arithmetically |
| Cost per filing | **$0.0023** | grouped calls cut this 62% |
| Retrieval hit@5 | **0.852** [0.797, 0.901] | 182 queries, bootstrap CI |
| Effect of metadata routing | **+70% MRR** | 0.399 → 0.678, same eval set |
| Reranking on numeric queries | **+0.111 MRR** (p=0.017) | pooled effect is null — see §4d |
| Abstention correctness | **12/12** | declined only where no ground truth exists |
| Disclosure features → 90d excess return | **null** (p = 0.34–0.95) | permutation-tested; §7 |

Full methodology, ablations and confidence intervals: **[EVALUATION.md](EVALUATION.md)**
What broke and why: **[FAILURE_MODES.md](FAILURE_MODES.md)**

---

## The idea

Analysts cannot read every 10-K, and the highest-signal content is usually what
*changed* since last year — a risk factor quietly rewritten, a new disclosure
added, an old one dropped. Nobody tracks that systematically because it means
reading two 200-page documents side by side.

FilingIQ does three things:

1. **Extracts** financial figures and risk disclosures from 10-K filings.
2. **Verifies** every number against SEC XBRL data — no LLM grades another LLM.
3. **Diffs** risk factors year-over-year to produce a quantified disclosure-drift
   score, then tests whether that score predicts anything.

The third step returns a null result. It is reported anyway; see below.

---

## Architecture

```
EDGAR API ─────▶ Docling parse ─▶ sectioniser ─▶ hierarchical chunker
    │                                                    │
    │                                          Qdrant (dense + BM25 sparse)
    │                                                    │
    └──▶ XBRL companyfacts ──▶ ground_truth ◀── verify ──┴── extraction agent
                                    │                            │
                                    ▼                            ▼
                        LangGraph supervisor ────────▶ analyst memo (cited)
                                    │                            │
                                    │                    hallucination gate
                                    ▼                    (arithmetic, not LLM)
                       feature store ──▶ Ridge ──▶ walk-forward + permutation test
```

**Five nodes; three make no LLM call.** Loading verified figures, computing the
YoY diff, and classifying risk themes are deterministic. "Multi-agent" describes
the topology, not a licence to use a model at every step.

---

## Three design decisions worth explaining

**XBRL as ground truth.** Every fact in the SEC's companyfacts API carries the
accession number of the filing that reported it. That gives per-filing truth for
every financial figure, so extraction accuracy is measurable rather than
asserted. This is the decision the whole project rests on.

**The model never does arithmetic.** It reports `value_as_stated: 391035` and
`scale: "millions"` separately; Python multiplies. Asking an LLM to return
391035000000 introduces arithmetic errors for no benefit.

**Detection is statistical, explanation is generative.** YoY risk diffing uses
embeddings and cosine similarity, not a model asked "which of these 70 risk
factors are new?" The LLM only explains what the statistics found.

---

## Selected findings

**Routing beat model selection.** Filtering retrieval by ticker and fiscal year
lifted MRR from 0.399 to 0.678 — more than any embedding upgrade available, at
zero query-time cost.

**Reranking helps and hurts, and the aggregate hides both.** It significantly
improves numeric queries (+0.111 MRR, p=0.017) while degrading qualitative ones.
The two cancel, so the pooled test reports "no difference" for a system whose
behaviour changed substantially. The design conclusion — apply reranking
*conditionally* — is invisible without stratifying.

**A coherent wrong answer defeats consistency checking.** JPMorgan files
Parent-Company-Only statements alongside consolidated ones, both labelled "Total
assets". The model read the wrong scope on all four years. Checking
assets = liabilities + equity does not catch it: parent-only statements balance
and report identical equity. The fix had to be lexical. (FM-015)

**A null result, reported.** Disclosure-drift features do not predict 90-day
excess returns (p = 0.34–0.95, n=155). The disclosure-only IC of +0.091 looks
like signal until the permutation null shows σ = 0.104. Without that test it
would have been reported as a finding. §7 explains why the null is the
theoretically expected outcome.

---

## Quickstart

```bash
git clone <repo> && cd filingiq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set SEC_USER_AGENT to your name + email (required)

pytest -q                   # 204 tests, no network or API key needed

python scripts/01_ingest.py         # EDGAR + XBRL ground truth
python scripts/03_parse.py          # sections
python scripts/06_chunk.py          # retrieval units
python scripts/07_embed.py          # vector + BM25 indexes
python scripts/09_eval_retrieval.py --use-filters   # ablation + significance
python scripts/10_extract.py        # extraction, verified against XBRL
python scripts/13_analyze.py --dry-run              # YoY diff (no LLM cost)
python scripts/14_build_features.py && python scripts/15_train_model.py
```

Everything except `10_extract.py` and the memo node runs without an LLM.

## Demo

```bash
streamlit run app/main.py
```

Five tabs: corpus overview, per-figure extraction verification against XBRL,
year-over-year risk drift, corpus search, and the evaluation dashboard.

The app reads **only artifacts already on disk** — no LLM calls, no network. A
live demo that depends on a provider is a demo that fails during the one three-
minute window that matters.

## Stack

LangGraph · Qdrant · BAAI/bge-small-en-v1.5 · bge-reranker / MiniLM cross-encoder ·
rank-bm25 · DuckDB · Pydantic · Groq (`openai/gpt-oss-120b`) with an Ollama-capable
router · scikit-learn · pytest

---

## Known limitations

Stated here rather than left to be discovered:

- **28 of 216 filings fail Item 1A parsing** (GE, INTC, MCD and others use
  structures the sectioniser does not handle). They are excluded from the
  feature store, not silently scored as zero change — see FM-019.
- **155 modelling rows over 6 years** cannot resolve an IC in the 0.02–0.05
  range where a real signal would live. The null is honest but underpowered.
- **Qualitative retrieval labels are a keyword proxy**, not human relevance
  judgments.
- **Large-cap US issuers only** — the most efficiently priced segment, and the
  least likely to show a disclosure-based effect.
- JPM Item 7A has no separately locatable content; reported as an unrecoverable
  cross-reference stub rather than guessed at.

## Data

[SEC EDGAR](https://www.sec.gov/edgar), public domain, no API key. Usage complies
with the SEC fair-access policy (descriptive User-Agent, ≤10 req/s).
