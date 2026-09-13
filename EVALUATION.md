# Evaluation

Every number here is reproducible from the scripts in this repo. Weak and
negative results are reported alongside good ones -- a portfolio that only
shows wins is evidence of selection, not of measurement.

---

## 1. Corpus (Weeks 1–2)

| | |
|---|---|
| Companies | 4 (AAPL, MSFT, WMT, JPM) — deliberately mixed sectors and filing styles |
| Filings | 16 × 10-K, FY2021–FY2024 |
| Sections extracted | 169 |
| Target-item coverage | 16/16 filings (Items 1, 1A, 7, 7A, 8) |
| Chunks | 6,054 (mean ≈ 390 tokens) |
| Chunks with malformed table markers | 0 |

Known gap: JPM's Item 7A has no standalone content — it is a cross-reference
into MD&A. Reported as unrecoverable rather than filled with the stub text.

## 2. Ground truth (Week 1)

Financial figures are verified against SEC XBRL company facts, keyed on the
accession number so each fact is the one *that filing* reported. 156 ground-truth
facts across 11 metrics, 100% metric coverage on all 16 filings.

This is what makes Week 4's extraction accuracy an objective measurement rather
than a self-assessment.

## 3. Retrieval evaluation set (Week 3)

182 queries in two families, scored separately because they measure different
things:

| Family | n | Relevance definition | Strength |
|---|---|---|---|
| Numeric | 64 | Chunk contains the XBRL figure, from the right filing | **Objective** — derived from SEC data, not judgment |
| Qualitative | 118 | Chunk is from the right filing + item **and** contains a topical keyword | **Proxy** — a keyword match is not proof of relevance |

Label set sizes: min 1, median 9, max 70.

> **Recall@5 is bounded above by 5 / |relevant| — 0.556 at the median label
> size, not 1.0.** Quoting recall without this bound overstates how far the
> system is from perfect. `hit@k` and `MRR` are the headline metrics for this
> label design.

## 4. Retrieval ablation

`BAAI/bge-small-en-v1.5` (384d, CPU), k=10, 50 candidates before fusion,
25 candidates to the cross-encoder. Reranker: `ms-marco-MiniLM-L-6-v2`.
All tables use the **same 182 queries**; only the stated variable changes.

### 4a. Metadata routing (the largest single effect)

Same eval set, same models — only ticker + fiscal-year pre-filtering toggled.

| Metric (hybrid) | No filter | With filter | Lift |
|---|---|---|---|
| MRR | 0.399 | **0.678** | +70% |
| hit@5 | 0.538 | **0.852** | +58% |
| nDCG@10 | 0.221 | **0.478** | +116% |

Sparse gains most from routing (MRR +144% vs dense's +82%), and this reverses
the numeric-query winner: unfiltered sparse *loses* to dense (R@5 0.120 vs
0.173), filtered it *wins* (0.515 vs 0.382). The context prefix
(`AAPL | FY2024 | Item 7`) was doing routing work badly — BM25 saturates on
those tokens and returns the right company but the wrong year. A metadata
filter does that job properly, leaving BM25 to do content matching, which is
what it is actually good at.

**Routing beat model selection.** No embedding upgrade available to this
project would deliver +70% MRR, and the filter costs nothing at query time.

### 4b. Retrieval modes (with metadata filtering)

| Mode | R@1 | R@5 | R@10 | hit@5 | nDCG@10 | MRR | ms/q |
|---|---|---|---|---|---|---|---|
| Dense | 0.088 | 0.236 | 0.341 | 0.830 | 0.448 | 0.648 | 115 |
| Sparse | 0.086 | 0.303 | 0.368 | 0.780 | 0.461 | 0.649 | **17** |
| **Hybrid** | 0.098 | 0.280 | 0.377 | **0.852** | 0.478 | **0.678** | 148 |
| Rerank | **0.110** | **0.306** | **0.415** | 0.835 | **0.493** | 0.670 | 2124 |

### 4c. Significance — paired bootstrap, 2000 resamples, vs hybrid

Pooled (n=182):

| Mode | MRR diff | p | hit@5 diff | p |
|---|---|---|---|---|
| Dense | −0.030 [−0.082, +0.021] | 0.264 | −0.022 | 0.521 |
| Sparse | −0.029 [−0.076, +0.017] | 0.215 | **−0.071** [−0.126, −0.016] | **0.012** |
| Rerank | −0.008 [−0.069, +0.052] | 0.757 | −0.016 | 0.633 |

**At n=182, only one pooled difference is significant: sparse-only is worse on
hit@5.** Dense, hybrid and rerank are statistically indistinguishable.

### 4d. Stratified significance — where the pooled test misleads

| Rerank vs hybrid | MRR diff | 95% CI | p | Verdict |
|---|---|---|---|---|
| Numeric (n=64) | **+0.111** | [+0.016, +0.203] | 0.017 | significantly better |
| Qualitative (n=118) | −0.073 | [−0.147, +0.001] | 0.056 | borderline worse |
| Pooled (n=182) | −0.008 | [−0.069, +0.052] | 0.757 | no difference |

Reranking lifts numeric hit@5 from 0.812 to **0.906** while degrading
qualitative retrieval by a comparable margin. The two effects cancel, so the
pooled test reports "no difference" for a system whose behaviour changed
substantially in both directions.

**Implication for the design:** reranking should be applied *conditionally* —
routed on query type — rather than globally. A query classifier that sends
numeric questions through the cross-encoder and qualitative ones straight from
RRF captures the gain without paying the cost or the regression. This is a
design conclusion the pooled table could not have produced.

**Why reranking may hurt qualitative queries.** Two candidate explanations,
untested: (a) `ms-marco-MiniLM` is trained on short web passages, not 390-token
financial prose; (b) the qualitative labels are a keyword *proxy*, so a
reranker optimising true relevance can move away from proxy agreement while
genuinely improving. Distinguishing these requires human relevance judgments.

### Caveats

**Multiple comparisons.** 24 tests were run (4 modes × 2 metrics × 3 strata).
At α=0.05 roughly one false positive is expected by chance. The rerank-numeric
result (p=0.017) would not survive Bonferroni correction (α=0.002); it is
reported as suggestive, and the pre-registered prediction that reranking helps
exact-answer retrieval makes it more credible than an unplanned finding. A
confirmatory run on a held-out query set is the correct next step.

**Sample size.** 182 queries over 4 companies. Resolving a 0.03 MRR gap needs
roughly 4× the queries — cheaper than any model change and the highest-value
next investment in this evaluation.

**Label quality.** Numeric labels are objective (XBRL-derived). Qualitative
labels are a keyword proxy and should not be read as measuring true relevance.

### Pending

- [ ] Confirmatory run on held-out queries for the rerank-numeric effect
- [ ] Model sweep: bge-small vs bge-base vs bge-m3 → accuracy/cost frontier
- [ ] Query-type classifier to route reranking conditionally

## 5. Extraction accuracy vs XBRL (Week 4) — headline metric

Model `openai/gpt-oss-120b` via Groq. 16 filings x 10 metrics = 160 figures.
Every figure compared arithmetically against the SEC's own XBRL data for that
accession. No LLM in the verification loop.

### Headline

| | |
|---|---|
| Figures attempted | 160 |
| Abstained (`found: false`) | 18 (11.3%) |
| Scored against ground truth | 134 |
| **Exact match** | **95.5%** |
| Within 2% | **98.5%** |
| Retrieval ceiling | 99.3% |
| Failed calls | 0 |
| Cost | $0.0023 / filing ($0.037 total) |
| Latency | p50 28.3 s, p95 47.2 s |

### Per company

Accuracy is measured over *scorable* figures. Metrics XBRL does not tag are
excluded rather than counted as failures — otherwise the companies whose
filings differ most from the norm are penalised for the ground truth's gaps,
not their own errors.

| Company | Correct / scored | Accuracy | Abstained | No ground truth |
|---|---|---|---|---|
| AAPL | 40/40 | 100% | 0 | 0 |
| MSFT | 40/40 | 100% | 0 | 0 |
| WMT | 32/32 | 100% (within 2%) | 8 | 0 |
| JPM | 20/22 | 91% | 10 | 8 |

JPM FY2021–23 scored 100% exact; FY2024 accounts for both remaining errors.

### Abstentions are correct behaviour

Every abstention coincided with an absent XBRL value — JPM and WMT do not
report R&D expense, and WMT does not tag total liabilities. The model declined
exactly where nothing existed to find, and never fabricated a figure to fill a
gap.

### Effect of the statement-scope guard (FM-015)

The largest single accuracy gain in Week 4.

| | Before | After |
|---|---|---|
| JPM FY2021 | 71% | **100%** |
| JPM FY2022 | 71% | **100%** |
| JPM FY2023 | 71% | **100%** |
| Overall exact match | 91.4% | **95.5%** |
| Unexplained errors | 9 | **2** |

Cause: JPM files Parent-Company-Only condensed statements (SEC Rule 12-04)
alongside consolidated ones, both correctly labelled "Total assets". The model
read the wrong scope on all four years at a near-constant ratio (0.152 / 0.152
/ 0.153 / 0.167) — a systematic wrong-table read, not model noise.

**Accounting identities cannot catch this.** Parent-only statements balance and
report the same equity as consolidated: 274,354 + 294,127 = 568,481 exactly. A
coherent wrong answer passes every arithmetic check. The working fix was
lexical — demoting chunks captioned "parent company only", "reportable
segment", or "variable interest entity", plus an explicit prompt instruction to
abstain rather than report a non-consolidated figure.

### Walmart: exact vs tolerance

WMT scores 88% exact but 100% within 2% on every year. The gap is rounding in
Walmart's own presentation, not extraction error — an argument for reporting
tolerance bands rather than a single exact-match figure.

### Retrieval vs generation

Retrieval ceiling 99.3%, end-to-end accuracy 81.9%. The correct figure was
present in context almost every time, so the residual gap is a generation
problem. Retrieval work cannot close it.

### Cost ablation: grouped vs per-metric calls

Metrics sharing a financial statement share one retrieval and one LLM call.

| | Per-metric | Grouped | Change |
|---|---|---|---|
| Calls per filing | 10 | 3 | −70% |
| Cost per filing | $0.0058 | $0.0022 | **−62%** |
| AAPL accuracy | 100% | 100% | none |
| MSFT accuracy | — | 100% | — |
| JPM accuracy | 86% | 71% | **−15 pts** |

**The cost optimisation is not free.** It is lossless on clean filings and
costly on JPM's 1.4M-character document, where more competing figures crowd a
shared context. `--no-group` retains the per-metric mode. Whether the scope
guard added in FM-015 recovers the JPM loss under grouping is the next
measurement.

### Retrieval anchors: query in the corpus vocabulary

Initial queries were built from metric *definitions* ("net cash generated by
operating activities for the year"). Rewriting them as the filing's own
line-item captions ("Cash generated by operating activities consolidated
statements of cash flows") moved AAPL FY2024 from 83.3% accuracy with 40%
abstention to 100% with zero, and lifted the retrieval ceiling from 60% to
100%. Retrieval matches text, so queries belong in the document's language,
not the question's.

### Pending

- [ ] Re-run with the statement-scope guard; measure JPM recovery
- [ ] Widen the corpus beyond 4 companies before treating 91.4% as stable
- [ ] `--k` ablation (4 / 6 / 8) for the accuracy-vs-latency frontier
- [ ] Local model comparison (Ollama) for the cost/accuracy frontier

## 6. Hallucination rate (Week 5)

Harness: `scripts/11_eval_hallucination.py`. Measurement logic is tested
offline (`tests/test_hallucination_eval.py`, 17 tests); the numbers below
require an API key and a built corpus.

    python scripts/11_eval_hallucination.py            # all three configs

| Configuration | Unsupported claims / memo (generated) | (shown to reader) |
|---|---|---|
| No verification — bare prompt, no gate | *pending* | *pending* |
| + XBRL cross-check — bare prompt, gate on | *pending* | *pending* |
| + citation enforcement — full prompt, gate on | *pending* | *pending* |

**Two columns, because one would be misleading.** Measured on what a reader
receives, rows 2 and 3 are 0.000 *by construction* — the gate deletes the
sentence containing an unsupported figure, so of course none survives. That
column demonstrates the gate is wired up, nothing more. The informative
quantity is how many bad claims the model **generated** before the gate saw
them, which is what separates "the instruction stopped it inventing figures"
from "the gate caught what the instruction missed".

**The baseline strips the deterrent.** The shipped `MEMO_SYSTEM` ends with
*"Every number you write will be checked automatically against SEC XBRL
data … an invented number costs you the sentence containing it."* That clause
is itself a mitigation, so a "no verification" row that kept it would measure
the gate's absence while the prompt quietly did the gate's job, and would
understate the raw rate. Configuration 1 uses `MEMO_SYSTEM_BARE`: the same
prompt with every reference to checking removed and every other rule intact,
so exactly one variable changes between rows. The user prompt is shared across
all three via `nodes.build_memo_prompt`.

**Scope.** Numeric claims only. "Management's tone has become more cautious" is
not checkable by arithmetic and is counted in neither direction — a memo can
score 0.000 here and still mislead in prose. This section should not be read as
a general hallucination rate.

## 7. Downstream model (Week 6) — a null result

**Question.** Do LLM-derived disclosure features predict 90-day excess returns
beyond what fundamentals provide?

**Answer: no, and that is the expected outcome.**

### Setup

| | |
|---|---|
| Observations | 155 company-years, 28 companies, FY2019–2024 |
| Target | 90-day excess return (stock minus SPY) from the **filing date** |
| Validation | Walk-forward: train on years ≤ t, test on year t+1 (4 folds) |
| Model | Ridge (α=10); ~40 features on ~130 training rows makes a boosted tree pure overfitting |
| Imputation & scaling | Fitted **within each fold** — banks lack `operating_income`, retailers lack `rnd_expense` |

### Ablation

| Feature set | Folds | n test | Mean IC | Std IC | Mean AUC | RMSE vs constant |
|---|---|---|---|---|---|---|
| Fundamentals | 4 | 103 | −0.066 | 0.163 | 0.465 | **−4.1%** |
| Disclosure | 4 | 103 | +0.091 | 0.112 | 0.520 | **−9.9%** |
| Combined | 4 | 103 | +0.009 | 0.065 | 0.454 | **−11.0%** |

**Every feature set is worse than predicting the training mean.** A model that
cannot beat a constant has not found a weak signal; it has found none and added
variance.

### Permutation test — 200 shuffles, target permuted within each year

| Feature set | Observed IC | Null mean ± σ | p | Verdict |
|---|---|---|---|---|
| Fundamentals | −0.066 | +0.001 ± 0.098 | 0.522 | indistinguishable from chance |
| Disclosure | +0.091 | +0.003 ± 0.104 | 0.343 | indistinguishable from chance |
| Combined | +0.009 | +0.001 ± 0.106 | 0.950 | indistinguishable from chance |

p-values use the (n+1) correction of Phipson & Smyth (2010), which counts the
observed arrangement as one of the permutations. The uncorrected estimator can
return p = 0.0 — a claim 200 shuffles cannot support — and reported 0.520 /
0.340 / 0.950 here. No conclusion changes; the floor is now 1/201 = 0.005.

**The disclosure IC of +0.091 is why this test is not optional.** It looks like
a usable signal — in quantitative finance an IC of 0.02–0.05 is genuinely
valuable — but the null distribution has σ = 0.104. The observed value sits
less than one standard deviation from chance. Reported without the null it
would have been a finding; reported with it, it is noise.

### Per-fold detail

| Test year | n train | n test | IC | AUC |
|---|---|---|---|---|
| 2021 | 52 | 25 | +0.118 | 0.660 |
| 2022 | 77 | 26 | −0.050 | 0.350 |
| 2023 | 103 | 26 | −0.002 | 0.405 |
| 2024 | 129 | 26 | −0.032 | 0.400 |

One positive fold and three flat-to-negative ones. An average carried by a
single lucky year is what overfitting looks like from the outside, and it is
invisible if only the mean is reported.

### Harness validation

The framework was calibrated on synthetic data before being trusted:

| Synthetic target | Mean IC | p |
|---|---|---|
| Planted signal (0.5 × drift + noise) | +0.896 | 0.000 |
| Pure noise | +0.136 | 0.167 |

Note that the **pure-noise run produced an IC of +0.136** — a large,
respectable-looking number generated entirely by chance on 26-row test folds.
That is the trap this layer exists to avoid.

### Interpretation

10-K risk-factor language is public, machine-readable, and parsed by every
quantitative fund with a text pipeline. A simple drift score predicting 90-day
excess returns would imply an inefficiency in one of the most heavily analysed
datasets in finance. The null is the theoretically expected result; a positive
would have warranted a hunt for leakage before celebration.

**What this does and does not show.** It does not show that disclosure change
is uninformative — only that *these* features, on *this* sample (155 rows,
6 years, 28 large-cap US issuers), do not predict *this* target at a horizon of
90 days. Plausible reasons the design could miss a real effect:

- **Power.** 155 rows over 4 test folds cannot resolve an IC of 0.02–0.05, the
  range where a real signal would live. Detecting that needs thousands of
  observations.
- **Horizon.** 90 days may be too long; disclosure changes plausibly resolve
  within days of filing.
- **Universe.** Large-cap US issuers are the most efficiently priced segment.
  Small caps receive far less analyst attention.
- **Target.** Excess return is noisy. Realised volatility, or the probability of
  a large adverse move, would be easier to predict and arguably more useful.
- **Measurement error in the predictor itself.** `drift_score` is not a
  validated measure of disclosure change — see below. A feature with
  substantial noise of its own attenuates any real correlation toward zero, so
  this null is consistent with both "no effect exists" and "the effect exists
  and the feature is too noisy to see it". The other four reasons above are
  about the study design; this one is about whether the input means what its
  name says.

### The predictor is unvalidated, and that is the weakest link

`drift_score` diffs **chunks**, not risk factors. Chunk boundaries fall at
cumulative token counts, so inserting one risk factor near the front of Item 1A
re-cuts everything after it. Text the company did not touch then lands in
differently-bounded chunks, and cosine similarity cannot distinguish that from
a rewrite.

Measured on a synthetic 40-factor Item 1A (40 factors → 20 chunks), inserting a
single new risk factor displaces this share of chunk boundaries:

| Insertion point | Chunks with identical text | Displaced |
|---|---|---|
| First | 0 / 20 | **100%** |
| 1/4 in | 4 / 20 | 80% |
| Middle | 9 / 20 | 55% |
| 3/4 in | 14 / 20 | 30% |
| Last | 19 / 20 | 5% |

One added factor, and between 5% and 100% of the section is no longer
byte-identical — determined by *where* the company inserted it, not *how much*
it changed. The published feature store has a median `drift_score` of 0.311.

**What this does and does not establish.** Displacement is a necessary
condition for spurious drift, not a sufficient one: a shifted chunk still
overlaps its neighbour heavily, so the embedder may well score it above the
0.95 "unchanged" threshold. The honest statement is that the upper bound on
contamination is large and the actual figure is **unmeasured**. Settling it
requires running the real embedder over displaced chunk pairs and reporting the
distribution of cosine against the 0.95/0.80 thresholds.

Those two thresholds are themselves asserted rather than calibrated. There is
no labelled set of "this risk factor was rewritten / was not", so no
precision-recall figure for the diff exists anywhere in this report.

**The design fix, if this matters to you:** segment Item 1A into risk factors
(they are delimited by bold or capitalised headings in most filings) and diff
factor-to-factor. Alignment then depends on content rather than on token
arithmetic, and inserting a factor changes exactly one unit.

### Why this section is in the report

The pipeline's measurable value is the extraction and verification layer:
**95.5% extraction accuracy against XBRL ground truth**, which stands on its
own. The return-prediction experiment asks whether the derived features carry
market-relevant information and finds no evidence at this sample size. Removing
a negative result because it is negative is how a portfolio becomes
unfalsifiable.

## 8. Cost & latency

| Component | Cost | Latency |
|---|---|---|
| Embedding 6,054 chunks (bge-small, CPU) | free | *see run log* |
| Dense query | free | 127 ms |
| BM25 query | free | 17 ms |
| Hybrid query | free | 159 ms |

## Reproducing

```bash
python scripts/01_ingest.py && python scripts/03_parse.py
python scripts/06_chunk.py && python scripts/07_embed.py
python scripts/08_build_evalset.py
python scripts/09_eval_retrieval.py --use-filters
```
