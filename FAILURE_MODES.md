# Failure Modes

> This is the highest-leverage document in the repo. Almost no portfolio project
> has one. It signals that you ran a real error analysis instead of stopping at
> the first working demo. Fill in a new entry every time something breaks --
> especially the ones you found embarrassing.

Template for each entry:

---

## FM-001: Comparative-year facts contaminate ground truth

**Symptom.** Extraction accuracy for revenue sat at ~66% even on filings where
the model's answer was visibly correct.

**Root cause.** A 10-K reports two to three years of comparative figures, and
every one of them carries the same `accn`. Naive selection of the first matching
fact returned the prior year.

**Fix.** Filter duration facts to those spanning 300–430 days AND whose
`end` date is within 10 days of the filing's `period_of_report`.

**How it's prevented from recurring.** `tests/test_xbrl.py::test_picks_the_annual_fact_not_the_comparative`
uses a fixture containing the exact trap.

**Cost of not catching it.** Every accuracy number for the whole project would
have been understated by ~30%, and the error analysis would have chased a
phantom "LLM hallucination" problem that didn't exist.

---

## FM-002: Table-of-contents excision swallowed the first real section

**Symptom.** Item 1A, 7, 7A and 8 all parsed correctly, but Item 1 was never
recovered from any filing. The pipeline reported 80% section coverage and
otherwise looked healthy.

**Root cause.** Two compounding errors in TOC detection.

First, the TOC span was computed as a fixed 6,000-character window anchored on
the first heading. The window extended past the end of the actual table of
contents and into the document body, so the real `Item 1. Business` heading was
inside the excised region and got dropped with the TOC rows.

Switching to an adjacency rule (group headings whose gaps are under 600 chars)
fixed the window problem but not the underlying one: the last TOC row and the
first body heading were only 48 characters apart, so they still grouped together.

**Fix.** Added a third, structural filter: a table of contents lists items in
ascending order, by definition. Truncate the run at the first point where item
numbering stops increasing. In the failing case the run read
`1, 1A, 1B, 1C, 2, 3, 5, 7, 7A, 8, 9A, 1` -- and that trailing `1` is the body
heading, identifiable without any distance threshold at all.

**Lesson.** The two distance-based heuristics both needed tuning and both broke
on plausible inputs. The ordering property is a fact about how 10-Ks are
constructed, so it needs no tuning and does not have a failure region. Prefer
structural invariants over thresholds.

**How it's prevented from recurring.**
`tests/test_sectioniser.py::test_item_1_is_the_body_section` and
`::test_cross_reference_does_not_split_item_1`.

**Known residual limitation.** The truncation rule also excludes the final
genuine TOC row from the excised span. In practice real 10-K contents lists run
through Item 15 or 16, so every item we target (1, 1A, 7, 7A, 8) sits safely in
the interior. Documented rather than fixed, because fixing it would reintroduce
a threshold.

---

## FM-003: lxml silently rejected filings with an XML declaration

**Symptom.** 7 of 16 filings reported `0 tables` and logged
`lxml parse failed (Unicode strings with encoding declaration are not
supported) -- using regex fallback`. Section extraction still "succeeded" at
100% coverage, so the summary output looked healthy.

**Root cause.** `html_to_text()` decoded the file to a `str` before handing it
to `lxml.html.fromstring()`. Filings that begin with
`<?xml version="1.0" encoding="UTF-8"?>` cause lxml to refuse a decoded string,
because the declaration's stated encoding can no longer be honoured. The
exception was caught by the fallback handler, which degraded to regex
extraction -- and regex extraction discards every table.

**Impact.** Every financial table in those 7 filings was shredded into
context-free numbers. Item 8 for MSFT (all years) and AAPL FY2024 contained
figures with no row or column labels attached. Had this gone unnoticed until
Week 4, the extraction agent would have shown poor accuracy on exactly those
filings and the obvious hypothesis -- "the model is bad at reading tables" --
would have been wrong.

**Fix.** Pass bytes to lxml; decode only on the regex fallback path.

**Lesson.** The fallback was too quiet. A degraded path that logs at WARNING
and then reports success looks identical to a healthy run in the summary line.
The `n_tables` column in the inspection script is what surfaced it -- which is
the argument for inspecting intermediate artifacts rather than only end metrics.

---

## FM-004: Wrapper filings -- numbered items are stubs, content is appended

**Symptom.** All four JPM filings missing Items 7, 7A and 8, while Item 15
absorbed ~167,000 words. A first fix (title-based fallback) changed nothing.

**Root cause -- and why the first hypothesis was wrong.** I assumed the item
headings were absent. A diagnostic dump of every heading position proved
otherwise: `Item 7` WAS matched, at 28.1% through the document. The problem was
what followed it:

> Item 7. Management's Discussion and Analysis... appears on pages 52-167.

JPMorgan's 10-K is a wrapper. Every numbered item is a few-hundred-character
cross-reference stub, and the substantive content is an annual report appended
*after* Item 15. So the stubs were found, then correctly rejected by the
minimum-length filter -- and with no heading after Item 15, Item 15 ran to the
end of the document.

My first fix searched for titles in the window between the numbered items,
which contains only stubs. It could never have worked. Writing the diagnostic
before the second attempt was the difference between fixing this and guessing
at it twice more.

**Fix.** Three-part recovery:

1. *Detect the wrapper* by its signature -- target items found by number but
   rejected as too short. No new heuristic needed; the stub rejection IS the
   signal.
2. *Search the trailing region* (everything after the last numbered item) with
   a line-anchored but not end-anchored pattern, since content headings run on
   into the opening sentence.
3. *Reject the second table of contents.* The appended report has its own index
   whose rows match the same title patterns. Real content and index rows are
   separable by what follows them -- pipes and digit density -- with no
   position tuning:

       index row : "...: | 172 | Consolidated Financial Statements 52 | ..."
       real start: "The following is Management's discussion and analysis of..."

Recovered sections are tagged `detect_method = 'wrapper-recovery'`.

**Item 7A remains unrecoverable, by design.** JPM's market-risk disclosure has
no standalone heading -- it lives inside MD&A. The system reports it as a
cross-reference stub rather than returning the stub text and calling it a
section. Reporting an honest gap beats emitting a plausible-looking wrong
answer that would silently corrupt every downstream metric.

**Related bug found along the way.** The apostrophe in "Management's" arrives
mojibaked as "ManagementÆs". My first pattern used `[^\w\s]` to absorb
apostrophe variants -- but Æ, â, €, ™ are LETTERS to Unicode's `\w`, so the
class matched none of them. Caught by a parametrised test over the variants.

**Lesson.** Two failed fixes came from reasoning about the document instead of
looking at it. `scripts/05_diagnose.py` exists so the next structural surprise
costs twenty minutes rather than two rounds of speculation.

**Prevention.** `tests/test_sectioniser.py` has a wrapper fixture built from
JPM's real structure, including the stub items, the second table of contents,
and the mojibaked apostrophe.

---

## FM-005: Cross-reference detector condemned real short sections

**Symptom.** After adding stub detection, JPM's Items 2 (Properties) and 9A
(Controls and Procedures) vanished from all four filings. They had parsed
correctly the run before.

**Root cause.** The detector flagged any section under 400 words containing
referring language plus a page number. But real sections cite other sections
constantly -- JPM's Item 9A is 328 words of genuine content that happens to end
"Refer to page 168 for further detail." Referring language alone cannot
separate a pointer from a section that merely contains one.

**Fix.** Dropped the ceiling to 150 words. JPM's actual Item 7A stub is 87
words; Items 2 (351) and 9A (328) now sit safely above the line. The
distinguishing property is not that a stub refers elsewhere -- it is that the
reference is essentially ALL there is.

**Lesson.** The first version optimised recall on stubs and paid for it in
precision on real sections. For a parser, a false positive is worse: a dropped
real section is visibly missing, while a fabricated one is silently wrong.

---

## FM-006: Orphan table markers from section boundaries inside tables

**Symptom.** 40 of 6,058 chunks (0.7%) carried unbalanced `[TABLE]` markers --
surfaced by a self-check in the chunking script, not by any test.

**Root cause.** Filers wrap page-number footers in tables, so an item heading
sometimes sits inside a table row. The sectioniser cuts at the heading, leaving
the section starting with a dangling `[/TABLE]` or ending with a dangling
`[TABLE]`. The orphan matched no balanced pair, flowed into a prose block, and
landed in a chunk.

**Impact.** A chunk containing `[/TABLE]` with no opening marker signals to a
retrieval model that a table is present when none is, and breaks any downstream
table-aware processing.

**Fix.** Strip orphan markers in `_prose_blocks`. Anything reaching that
function is an orphan by construction -- balanced pairs are consumed earlier.

**Lesson.** The synthetic test fixtures were all well-formed, so the unit tests
passed while 40 real chunks were malformed. The bug was caught only because the
chunking script asserts an invariant over the actual corpus. Both layers are
necessary: tests for logic, corpus assertions for the inputs you did not
imagine.

---

## FM-007: BM25 indexed the wrong field, and my hypothesis was wrong anyway

**Symptom.** Sparse retrieval lost badly to dense on NUMERIC queries
(MRR 0.157 vs 0.326) -- the exact opposite of the prediction the hybrid design
was built on.

**Root cause, part one (a bug).** The BM25 index was built over `raw_text`
rather than the context-prefixed `text`. The prefix carries
`AAPL | FY2024 | Item 7`, so the ticker and fiscal year were invisible to
lexical search. Those are precisely the rare, exact tokens a question supplies.

**Root cause, part two (a wrong belief, which matters more).** I argued BM25
would win on numeric queries because filings contain exact figures like
"391,035" that embeddings blur away. That is true of the DOCUMENTS and
irrelevant to the QUERIES. The user asks "what was Apple's revenue in FY2024?"
-- they do not know the number; that is why they are asking. The figure lives
in the answer, never in the question, so BM25's exact-match advantage was never
available on this query distribution.

Lexical retrieval would win on queries that quote text verbatim -- error-code
lookups, legal citations, "find the clause containing X". Financial Q&A is not
that shape.

**Fix.** Index the prefixed text.

**Outcome, and a correction to the above.** After the fix, sparse DOES beat
dense on numeric queries (Recall@5 0.515 vs 0.382). So the original prediction
was right and the stated reason was wrong. The advantage does not come from
figures -- those genuinely never appear in queries. It comes from the context
prefix supplying `AAPL`, `FY2024`, `Item 7`: rare, exact, high-IDF tokens that
a question DOES contain and that a 384-dimensional embedding compresses away.

**Lesson.** Two separate errors pointed the same direction and nearly cancelled
out. Had the index bug not existed, sparse would have won, I would have
declared my reasoning confirmed, and the wrong mechanism would have gone into
the writeup unchallenged. Being right for the wrong reason is the failure mode
an ablation is least likely to catch on its own -- it took a bug to expose it.

---

## FM-008: Label sets too large for Recall@k to mean anything

**Symptom.** Qualitative queries scored Recall@5 = 0.018 while hit@5 = 0.680.
The two numbers appear contradictory and one of them is uninformative.

**Root cause.** Section-anchored labels marked EVERY chunk of Item 1A as
relevant. A filing with 70 risk-factor chunks therefore has 70 relevant items,
and Recall@5 is bounded above by 5/70 = 0.071. The metric physically cannot
move, so it measures nothing about the retriever.

**Fix.** Two changes.

1. Qualitative relevance now requires a topical keyword ("supply chain",
   "cybersecurity") in addition to the right item, shrinking label sets to a
   handful and restoring the metric's ability to discriminate.
2. The eval script prints the label-size distribution and the arithmetic
   Recall@5 ceiling before any results, so the bound is never invisible.

**Lesson.** A metric can be computed correctly and still be meaningless. Both
numbers here were right; the label design made one of them uninterpretable.
Always check what your metric's maximum achievable value actually is on your
data before you quote it.

---

## FM-009: --limit sampled the head of the eval set, not a representative slice

**Symptom.** The reranking run reported label sets of `median=4 max=14`, while
the full eval set has `median=9 max=70`. Same data, same script, different
distribution.

**Root cause.** `--limit 60` took `queries[:60]`. The eval set is built numeric-
first, then qualitative, so the first 64 entries are ALL numeric. The reranker
was therefore evaluated exclusively on value-anchored numeric queries and
compared against baselines from the full mixed set.

**Impact.** Any rerank-vs-baseline claim from that run would be invalid --
two variables changed at once, and the difference in query mix is larger than
the effect being measured.

**Fix.** `--limit` now draws a stratified random sample, proportional across
query families, with a fixed seed so runs stay comparable. The sample
composition is printed before results.

**Lesson.** The convenience flag quietly became a confound. Anything that
subsets data for speed needs to preserve the distribution, or the speedup buys
a number that cannot be compared to anything -- which is worse than not having
run it.

---

## FM-010: Point estimates on 60 queries reversed the ranking

**Symptom.** Two runs of the same system, same configuration, opposite
conclusions:

| | dense MRR | hybrid MRR | winner |
|---|---|---|---|
| 182 queries | 0.648 | 0.678 | hybrid |
| 60-query subsample | 0.663 | 0.617 | dense |

**Root cause.** Every metric here is a mean over queries, so it has a sampling
distribution -- and the eval harness reported only point estimates. On 60
queries (21 numeric, 39 qualitative) an MRR gap of 0.03-0.05 sits well inside
the noise. The reversal is not a finding; it is what sampling variance looks
like when you forget to measure it.

**Why this one matters most.** Every other failure in this document produced a
visibly wrong artifact. This one produces a number that looks perfectly fine,
goes into a README, gets quoted in an interview, and is unreproducible. For a
data science portfolio that is the worst category of error.

**Fix.** Percentile bootstrap confidence intervals on every headline metric,
plus a PAIRED bootstrap test between modes. Paired, not independent, because
both systems answer the same queries and query difficulty is the dominant
variance component -- pairing removes it and roughly halves the interval.

The harness now prints "NOT distinguishable" when a difference's CI spans zero,
so the honest conclusion is the default rather than something you have to
remember to check.

**Consequence for the reported results.** The reranking gain on NUMERIC queries
(hit@5 0.762 -> 0.905) is large enough to survive n=21; the qualitative
DEGRADATION (MRR 0.626 -> 0.591) is not, and is reported as inconclusive
pending a larger run.

**Lesson.** "Is that difference significant?" is the first question a
competent interviewer asks about any comparison table. Not being able to answer
it undoes the credibility of every number above it.

---

## FM-011: A misconfigured provider produced a clean-looking results table

**Symptom.** Ran the extraction pipeline with `LLM_PROVIDER` still set to
`ollama` while Ollama was not running. Thirty calls failed with HTTP 404. The
script then printed a full accuracy report:

    Figures attempted      : 10
    Abstained (found=false): 10 (100.0%)
    Exact match            : 0.0%

**Root cause, two parts.**

1. No preflight check. The run started, failed every call, and treated each
   failure as an abstention -- which is a legitimate model behaviour, so the
   summariser had no way to distinguish "the model declined" from "there was
   no model".
2. HTTP 404 was retried three times with exponential backoff. A 404 means the
   endpoint does not exist; it will not exist on the third attempt either.
   Config errors were being handled as transient ones, turning a two-second
   failure into a six-minute one.

**Why this is the worst kind of bug in this project.** Every number in the
report was internally consistent and formatted correctly. Nothing crashed. If
the provider had been misconfigured subtly rather than totally -- pointing at a
weaker model, say -- the run would have completed and silently reported that
model's accuracy under the wrong label.

**Fix.**
- `ModelRouter.healthcheck()` runs one trivial call before any work starts; the
  script aborts with configuration guidance if it fails.
- 404 / 401 / invalid-key / model-not-found are classified as fatal and raised
  immediately instead of retried.
- The healthcheck's own tokens are excluded from the cost accounting.

**Lesson.** Abstention and failure look identical downstream unless you
distinguish them at the source. Any pipeline where "no answer" is a valid
output needs to separate *declined* from *errored*, or the error rate hides
inside the abstention rate.

---

## FM-012: A hardcoded model name expired

**Symptom.** Preflight failed immediately:

    404 - The model `llama-3.3-70b-versatile` does not exist or you do not
    have access to it.

**Root cause.** Groq deprecated its entire Llama chat family on 17 June 2026,
recommending `openai/gpt-oss-120b` as the replacement. The model name was
hardcoded as a default and had simply expired. Nothing in the codebase was
wrong when it was written.

**Why this deserves an entry rather than a one-line fix.** Provider catalogues
churn faster than project code. A pinned model name is a dependency with an
undocumented expiry date, and it fails as a 404 that looks like a bug in your
own program. The reflex fix -- swap in today's model name -- reproduces the
same failure on a later date.

**Fix.** Discover rather than assume.
- `list_models()` queries `/openai/v1/models` for what the key can actually
  reach today.
- `autoselect_model()` walks an ordered preference list and falls back to any
  chat-capable model rather than failing.
- Preflight detects `model_not_found`, queries the catalogue, and retries once
  automatically before giving up.
- `--list-models` lets a user see their options without reading a changelog.

**Interaction with FM-011 worth noting.** The preflight check added in FM-011
turned this into a two-second failure with an actionable message. Without it,
the run would have issued 160 calls, retried each three times, and printed a
"100% abstention" report thirty minutes later. Two independent robustness
fixes compounding is the argument for fixing infrastructure problems when you
find them rather than routing around them.

**Lesson.** Anything supplied by an external service -- model names, endpoints,
schemas -- should be discovered at runtime where the provider offers a
discovery endpoint. Hardcode it and you have written a time bomb with a
vendor-controlled fuse.

---

## FM-013: Reasoning models silently consumed the entire token budget

**Symptom.** After auto-recovering from the deprecated model (FM-012), the
preflight failed against `openai/gpt-oss-120b`:

    400 - Failed to validate JSON. Please adjust your prompt.
    'failed_generation': ''

**Root cause.** `gpt-oss` is a reasoning model: it emits internal
chain-of-thought before the answer, and those tokens count against
`max_tokens`. The healthcheck allowed 20 tokens. The model spent all 20
thinking, produced an empty completion, and Groq's JSON-mode validator
rejected the empty string.

**Why the error message points the wrong way.** "Failed to validate JSON.
Please adjust your prompt" describes a malformed prompt. The prompt was fine.
The budget was wrong. Acting on the message as written -- rewriting the prompt
-- would have wasted an hour and fixed nothing.

**Fix.**
- `is_reasoning_model()` detects gpt-oss / qwen3 / deepseek-r1 / o-series names.
- A 1024-token floor is applied to any reasoning model regardless of what the
  caller requested.
- `reasoning_effort="low"` is set for them: extraction is a reading task, not a
  puzzle, so minimal reasoning cuts latency and token spend. (Worth re-testing
  at "medium" as an ablation once accuracy numbers exist.)
- `json_validate_failed` is reclassified as retryable rather than fatal, since
  it usually indicates truncation rather than misconfiguration.

**Lesson.** Reasoning models change the cost model, not just the quality. Token
budgets sized for a non-reasoning model produce empty completions, and the
resulting provider error describes a symptom rather than the cause. When a
model swap breaks something, suspect the budget before the prompt.

---

## FM-014: Free-tier token budget exhausted, and the crash destroyed the run

**Symptom.** The 16-filing extraction run reached JPM FY2023 and hit
`429 rate_limit_exceeded: tokens per day (TPD): Limit 200000, Used 198150`.
The exception propagated and the process died, losing six filings of completed,
already-paid-for work.

**Root cause, two independent design errors.**

*Cost.* One LLM call per metric meant the same Item 8 context was retrieved and
transmitted ten times per filing. 16 filings x 10 metrics x ~3,300 tokens =
~528,000 tokens against a 200,000/day allowance. The architecture never fitted
the budget, and nothing in the code had ever computed whether it would.

*Durability.* Results were accumulated in memory and written once at the end.
Any failure in a run costing money and an hour of wall-clock discarded
everything before it.

**Fix.**

1. *Statement-grouped extraction.* Metrics living in the same financial
   statement share one retrieval and one call: income statement (5 metrics),
   balance sheet (4), cash flow (1). Ten calls become three, cutting tokens
   ~68%. Grouping is by statement rather than arbitrary batching so the shared
   context stays relevant to every metric in the group.
2. *Checkpointing.* Each filing's results are persisted the moment they exist.
   A resumed run skips completed filings automatically.
3. *Pre-flight budget estimate.* The token cost is computed and compared
   against the free-tier limit before any call is made, with concrete
   mitigations printed.
4. *Graceful 429 handling.* Rate limits stop the run cleanly and report what
   is safely checkpointed rather than raising.

**Trade-off accepted.** Per-metric calls gave clean blame attribution -- if
net income was wrong, only its own retrieval could be responsible. Grouping
sacrifices some of that for a 68% cost reduction. `--no-group` preserves the
diagnostic mode, and whether grouping costs accuracy is a stated ablation
rather than an assumption.

**Lesson.** Token budget is an architectural constraint, not an operational
detail discovered at runtime. Any pipeline whose unit of work costs money or
minutes needs checkpointing from the first version -- finding that out by
losing an hour is the expensive way to learn it.

---

## FM-015: Parent-Company-Only statements read as consolidated

**Symptom.** 8 of 9 extraction errors in the full run were JPMorgan's
`total_assets` and `total_liabilities`, wrong on all four years with
suspiciously stable ratios:

| Year | Extracted assets | True assets | Ratio |
|---|---|---|---|
| 2021 | $568.5B | $3,743.6B | 0.152 |
| 2022 | $555.7B | $3,665.7B | 0.152 |
| 2023 | $591.7B | $3,875.4B | 0.153 |
| 2024 | $669.6B | $4,002.8B | 0.167 |

A constant fraction across four independent filings is not an unreliable
model. It is a model reliably reading the wrong table.

**Root cause.** JPM's 10-K includes Parent-Company-Only condensed financial
statements (SEC Rule 12-04, Schedule I) alongside the consolidated ones. Those
tables are correctly labelled "Total assets" and "Total liabilities" -- they
simply cover the holding company rather than the group. Retrieval surfaced
them; the model read them faithfully.

**Why accounting identities did NOT catch it.** The obvious defence is
assets = liabilities + equity, which needs no ground truth. It fails here:

    274,354 + 294,127 = 568,481   exactly

Parent-only statements balance, and they report the SAME stockholders' equity
as the consolidated statements, so the identity holds precisely. **A coherent
wrong answer defeats consistency checking.** The check was kept -- it catches
other error classes -- but this case is documented as a known blind spot rather
than dressed up as a success.

**Fix.** The distinguishing signal is lexical, not arithmetic: the table
caption says which statement it is.

1. `deprioritise_non_consolidated()` demotes chunks mentioning "parent company
   only", "condensed financial information", "reportable segment", "variable
   interest entit", or "regulatory capital" to the back of the context.
   Demoted, not deleted -- models weight earlier excerpts more heavily, and
   deleting risks discarding a filing's only usable table.
2. The system prompt names these statement types explicitly and instructs
   abstention over reporting a non-consolidated figure.

**Why abstention was not the failure here.** All 12 abstentions in the run were
correct: JPM and WMT do not report R&D, and WMT does not tag total liabilities
in XBRL. Every abstention coincided with an absent ground-truth value. The
model declined exactly where there was nothing to find.

**Lesson.** Consistency checks verify that an answer is coherent, not that it
is the answer to the question asked. When a document contains multiple valid
answers at different scopes, disambiguation has to come from context -- the
caption, the surrounding text -- and no amount of arithmetic will substitute
for it.

---

## FM-016: A rate limit wrote fake results into the checkpoint

**Symptom.** Re-running JPM to test the FM-015 scope guard:

    JPM FY2021: exact=100%  (was 71% -- the fix worked)
    JPM FY2022: exact=100%  (was 71% -- the fix worked)
    JPM FY2023: exact=0%  abstained=10/10   <- rate limited
    JPM FY2024: exact=0%  abstained=10/10   <- rate limited

The aggregate reported 28% accuracy and 62.5% abstention. Both numbers are
meaningless: two real filings averaged with two whose calls never completed.
Worse, the failed filings were written to the checkpoint, so a resumed run
would have skipped them permanently.

**Root cause.** The router returns an `LLMResponse` carrying an `error` field
rather than raising, so the agent produced `found: false` figures for every
failed call. Downstream, `found: false` is a legitimate model behaviour --
abstention -- and nothing distinguished the two. The `break` intended to stop
the run never fired, because no exception ever propagated.

**This is FM-011 at a different layer.** That entry established that abstention
and failure must be distinguished, and fixed it at preflight. The same
conflation survived at the per-call level. Fixing an instance of a problem is
not the same as fixing the class of it.

**Fix.**
1. `Verdict` gains an `errored` band, separate from `abstained`. Failed calls
   are excluded from scoring entirely -- they are neither correct, wrong, nor
   declined.
2. A filing with any failed call is **never checkpointed**; its partial rows
   are discarded.
3. When most calls in a filing fail, the run aborts and reports what is safely
   checkpointed.
4. The summary prints failed-call count separately, flagged as incomplete.

**Why discarding beats keeping partial results.** A half-extracted filing in
the checkpoint is worse than an absent one: absent gets retried, partial gets
skipped forever and silently deflates every future aggregate.

**Result of the underlying experiment (the reason for the run).** The FM-015
scope guard raised JPM from 71% to 100% exact on both filings that completed.
Confirmation on the remaining two is pending quota.

**Lesson.** Any code path that converts an exception into a value must preserve
the distinction between "no answer because there is none" and "no answer
because we failed". Otherwise a silent outage is indistinguishable from a
well-behaved system being appropriately cautious -- and the second reads as
success.

---

## FM-017: Three ways generated output looked verified when it was not

Three defects found while building the Week 5 memo pipeline. Each produced
output that looked correct.

**(a) The safety gate silently removed nothing.**
`strip_unsupported_sentences()` matched sentences by string equality against
sentences located by `_sentence_around()`, which split on any `.` -- including
the decimal in "$47.2 billion". The located sentence was the fragment
"Charges were $47.", which matched no real sentence, so the strip removed
nothing while reporting success. Fixed by treating a sentence boundary as
`[.!?]` followed by whitespace or end-of-string.

**(b) The gate could not see per-share figures.**
Bare integers under 1000 were excluded to avoid flagging years and counts
("12 segments"). That exclusion also covered every per-share amount. A memo
stating "earnings per share were $6" against a true $6.08 passed untouched --
precisely the error class the gate exists to catch. Fixed by treating a
currency symbol as overriding the magnitude exclusion.

**(c) Missing baseline reported as total change.**
The corpus starts at FY2021, so FY2021 has no prior filing to diff against.
Every risk factor was classified "new" and the drift score came back 1.0 for
all four companies. That is a missing baseline, not a disclosure event.
Un-fixed, it would have entered the Week 6 feature store as a real observation
and taught the model that every company overhauls its risk disclosure exactly
once, in whichever year the corpus happens to begin. Drift is now `None` when
`has_baseline` is false, and the memo prompt is told to say so.

**Common thread.** None of these crashed. Each produced plausible, well-formed
output: an unmodified memo that claimed to be sanitised, a clean claim report
that had skipped the checkable figure, and a confident drift score computed
from nothing. The recurring risk in generative pipelines is not failure but
fluent, formatted, unwarranted confidence -- which is why every layer here
carries an independent arithmetic check rather than trusting the layer below.

---

## FM-018: LangGraph silently dropped an undeclared state field

**Symptom.** The taxonomy node reported `modified: none` for all 16 filings,
while the diff node had just reported 14 to 37 modified risk factors in each.
No error, no warning. The feature was simply absent from every result.

**Root cause.** `AnalysisState` is a `TypedDict`, and LangGraph propagates only
the keys declared in it. `diff_node` returned `modified_risks`; the schema had
never declared it; the key was discarded before `taxonomy_node` ran, which then
correctly reported no themes for an empty list.

**Why it went unnoticed.** "No new risk themes" is a plausible result -- after
the first year, filings genuinely add few brand-new risk factors. The output
was consistent with reality, just not with what the pipeline had computed. A
missing feature that produces believable output is far harder to spot than one
that crashes.

**Fix.** Declared `modified_risks` and `modified_themes` in the schema, and
added a test asserting that every key any node returns is declared. That test
catches the next instance of this at commit time rather than after a full run.

**Lesson.** Typed state is what makes an agent graph debuggable, but the same
typing silently discards anything undeclared. When a framework enforces a
schema, an assertion that producers and schema agree belongs in the test suite
-- otherwise the schema quietly becomes a filter instead of a contract.

---

## FM-019: An unparsed section scored as "nothing changed"

**Symptom.** After widening the corpus to 32 companies, the feature store
accepted rows like these:

    BAC FY2023: has no Item 1A chunks -> drift=0.0, "100 removed"
    GS  FY2020: has no Item 1A chunks -> drift=0.0, "93 removed"
    MCD FY2019: has no Item 1A chunks -> drift=0.0, "31 removed"
    NKE FY2024: has no Item 1A chunks -> drift=0.0, "102 removed"

**Root cause.** The sectioniser failed on those filings, so the current year
had zero Item 1A chunks. Every prior-year chunk was then classified "removed",
`total_current` was 0, and

    drift = (new + modified) / max(total_current, 1) = 0 / 1 = 0.0

A drift of 0.0 is a legitimate, meaningful value -- it means the company
carried its risk disclosure forward verbatim. So a parsing failure produced a
plausible observation that would have entered the training set as fact.

**The asymmetry that made this dangerous.** FM-017(c) already handled the
missing PRIOR year, which yields drift=1.0 -- conspicuous, and caught. The
mirror case yields 0.0, which is unremarkable, and was not.

**Fix.** Drift is `None` -- never a number -- under three conditions, each
reported separately so the reason is visible:

| Condition | Would have produced | Now |
|---|---|---|
| No usable prior year | 1.0 | `None` |
| Current year unparsed | **0.0** | `None` |
| Sections differ >3x in size | a meaningless number | `None` |

The size check catches a subtler variant: AT&T FY2018 parsed to 5 chunks
against FY2019's 33, giving drift=1.0 that described the parser rather than the
company. Both sides now need at least 5 chunks and a size ratio above 0.35.

**Also fixed: duplicate company-years.** JNJ filed twice for FY2023. Two rows
double-weight one observation and place identical data on both sides of a
walk-forward split boundary.

**Two defects found while writing the fix**, both caught by the tests: a
duplicated dict key in `summary()`, and the `size_mismatch` branch never being
checked -- so mismatches were flagged and then ignored.

**Lesson.** When a computation degrades, ask what value it degrades TO. If
that value is indistinguishable from a legitimate result, the failure is
invisible and will be treated as data. Failure modes that produce
*conspicuous* values get found; ones that produce plausible values get trained
on.

---

## FM-020: Eight weeks of green tests that were broken on Windows

**Symptom.** Three UI tests failed on the developer's Windows machine while
passing everywhere they had been written:

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in
    position 5774: character maps to <undefined>

**Root cause.** `Path.read_text()` with no `encoding` argument uses
`locale.getpreferredencoding()` -- UTF-8 on Linux and macOS, **cp1252** on most
Windows installs. `app/main.py` contains a curly quote in a docstring. cp1252
cannot decode it.

An AST audit found **17 such call sites** across `src`, `scripts`, `tests` and
`app`. Every one had been correct on the machine it was written on and latent
on the machine it would be run on.

**Why nothing caught it earlier.** CI runs Ubuntu. The suite was green for
eight weeks. The bug was not in any tested behaviour -- it was in how the test
harness read a file, on a platform CI never exercised.

**Fix.**
- Explicit `encoding="utf-8"` on all 17 sites.
- `tests/test_portability.py` walks the AST of every source file and fails if
  any `read_text` / `write_text` / text-mode `open()` omits an encoding. Grep
  cannot do this -- the calls span multiple lines -- but the AST can.

**Note.** PEP 686 makes UTF-8 the default in Python 3.15, which will retire
this whole class of bug. Until then it has to be written out.

**Lesson.** "Works on my machine" has a specific technical form: default
encodings, path separators, and line endings differ by platform, and none of
them appear in your test output until someone else runs your code. A static
check over the AST costs nothing per run and catches the entire class rather
than the instance that happened to surface.

---

## FM-021: (yours goes here)

Candidates you will almost certainly hit:
- Banks (JPM, GS) missing most metrics — different us-gaap tag families
- Retailers with 52/53-week fiscal calendars shifting period ends
- Tables split across page boundaries during parsing (Week 2)
- Item 7 sectioniser tripping on the table-of-contents entry rather than the
  actual section heading (Week 2)
- Filings that exceed context window (Week 4)
- Retrieval returning the risk-factor *summary* instead of the detailed clause (Week 3)

