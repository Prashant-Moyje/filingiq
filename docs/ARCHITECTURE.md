# Architecture

## Why these choices

**Ground truth via XBRL `accn` matching.** Each fact in the SEC companyfacts
payload names the filing that reported it. Keying on that gives per-filing truth
rather than a company-level time series, which is what makes extraction accuracy
measurable at all.

**Statistical verification over LLM self-evaluation.** An LLM asked "is this
correct?" agrees with itself. Numeric claims are checked against XBRL by
arithmetic comparison. Only genuinely unverifiable claims (narrative statements)
fall back to entailment checking.

**LangGraph over CrewAI/AutoGen.** Explicit state machines give bounded loops,
checkpointing, and inspectable state. Autonomous free-form agent chat is harder
to debug and impossible to cost-bound.

**Hierarchical chunking.** doc > item > subsection > paragraph, with metadata
(CIK, ticker, fiscal year, item number). Retrieval indexes small units and
returns parent context — small-to-big.

## Agent graph (Week 5)

```
                 ┌──────────────┐
                 │  Supervisor  │  routing, retries, budget caps
                 └──────┬───────┘
        ┌───────┬───────┼───────┬────────┐
        ▼       ▼       ▼       ▼        ▼
   Extraction  Risk    Diff  Verify  Synthesis
     Agent    Taxo.   Agent   Agent    Agent
        └───────┴───────┴───────┴────────┘
                shared Pydantic state
```

State is a single typed object. Every node reads and writes it; nothing passes
free-form strings between agents.

## Guardrails

1. Pydantic schema validation with retry-on-failure (max 2 retries, then abstain)
2. Citation enforcement — claims without a chunk ID are dropped before synthesis
3. Numeric cross-check against XBRL; mismatches flagged, never silently corrected
4. Confidence thresholds route low-certainty extractions to a human queue
5. Per-filing token budget with graceful degradation
