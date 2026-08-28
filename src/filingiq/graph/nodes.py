"""Graph nodes.

Each node does one thing, records what it did in `trace`, and either advances
the state or records why it could not. Nodes never raise: a failed node writes
to `errors` and the supervisor decides what happens next. That keeps a partial
result available instead of losing the whole run to one bad step -- the same
lesson as FM-016, applied at the orchestration layer.
"""
from __future__ import annotations

import logging
import time

from filingiq.analysis.claims import (
    annotate_unsupported, strip_unsupported_sentences, verify_claims,
)
from filingiq.analysis.diff import diff_years
from filingiq.graph.state import AnalysisState

log = logging.getLogger(__name__)

RISK_THEMES = {
    "supply_chain": ["supply chain", "supplier", "component", "manufactur",
                     "single source", "sole source", "logistics"],
    "cybersecurity": ["cyber", "data breach", "information security",
                      "unauthorized access", "ransomware"],
    "regulatory": ["regulat", "compliance", "antitrust", "governmental",
                   "legislation", "sanction"],
    "litigation": ["litigation", "lawsuit", "legal proceeding", "claim against"],
    "macro": ["inflation", "interest rate", "recession", "economic condition",
              "geopolitical"],
    "talent": ["personnel", "talent", "key employee", "retention", "labor"],
    "climate": ["climate", "environmental", "emission", "sustainability"],
    "concentration": ["concentration", "significant customer", "depend on a",
                      "substantial portion"],
    "liquidity": ["liquidity", "credit facility", "indebtedness", "covenant"],
    "ai_technology": ["artificial intelligence", "machine learning",
                      "generative ai"],
}


def _t(state: AnalysisState, msg: str) -> dict:
    log.info("[%s FY%s] %s", state.get("ticker"), state.get("fiscal_year"), msg)
    return {"trace": [f"{time.strftime('%H:%M:%S')} {msg}"]}


def load_figures_node(state: AnalysisState, con) -> dict:
    """Pull XBRL-verified figures. No LLM: these are already established."""
    rows = con.execute(
        "SELECT metric, value FROM ground_truth WHERE accession = ?",
        [state["accession"]],
    ).fetchall()
    values = {m: v for m, v in rows}
    return {"figures": values, "verified_values": values,
            **_t(state, f"loaded {len(values)} verified figures")}


def diff_node(state: AnalysisState, con, embedder) -> dict:
    """Year-over-year risk diff. Statistical, deterministic, no LLM."""
    prior = state.get("prior_year") or state["fiscal_year"] - 1
    try:
        result = diff_years(con, state["ticker"], state["fiscal_year"], prior,
                            embedder)
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"diff failed: {exc}"], **_t(state, "diff FAILED")}

    summary = result.summary()
    new = [{"chunk_id": e.current.chunk_id, "text": e.current.text[:1200],
            "similarity": round(e.similarity, 3)}
           for e in result.by_status("new") if e.current]
    # After the first year almost nothing is genuinely NEW -- filings carry
    # risk factors forward and rewrite them. Themeing only new risks therefore
    # reported "none" for 12 of 16 filings while 14-37 factors had been
    # rewritten. Modified risks are where the signal actually is.
    modified = [{"chunk_id": e.current.chunk_id, "text": e.current.text[:1200],
                 "similarity": round(e.similarity, 3)}
                for e in result.by_status("modified") if e.current]
    return {
        "diff_summary": summary, "new_risks": new, "modified_risks": modified,
        **_t(state, f"diff: {summary['new']} new, {summary['modified']} modified, "
                    f"drift={summary['drift_score']}"),
    }


def taxonomy_node(state: AnalysisState) -> dict:  # noqa: D401
    """Classify new risks into a fixed taxonomy by keyword.

    Deliberately not an LLM call. The taxonomy is closed and the vocabulary is
    stable, so keyword matching is deterministic, free, and auditable. An LLM
    here would add cost and variance for no accuracy that matters.
    """
    def classify(risks) -> dict[str, int]:
        out: dict[str, int] = {}
        for risk in risks or []:
            text = risk["text"].lower()
            for theme, keywords in RISK_THEMES.items():
                if any(k in text for k in keywords):
                    out[theme] = out.get(theme, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    new_themes = classify(state.get("new_risks"))
    modified_themes = classify(state.get("modified_risks"))
    return {"risk_themes": new_themes,
            "modified_themes": modified_themes,
            **_t(state, f"themes -- new: {new_themes or 'none'}, "
                        f"modified: {modified_themes or 'none'}")}


MEMO_SYSTEM = """You are writing a short analyst note on a company's 10-K.

Rules:
- Use ONLY the figures and risk excerpts provided. Introduce no other numbers.
- If you do not have a figure, do not estimate it -- omit the point entirely.
- State changes plainly. Do not editorialise or give investment advice.
- 150-250 words, plain prose, no headings, no bullet points.
- Write large figures readably ("$391.0 billion"), not as full digit strings.
- Quote per-share amounts to the cent.

Every number you write will be checked automatically against SEC XBRL data. \
Unverifiable figures are removed before the note is shown to anyone, so an \
invented number costs you the sentence containing it."""

MEMO_USER = """Company: {ticker}, fiscal year {fiscal_year}

Verified financial figures (USD):
{figures}

Year-over-year risk disclosure change vs FY{prior_year}:
  new risk factors: {n_new}
  modified: {n_modified}
  unchanged: {n_unchanged}
  drift score: {drift}{baseline_note}
  themes among new risks: {themes}

Excerpts from newly-added risk factors:
{new_risk_text}

Write the analyst note."""


def memo_node(state: AnalysisState, router) -> dict:
    budget = state.get("budget")
    if budget:
        blocked = budget.exceeded()
        if blocked:
            return {"warnings": [f"memo skipped: {blocked}"],
                    **_t(state, f"memo SKIPPED ({blocked})")}

    figures = state.get("figures", {})
    def fmt(v: float) -> str:
        if abs(v) >= 1e9:
            return f"{v/1e9:,.1f} billion (exactly {v:,.0f})"
        if abs(v) >= 1e6:
            return f"{v/1e6:,.1f} million (exactly {v:,.0f})"
        return f"{v:,.2f}"

    fig_lines = "\n".join(f"  {k}: {fmt(v)}" for k, v in sorted(figures.items())
                          if v is not None) or "  (none available)"
    diff = state.get("diff_summary", {})
    excerpts = "\n\n".join(f"- {r['text'][:600]}"
                           for r in state.get("new_risks", [])[:4]) or "  (none)"

    resp = router.complete(
        MEMO_SYSTEM,
        MEMO_USER.format(
            ticker=state["ticker"], fiscal_year=state["fiscal_year"],
            prior_year=state.get("prior_year", state["fiscal_year"] - 1),
            figures=fig_lines, n_new=diff.get("new", 0),
            n_modified=diff.get("modified", 0),
            n_unchanged=diff.get("unchanged", 0),
            drift=diff.get("drift_score") if diff.get("drift_score") is not None
                  else "undefined",
            baseline_note=("" if diff.get("has_baseline", True) else
                           "  (NO PRIOR-YEAR FILING IN CORPUS -- do not "
                           "describe this as change; say the baseline is "
                           "unavailable)"),
            themes=", ".join(state.get("risk_themes", {})) or "none",
            new_risk_text=excerpts),
        json_mode=False, max_tokens=1500)

    if budget:
        budget.calls_used += 1
        budget.tokens_used += resp.tokens_in + resp.tokens_out

    if resp.error:
        return {"errors": [f"memo generation failed: {resp.error[:120]}"],
                **_t(state, "memo FAILED")}
    return {"memo": resp.text.strip(), "budget": budget,
            **_t(state, f"memo drafted ({len(resp.text.split())} words)")}


def verify_node(state: AnalysisState) -> dict:
    """The hallucination gate. Arithmetic, not another model's opinion."""
    memo = state.get("memo", "")
    if not memo:
        return {"warnings": ["nothing to verify"], **_t(state, "verify skipped")}

    # Counts and the drift score are values THIS pipeline computed, so they
    # are as verifiable as XBRL figures. Excluding them made the gate strip
    # legitimate sentences citing our own analysis -- a false positive that
    # would erode trust in the flags.
    verifiable = dict(state.get("verified_values", {}))
    diff = state.get("diff_summary") or {}
    for key in ("new", "modified", "unchanged", "removed", "drift_score"):
        val = diff.get(key)
        if isinstance(val, (int, float)):
            verifiable[f"diff_{key}"] = float(val)

    report = verify_claims(memo, verifiable)
    summary = report.summary()
    annotated = annotate_unsupported(memo, report)
    cleaned = strip_unsupported_sentences(memo, report)

    return {
        "memo_verified": cleaned,
        "claim_report": {**summary,
                         "annotated": annotated,
                         "unsupported_examples": [c.raw for c in report.unsupported[:5]]},
        **_t(state, f"claims: {summary['supported']}/{summary['numeric_claims']} "
                    f"supported, {summary['unsupported']} removed"),
    }
