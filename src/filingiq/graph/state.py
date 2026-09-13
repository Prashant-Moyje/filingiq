"""Shared state for the analysis graph.

A single typed object passed between nodes. Nothing passes free-form strings:
every node reads and writes named, validated fields, so an inspection of the
state at any point tells you exactly what has been established and what has
not. This is the main practical reason to prefer LangGraph's explicit state
machine over free-form agent conversation -- it is debuggable.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


def _merge_lists(a: list, b: list) -> list:
    return (a or []) + (b or [])


class Budget(BaseModel):
    """Hard limits, enforced by the supervisor rather than hoped for.

    An agent graph without a budget is an unbounded loop with a credit card
    attached. Both counters are checked before every LLM node.
    """
    max_llm_calls: int = 12
    max_tokens: int = 40_000
    calls_used: int = 0
    tokens_used: int = 0

    def exceeded(self) -> str | None:
        if self.calls_used >= self.max_llm_calls:
            return f"call budget exhausted ({self.calls_used}/{self.max_llm_calls})"
        if self.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.tokens_used:,}/{self.max_tokens:,})"
        return None


class AnalysisState(TypedDict, total=False):
    """LangGraph state.

    IMPORTANT: keys a node returns that are NOT declared here are silently
    DROPPED between nodes. `modified_risks` was returned by the diff node,
    never declared, and therefore never reached the taxonomy node -- which
    reported "no themes" for every filing while 14-37 risk factors had been
    rewritten. Nothing errored; a whole feature was simply absent.

    Any new field a node produces must be added here.
    """
    # --- inputs ---
    ticker: str
    fiscal_year: int
    prior_year: int
    accession: str

    # --- populated by nodes ---
    figures: dict            # metric -> normalised value (XBRL-verified)
    verified_values: dict    # metric -> ground truth, for the claim gate
    diff_summary: dict       # YoY drift counts and score
    new_risks: list          # DiffEntry-derived, status == 'new'
    modified_risks: list     # status == 'modified' -- where the signal is
    risk_themes: dict        # theme -> count over NEW risks
    modified_themes: dict    # theme -> count over MODIFIED risks
    memo: str
    memo_verified: str
    claim_report: dict

    # --- control ---
    budget: Budget
    errors: Annotated[list, _merge_lists]
    warnings: Annotated[list, _merge_lists]
    trace: Annotated[list, _merge_lists]
    # NOTE: no `next_step`. One was declared and never read -- routing
    # is done by build.should_write_memo via conditional edges, not by a
    # field any node writes. A state key nothing sets or reads describes
    # a control flow the graph does not have.
