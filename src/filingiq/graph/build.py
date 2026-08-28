"""Graph assembly.

WHY LANGGRAPH RATHER THAN AGENTS TALKING TO EACH OTHER
------------------------------------------------------
The popular framing of "multi-agent" is several LLMs conversing until they
agree. That is expensive, non-deterministic, and nearly impossible to debug or
cost-bound.

This graph is an explicit state machine. Three of its five nodes make no LLM
call at all -- loading verified figures, computing the YoY diff, and
classifying risk themes are deterministic operations that would only become
slower and less reliable if delegated to a model. The one generative node is
bounded by an explicit budget and followed by an arithmetic verification gate.

"Multi-agent" is a description of the topology, not a licence to use an LLM at
every node. Knowing which steps do NOT need a model is the more valuable half
of the skill.
"""
from __future__ import annotations

import logging

from filingiq.graph.nodes import (
    diff_node, load_figures_node, memo_node, taxonomy_node, verify_node,
)
from filingiq.graph.state import AnalysisState, Budget

log = logging.getLogger(__name__)


def build_graph(con, embedder, router):
    """Wire the nodes into a LangGraph state machine."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(AnalysisState)

    g.add_node("load_figures", lambda s: load_figures_node(s, con))
    g.add_node("diff", lambda s: diff_node(s, con, embedder))
    g.add_node("taxonomy", taxonomy_node)
    g.add_node("memo", lambda s: memo_node(s, router))
    g.add_node("verify", verify_node)

    g.set_entry_point("load_figures")
    g.add_edge("load_figures", "diff")
    g.add_edge("diff", "taxonomy")

    def should_write_memo(state: AnalysisState) -> str:
        """Supervisor routing.

        Skip generation when there is nothing to say or no budget to say it
        with. A memo built from zero verified figures would consist entirely of
        unsupported claims -- the gate would strip it to nothing, having paid
        for the tokens first.
        """
        budget = state.get("budget")
        if budget and budget.exceeded():
            return "skip"
        if not state.get("figures"):
            return "skip"
        return "write"

    g.add_conditional_edges("taxonomy", should_write_memo,
                            {"write": "memo", "skip": END})
    g.add_edge("memo", "verify")
    g.add_edge("verify", END)

    return g.compile()


def analyse_filing(graph, ticker: str, fiscal_year: int, accession: str,
                   prior_year: int | None = None,
                   budget: Budget | None = None) -> dict:
    return graph.invoke({
        "ticker": ticker.upper(),
        "fiscal_year": int(fiscal_year),
        "prior_year": prior_year or int(fiscal_year) - 1,
        "accession": accession,
        "budget": budget or Budget(),
        "errors": [], "warnings": [], "trace": [],
    })
