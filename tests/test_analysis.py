"""Tests for YoY diffing and the claim-verification gate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from filingiq.analysis.claims import (  # noqa: E402
    annotate_unsupported, extract_claims, strip_unsupported_sentences,
    verify_claims,
)
from filingiq.analysis.diff import ChunkRef, DiffResult, diff_sections  # noqa: E402


TRUTH = {"revenue": 391_035_000_000.0, "net_income": 93_736_000_000.0}


# --- claim extraction ------------------------------------------------------

def test_years_are_not_treated_as_financial_claims():
    """A gate that flags 'fiscal 2024' as unverified gets ignored, and an
    ignored gate is worse than none."""
    claims = extract_claims("In fiscal 2024 and 2023 the company grew.")
    assert claims == []


def test_currency_amounts_that_look_like_years_are_kept():
    claims = extract_claims("Revenue was $2,024 million.")
    assert len(claims) == 1 and claims[0].value == 2_024_000_000


def test_scale_words_are_applied():
    c = extract_claims("Revenue of $391.0 billion")[0]
    assert c.value == pytest.approx(391_000_000_000)


def test_percentages_are_marked_and_excluded_from_numeric_checks():
    r = verify_claims("Gross margin was 46.2%.", TRUTH)
    assert r.summary()["numeric_claims"] == 0


def test_small_bare_integers_are_ignored():
    assert extract_claims("The company has 12 segments and 3 regions.") == []


# --- verification ----------------------------------------------------------

def test_supported_claim_matches_verified_value():
    r = verify_claims("Net sales were $391.0 billion.", TRUTH)
    assert r.summary()["unsupported"] == 0
    assert r.numeric_claims[0].matched_metric == "revenue"


def test_fabricated_figure_is_caught():
    r = verify_claims("The company disclosed $47.2 billion in charges.", TRUTH)
    assert r.summary()["unsupported"] == 1


def test_claim_matches_at_any_scale():
    """A memo may print the millions-scaled figure straight from the table."""
    r = verify_claims("Net sales were 391,035 million.", TRUTH)
    assert r.summary()["unsupported"] == 0


def test_rounding_within_tolerance_is_supported():
    r = verify_claims("Revenue reached $391.0 billion.", TRUTH)
    assert r.numeric_claims[0].supported


def test_no_verified_values_means_nothing_is_supported():
    """Fail closed: with no ground truth, no claim may be asserted as verified."""
    r = verify_claims("Revenue was $391.0 billion.", {})
    assert r.summary()["numeric_claims"] == 1
    assert r.summary()["unsupported"] == 1


def test_annotation_preserves_offsets():
    text = "A was $391.0 billion and B was $47.2 billion and C was $93.7 billion."
    out = annotate_unsupported(text, verify_claims(text, TRUTH))
    assert out.count("[UNVERIFIED]") == 1
    assert "$391.0 billion and" in out


def test_sentence_boundaries_survive_decimal_points():
    """Regression: splitting on a bare '.' cut '$47.2 billion' into the
    fragment 'Charges were $47.', which matched no real sentence -- so the
    strip silently removed nothing."""
    text = "Revenue was $391.0 billion. Charges were $47.2 billion."
    r = verify_claims(text, TRUTH)
    assert r.unsupported[0].sentence == "Charges were $47.2 billion."


def test_stripping_removes_the_whole_sentence():
    """Deleting only the number leaves a mutilated sentence that still reads
    as an assertion."""
    text = "Revenue was $391.0 billion. Charges were $47.2 billion."
    out = strip_unsupported_sentences(text, verify_claims(text, TRUTH))
    assert "391.0" in out and "47.2" not in out


# --- YoY diff --------------------------------------------------------------

class FakeEmbedder:
    """Maps marker words to fixed orthogonal vectors, so similarity is exact
    and the test asserts on the ALIGNMENT LOGIC rather than on a model."""
    VECTORS = {
        "supply": [1.0, 0.0, 0.0],
        "cyber": [0.0, 1.0, 0.0],
        "climate": [0.0, 0.0, 1.0],
        "supplyish": [0.92, 0.39, 0.0],   # ~0.92 cosine with 'supply'
    }

    def embed_passages(self, texts, show_progress=False):
        out = []
        for t in texts:
            # Longest key first: "supplyish" must not be matched by "supply".
            key = next((k for k in sorted(self.VECTORS, key=len, reverse=True)
                        if k in t), "supply")
            v = np.array(self.VECTORS[key], dtype="float32")
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def C(cid, text):
    return ChunkRef(cid, text, "1A", 0)


def test_identical_risk_is_unchanged():
    e = diff_sections([C("a", "supply chain")], [C("b", "supply chain")],
                      FakeEmbedder())
    assert e[0].status == "unchanged"


def test_rewritten_risk_is_modified_not_new():
    """The interesting case: same risk, escalated language. Classifying it as
    'new' would flood the report with false alarms every year."""
    e = diff_sections([C("a", "supplyish rewritten")], [C("b", "supply chain")],
                      FakeEmbedder())
    assert e[0].status == "modified"
    assert 0.80 <= e[0].similarity < 0.95


def test_genuinely_new_risk_is_flagged():
    e = diff_sections([C("a", "cyber attack")], [C("b", "supply chain")],
                      FakeEmbedder())
    assert e[0].status == "new"


def test_dropped_risk_is_reported_as_removed():
    e = diff_sections([C("a", "supply chain")],
                      [C("b", "supply chain"), C("c", "climate risk")],
                      FakeEmbedder())
    assert any(x.status == "removed" for x in e)


def test_first_filing_has_no_prior_year():
    e = diff_sections([C("a", "supply chain")], [], FakeEmbedder())
    assert e[0].status == "new"


def test_drift_score_reflects_change():
    stable = DiffResult("X", 2024, 2023, diff_sections(
        [C("a", "supply chain")], [C("b", "supply chain")], FakeEmbedder()))
    churned = DiffResult("X", 2024, 2023, diff_sections(
        [C("a", "cyber attack")], [C("b", "supply chain")], FakeEmbedder()))
    assert stable.summary()["drift_score"] == 0.0
    assert churned.summary()["drift_score"] == 1.0


# --- graph nodes and budget ------------------------------------------------

def test_budget_blocks_when_calls_exhausted():
    from filingiq.graph.state import Budget
    b = Budget(max_llm_calls=2, calls_used=2)
    assert "call budget" in (b.exceeded() or "")


def test_budget_blocks_when_tokens_exhausted():
    from filingiq.graph.state import Budget
    b = Budget(max_tokens=1000, tokens_used=1000)
    assert "token budget" in (b.exceeded() or "")


def test_memo_node_respects_budget_without_calling_the_model():
    """An agent graph without an enforced budget is an unbounded loop with a
    credit card attached. The block must happen BEFORE the call."""
    from filingiq.graph.nodes import memo_node
    from filingiq.graph.state import Budget

    called = {"n": 0}

    class Router:
        def complete(self, *a, **k):
            called["n"] += 1
            raise AssertionError("model called despite exhausted budget")

    out = memo_node({"ticker": "AAPL", "fiscal_year": 2024,
                     "budget": Budget(max_llm_calls=0)}, Router())
    assert called["n"] == 0
    assert any("budget" in w for w in out["warnings"])


def test_taxonomy_classifies_new_risks_without_an_llm():
    from filingiq.graph.nodes import taxonomy_node
    state = {"ticker": "X", "fiscal_year": 2024, "new_risks": [
        {"text": "Our reliance on a single source supplier for components"},
        {"text": "A cyber attack or data breach could disrupt operations"},
        {"text": "Increased regulatory scrutiny and antitrust litigation"},
    ]}
    themes = taxonomy_node(state)["risk_themes"]
    assert "supply_chain" in themes and "cybersecurity" in themes


def test_verify_node_strips_unsupported_claims_from_the_memo():
    from filingiq.graph.nodes import verify_node
    state = {"ticker": "AAPL", "fiscal_year": 2024,
             "memo": "Revenue was $391.0 billion. Charges were $47.2 billion.",
             "verified_values": {"revenue": 391_035_000_000.0}}
    out = verify_node(state)
    assert "391.0" in out["memo_verified"]
    assert "47.2" not in out["memo_verified"], "fabricated figure survived the gate"
    assert out["claim_report"]["unsupported"] == 1


def test_verify_node_handles_an_empty_memo():
    from filingiq.graph.nodes import verify_node
    out = verify_node({"ticker": "X", "fiscal_year": 2024, "memo": "",
                       "verified_values": {}})
    assert out["warnings"]


def test_nodes_record_a_trace():
    from filingiq.graph.nodes import taxonomy_node
    out = taxonomy_node({"ticker": "X", "fiscal_year": 2024, "new_risks": []})
    assert out["trace"] and isinstance(out["trace"], list)


# --- missing baseline ------------------------------------------------------

def test_no_prior_year_yields_undefined_drift_not_one():
    """The corpus starts at FY2021, so FY2021 has no baseline. Reporting
    drift=1.0 would teach a downstream model that every company overhauls its
    risk disclosure exactly once, in whichever year the corpus begins."""
    r = DiffResult("X", 2021, 2020,
                   diff_sections([C("a", "supply chain")], [], FakeEmbedder()),
                   has_baseline=False)
    s = r.summary()
    assert s["drift_score"] is None
    assert s["has_baseline"] is False


def test_baseline_present_gives_a_numeric_drift():
    r = DiffResult("X", 2024, 2023,
                   diff_sections([C("a", "cyber attack")],
                                 [C("b", "supply chain")], FakeEmbedder()),
                   has_baseline=True)
    assert r.summary()["drift_score"] == 1.0


# --- gate coverage of per-share figures ------------------------------------

def test_fabricated_eps_is_caught():
    """A currency symbol makes a small number a financial claim. Excluding
    everything under 1000 let '$6' pass against a true $6.08."""
    r = verify_claims("Earnings per share were $6.", {"eps_diluted": 6.08})
    assert r.summary()["unsupported"] == 1


def test_correct_eps_is_supported():
    r = verify_claims("Diluted EPS was $6.08.", {"eps_diluted": 6.08})
    assert r.summary()["unsupported"] == 0


def test_bare_counts_are_still_ignored():
    r = verify_claims("The company has 12 segments in 3 regions.", {"x": 1.0})
    assert r.summary()["numeric_claims"] == 0


def test_modified_risks_are_themed():
    """After year one nearly nothing is NEW, so theming only new risks
    reported 'none' for most filings while dozens were rewritten."""
    from filingiq.graph.nodes import taxonomy_node
    out = taxonomy_node({
        "ticker": "X", "fiscal_year": 2024, "new_risks": [],
        "modified_risks": [{"text": "A cyber attack could disrupt operations"}]})
    assert out["modified_themes"].get("cybersecurity") == 1


def test_every_node_output_key_is_declared_in_the_state_schema():
    """Guards the LangGraph footgun that cost a whole feature.

    Keys returned by a node but absent from the state TypedDict are dropped
    silently between nodes -- no error, no warning, just a missing field.
    `modified_risks` was returned, never declared, and the taxonomy node
    reported 'no themes' for all 16 filings while dozens of risks had been
    rewritten."""
    from filingiq.graph.state import AnalysisState

    declared = set(AnalysisState.__annotations__)
    produced = {
        "figures", "verified_values",           # load_figures_node
        "diff_summary", "new_risks", "modified_risks",   # diff_node
        "risk_themes", "modified_themes",       # taxonomy_node
        "memo", "budget",                       # memo_node
        "memo_verified", "claim_report",        # verify_node
        "errors", "warnings", "trace",          # any node
    }
    missing = produced - declared
    assert not missing, f"node outputs not declared in AnalysisState: {missing}"


def test_empty_current_year_gives_undefined_drift():
    from filingiq.analysis.diff import DiffResult, diff_sections
    r = DiffResult("BAC", 2023, 2022,
                   diff_sections([], [C(f"p{i}", "supply chain") for i in range(20)],
                                 FakeEmbedder()),
                   has_baseline=True, has_current=False)
    s = r.summary()
    assert s["drift_score"] is None, "unparsed year scored as zero change"
    assert "NOT 0.0" in s["note"]
    assert s["total_current"] == 0


def test_size_mismatch_gives_undefined_drift():
    from filingiq.analysis.diff import DiffResult
    r = DiffResult("T", 2019, 2018, [], has_baseline=True, has_current=True,
                   size_mismatch=True)
    assert r.summary()["drift_score"] is None


def test_summary_always_reports_all_three_quality_flags():
    """Every caller can tell WHY a drift score is missing without guessing."""
    from filingiq.analysis.diff import DiffResult
    for kwargs in ({"has_current": False}, {"has_baseline": False},
                   {"size_mismatch": True}, {}):
        s = DiffResult("X", 2024, 2023, [], **kwargs).summary()
        assert {"has_baseline", "has_current", "size_mismatch"} <= set(s)
