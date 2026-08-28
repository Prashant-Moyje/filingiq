"""Tests for the 10-K sectioniser.

The fixture below is a miniature 10-K that reproduces the exact traps real
filings set:

  TRAP 1  A table of contents listing every item, near the front.
  TRAP 2  Cross-references in body text ("as discussed in Item 1A").
  TRAP 3  Punctuation variation: "Item 1A.", "ITEM 7 -", "Item 7A:".
  TRAP 4  'Item 1' must not match the start of 'Item 1A'.

If the sectioniser passes these, it will handle most real filings. If it
fails any, it would have silently produced a 40-character 'Risk Factors'
section and every downstream metric would be meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.parsing.html_text import html_to_text, normalise_whitespace  # noqa: E402
from filingiq.parsing.sectioniser import (  # noqa: E402
    detect_toc_span,
    find_candidates,
    sectionise,
)


def _filler(word: str, n: int) -> str:
    return " ".join([word] * n)


# --- TRAP 1: a realistic table of contents ---------------------------------
TOC = "\n".join(
    [
        "TABLE OF CONTENTS",
        "Item 1. Business ... 3",
        "Item 1A. Risk Factors ... 12",
        "Item 1B. Unresolved Staff Comments ... 28",
        "Item 1C. Cybersecurity ... 29",
        "Item 2. Properties ... 30",
        "Item 3. Legal Proceedings ... 31",
        "Item 5. Market for Registrant's Common Equity ... 33",
        "Item 7. Management's Discussion and Analysis ... 35",
        "Item 7A. Quantitative and Qualitative Disclosures ... 52",
        "Item 8. Financial Statements and Supplementary Data ... 54",
        "Item 9A. Controls and Procedures ... 98",
    ]
)

DOC = "\n\n".join(
    [
        "ANNUAL REPORT PURSUANT TO SECTION 13 OF THE SECURITIES EXCHANGE ACT OF 1934",
        TOC,
        "PART I",
        "Item 1. Business",
        _filler("business", 900),
        # TRAP 2: cross-reference mid-sentence, must NOT be treated as a heading
        "Our operations are subject to the risks described in Item 1A. Risk Factors below.",
        _filler("business", 400),
        "Item 1A. Risk Factors",
        _filler("risk", 1600),
        "Item 1B. Unresolved Staff Comments",
        "None.",
        "Item 2. Properties",
        _filler("property", 300),
        "PART II",
        # TRAP 3: different punctuation and casing
        "ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
        _filler("discussion", 1600),
        "Item 7A: Quantitative and Qualitative Disclosures About Market Risk",
        _filler("marketrisk", 200),
        "Item 8. Financial Statements and Supplementary Data",
        _filler("financials", 1000),
        "[TABLE]",
        "Net sales | $ | 391,035 | $ | 383,285",
        "[/TABLE]",
        _filler("financials", 500),
        "Item 9A. Controls and Procedures",
        _filler("controls", 400),
    ]
)


@pytest.fixture
def result():
    return sectionise(DOC)


# --- TOC handling ----------------------------------------------------------

def test_toc_is_detected():
    span = detect_toc_span(find_candidates(DOC))
    assert span is not None, "failed to detect the table of contents"
    start, end = span
    assert start < DOC.index("PART I"), "TOC span should sit before the body"


def test_risk_factors_is_the_body_section_not_the_toc_row(result):
    assert "1A" in result.sections, "Item 1A was not recovered"
    sec = result.sections["1A"]
    assert sec.n_chars > 5000, (
        f"Item 1A is only {sec.n_chars} chars -- this is the TOC row, "
        "not the real section"
    )
    assert "risk risk" in sec.text


def test_item_1_is_the_body_section(result):
    assert "1" in result.sections
    assert result.sections["1"].n_chars > 3000


# --- cross-reference handling ---------------------------------------------

def test_cross_reference_does_not_split_item_1(result):
    """The phrase 'in Item 1A. Risk Factors below' sits mid-sentence inside
    Item 1. If it were treated as a heading, Item 1 would be truncated and
    Item 1A would start in the wrong place."""
    sec1 = result.sections["1"]
    assert "risks described in Item 1A" in sec1.text, (
        "Item 1 was truncated at a cross-reference"
    )


# --- punctuation and casing variants --------------------------------------

def test_uppercase_dash_variant_matches(result):
    assert "7" in result.sections, "'ITEM 7 - MANAGEMENT'S...' was not matched"
    assert result.sections["7"].n_chars > 5000


def test_colon_variant_matches(result):
    assert "7A" in result.sections, "'Item 7A:' was not matched"


def test_item_1_does_not_swallow_item_1a():
    """The negative lookahead must stop 'Item 1' matching the '1' in 'Item 1A'."""
    cands = find_candidates("Item 1A. Risk Factors", items=["1"])
    assert cands == [], "'Item 1' pattern incorrectly matched 'Item 1A'"


# --- boundaries and ordering ----------------------------------------------

def test_sections_are_ordered_and_non_overlapping(result):
    secs = sorted(result.sections.values(), key=lambda s: s.start)
    for a, b in zip(secs, secs[1:]):
        assert a.end <= b.start, f"Item {a.item} overlaps Item {b.item}"


def test_item_8_contains_the_financial_table(result):
    assert "8" in result.sections
    assert "391,035" in result.sections["8"].text, "financial table lost"


def test_target_coverage_is_complete(result):
    assert result.coverage() == 100.0, f"missing sections: {result.warnings}"


def test_short_sections_are_rejected_not_kept(result):
    """Item 1B is 'None.' -- correctly below threshold. It should be reported
    as rejected rather than silently stored as a valid section."""
    assert "1B" not in result.sections
    assert any("1B" in r for r in result.rejected)


# --- HTML extraction -------------------------------------------------------

def test_html_block_tags_become_newlines():
    doc = html_to_text("<div>Item 7. MD&amp;A</div><p>Some text</p>")
    assert "Item 7. MD&A" in doc.text
    assert "\n" in doc.text


def test_html_tables_are_pipe_delimited():
    pytest.importorskip("lxml")
    doc = html_to_text(
        "<table><tr><td>Net sales</td><td>$</td><td>391,035</td></tr></table>"
    )
    assert "[TABLE]" in doc.text
    assert "Net sales | 391,035" in doc.text, "table row was shredded"
    assert doc.n_tables == 1


def test_nbsp_is_normalised():
    assert normalise_whitespace("Item\u00a07.\u00a0MD&A") == "Item 7. MD&A"


# --- title fallback (banks that don't use Item numbers) --------------------

BANK_DOC = "\n\n".join(
    [
        "FORM 10-K INDEX",
        "\n".join([
            "Item 1. Business ... refer to pages 1-10",
            "Item 1A. Risk Factors ... refer to pages 11-40",
            "Item 2. Properties ... 41",
            "Item 5. Market for Common Equity ... 42",
            "Item 7. MD&A ... refer to pages 50-120",
            "Item 8. Financial Statements ... refer to pages 121-300",
            "Item 9A. Controls and Procedures ... 301",
            "Item 12. Security Ownership ... 305",
            "Item 15. Exhibits ... 310",
        ]),
        "PART I",
        "Item 1. Business",
        _filler("banking", 900),
        "Item 1A. Risk Factors",
        _filler("bankrisk", 1600),
        "Item 2. Properties",
        _filler("premises", 300),
        # No "Item 7" label anywhere -- just the bare title, as JPM does it.
        "Management's discussion and analysis",
        _filler("mdna", 1800),
        "Quantitative and qualitative disclosures about market risk",
        _filler("varrisk", 300),
        "Report of Independent Registered Public Accounting Firm",
        _filler("audited", 1200),
        "Item 9A. Controls and Procedures",
        _filler("controls", 400),
    ]
)


@pytest.fixture
def bank():
    return sectionise(BANK_DOC)


def test_title_fallback_recovers_mdna(bank):
    """JPM-style filings label MD&A by title only. Without the fallback,
    Item 2 absorbs the rest of the document."""
    assert "7" in bank.sections, "MD&A not recovered by title fallback"
    assert bank.sections["7"].source == "title-fallback"
    assert "mdna mdna" in bank.sections["7"].text


def test_title_fallback_recovers_item_8(bank):
    assert "8" in bank.sections
    assert "audited" in bank.sections["8"].text


def test_numbered_sections_still_prefer_item_headings(bank):
    """Item 1A IS labelled by number here, so it must not be downgraded
    to the fallback path."""
    assert bank.sections["1A"].source == "item-heading"


def test_fallback_use_is_flagged_in_warnings(bank):
    assert any("title" in w for w in bank.warnings), (
        "title-fallback sections must be flagged for human verification"
    )


def test_earlier_section_no_longer_swallows_document(bank):
    """Item 2 previously ran to end-of-document because nothing after it
    was matched. It should now stop at the MD&A heading."""
    assert bank.sections["2"].n_words < 1000


# --- wrapper filings (JPM-style) -------------------------------------------
# Modelled directly on the JPM FY2024 diagnostic output: numbered items are
# cross-reference stubs, the real content is an annual report appended after
# Item 15, and that report has its own table of contents which matches the
# same title patterns as the real headings.

WRAPPER_DOC = "\n\n".join(
    [
        "FORM 10-K INDEX",
        "\n".join([
            "Item 1. | Business. | 1",
            "Item 1A. | Risk Factors. | 10-37",
            "Item 2. | Properties. | 38",
            "Item 5. | Market for Common Equity. | 39",
            "Item 7. | ManagementÆs Discussion and Analysis. | 39",
            "Item 7A. | Quantitative and Qualitative Disclosures About Market Risk. | 39",
            "Item 8. | Financial Statements and Supplementary Data. | 40",
            "Item 9A. | Controls and Procedures. | 40",
            "Item 15. | Exhibits. | 44-47",
        ]),
        "Item 1. Business. Overview " + _filler("banking", 900),
        "Item 1A. Risk Factors. " + _filler("bankrisk", 1600),
        "Item 2. Properties. " + _filler("premises", 200),
        "Item 5. Market for RegistrantÆs Common Equity. " + _filler("equity", 200),
        # STUBS: found by number, but only a cross-reference
        "Item 7. ManagementÆs Discussion and Analysis of Financial Condition and "
        "Results of Operations. ManagementÆs discussion and analysis of financial "
        "condition and results of operations appears on pages 52-167.",
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk. "
        "Refer to the Market Risk Management section on pages 141-149.",
        "Item 8. Financial Statements and Supplementary Data. The Consolidated "
        "Financial Statements appear on pages 172-321.",
        "Item 9A. Controls and Procedures. " + _filler("controls", 200),
        "Item 15. Exhibits, Financial Statement Schedules.",
        _filler("exhibitlist", 400),
        # TRAP: the annual report's own contents index, same titles, pipe-delimited
        "ManagementÆs discussion and analysis: | 172 | Consolidated Financial "
        "Statements 52 | Introduction | 177 | Notes | 180 | Report of Independent "
        "Registered Public Accounting Firm | 169",
        # REAL content starts here
        "ManagementÆs discussion and analysis The following is ManagementÆs "
        "discussion and analysis of the financial condition and results of "
        "operations of the Firm. " + _filler("mdna", 2000),
        "Report of Independent Registered Public Accounting Firm To the Board of "
        "Directors and Shareholders: Opinions on the Financial Statements. "
        + _filler("audited", 1500),
    ]
)


@pytest.fixture
def wrapper():
    return sectionise(WRAPPER_DOC)


def test_wrapper_recovers_real_mdna_not_the_stub(wrapper):
    assert "7" in wrapper.sections, "MD&A not recovered from the appended report"
    sec = wrapper.sections["7"]
    assert sec.source == "wrapper-recovery"
    assert "mdna mdna" in sec.text, "recovered the stub or the index row, not content"
    assert "appears on pages 52-167" not in sec.text


def test_wrapper_recovers_real_item_8(wrapper):
    assert "8" in wrapper.sections
    assert "audited audited" in wrapper.sections["8"].text


def test_wrapper_skips_the_second_table_of_contents(wrapper):
    """The appended report's index matches the same title patterns. Prose
    detection must reject it -- pipes and page numbers give it away."""
    assert "| 172 |" not in wrapper.sections["7"].text[:300]


def test_wrapper_item_15_no_longer_swallows_the_report(wrapper):
    """Item 15 previously absorbed 167k words because nothing after it matched."""
    if "15" in wrapper.sections:
        assert wrapper.sections["15"].n_words < 2000


def test_mojibake_apostrophe_still_matches():
    """Encoding mismatches render Management's as ManagementÆs. The pattern
    must tolerate that without enumerating every variant."""
    from filingiq.parsing.sectioniser import _wrapper_title_pattern
    pat = _wrapper_title_pattern("7")
    for variant in ["Management's discussion and analysis",
                    "ManagementÆs discussion and analysis",
                    "Management\u2019s discussion and analysis",
                    "Managementâ€™s discussion and analysis"]:
        assert pat.search(variant), f"failed on {variant!r}"


def test_unrecoverable_stub_is_reported_honestly(wrapper):
    """Item 7A has no standalone content -- it lives inside MD&A. The system
    must say so rather than silently returning the stub."""
    assert any("stub" in w for w in wrapper.warnings)


def test_prose_detector_separates_index_rows_from_content():
    from filingiq.parsing.sectioniser import _looks_like_prose
    assert _looks_like_prose(
        " The following is a discussion of the financial condition and results "
        "of operations of the Firm during the year."
    )
    assert not _looks_like_prose(": | 172 | Consolidated Financial Statements 52 | Introduction | 177 |")


# --- cross-reference stub detection ----------------------------------------

def test_xref_stub_detected():
    from filingiq.parsing.sectioniser import _is_cross_reference_stub
    # JPM's actual Item 7A body
    assert _is_cross_reference_stub(
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk.\n"
        "Refer to the Market Risk Management section of Management's discussion "
        "and analysis on pages 141-149 for a discussion of quantitative and "
        "qualitative disclosures about market risk."
    )


def test_short_real_section_is_not_a_stub():
    """Item 7A is legitimately brief for many filers. Brevity alone must not
    condemn it -- only referring language plus a page pointer."""
    from filingiq.parsing.sectioniser import _is_cross_reference_stub
    assert not _is_cross_reference_stub(
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk.\n"
        "The Company is exposed to market risk from changes in interest rates "
        "and foreign currency exchange rates. A hypothetical 10% strengthening "
        "of the U.S. dollar would not have had a material impact on the "
        "Company's results of operations for the periods presented. The Company "
        "does not hold derivative instruments for speculative purposes and "
        "manages counterparty exposure through master netting arrangements."
    )


def test_long_section_with_refer_to_is_not_a_stub():
    """Real MD&A is full of 'Refer to Note 30'. Only SHORT bodies qualify."""
    from filingiq.parsing.sectioniser import _is_cross_reference_stub
    body = "Refer to Note 30 for details on pages 200. " + _filler("analysis", 600)
    assert not _is_cross_reference_stub(body)


def test_thresholds_scale_with_document_size():
    from filingiq.parsing.sectioniser import _scaled_thresholds, MIN_SECTION_CHARS
    small = _scaled_thresholds(220_000, MIN_SECTION_CHARS)
    large = _scaled_thresholds(1_470_000, MIN_SECTION_CHARS)
    assert large["1A"] > small["1A"], "thresholds must scale with document size"


def test_threshold_scaling_is_clamped():
    from filingiq.parsing.sectioniser import _scaled_thresholds, MIN_SECTION_CHARS
    tiny = _scaled_thresholds(1_000, MIN_SECTION_CHARS)
    huge = _scaled_thresholds(50_000_000, MIN_SECTION_CHARS)
    assert tiny["1A"] == int(MIN_SECTION_CHARS["1A"] * 0.5)
    assert huge["1A"] == int(MIN_SECTION_CHARS["1A"] * 3.0)


def test_short_but_real_sections_survive_xref_check():
    """JPM's Item 9A is 328 words of real content that cites a page number.
    An earlier 400-word ceiling wrongly condemned it. Regression guard."""
    from filingiq.parsing.sectioniser import _is_cross_reference_stub
    body = ("Item 9A. Controls and Procedures.\n"
            + "The internal control framework provides reasonable assurance "
              "regarding the reliability of financial reporting. " * 20
            + " Refer to page 168 for further detail.")
    assert len(body.split()) > 150
    assert not _is_cross_reference_stub(body)
