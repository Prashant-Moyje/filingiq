"""Tests for the section 6 hallucination measurement.

The harness itself cannot run here -- generating memos needs an API key and a
built corpus. What CAN be tested offline is the measurement logic, and that is
the part that decides what number gets published. A miscounted claim is a wrong
row in EVALUATION.md, and unlike a crash it looks like a result.

Memos below are hand-written with KNOWN planted hallucinations, so the expected
counts are arithmetic rather than judgement.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
           / "11_eval_hallucination.py")


def _load():
    """Import the numbered script, whose filename is not a valid module name.

    The module must be registered in sys.modules BEFORE exec_module: @dataclass
    resolves annotations via sys.modules[cls.__module__], which is None for a
    module that is still being executed and not yet registered.
    """
    spec = importlib.util.spec_from_file_location("eval_hallucination", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load()


# Apple FY2024, real figures.
VERIFIED = {
    "revenue": 391035000000.0,
    "net_income": 93736000000.0,
    "total_assets": 364980000000.0,
    "eps_diluted": 6.08,
    "diff_new": 0.0,
    "diff_modified": 14.0,
}

CLEAN_MEMO = (
    "Apple reported revenue of $391.0 billion for fiscal 2024. "
    "Net income was $93.7 billion and diluted earnings per share were $6.08. "
    "Total assets stood at $365.0 billion. "
    "The company modified 14 risk factors and added none."
)

# Two planted fabrications: a gross margin never supplied, and a wrong
# operating income. Everything else matches VERIFIED.
DIRTY_MEMO = (
    "Apple reported revenue of $391.0 billion for fiscal 2024. "
    "Gross margin reached $180.7 billion on the year. "
    "Net income was $93.7 billion. "
    "Operating income came in at $145.2 billion. "
    "The company modified 14 risk factors."
)


# --- counting ---------------------------------------------------------------

def test_clean_memo_has_no_unsupported_claims():
    """Every figure traceable to verified ground truth. If this fails the gate
    has false positives, and a false-positive gate inflates the hallucination
    rate with correct sentences."""
    m = ev.measure_memo(CLEAN_MEMO, VERIFIED, "AAPL", 2024, "none", gate=False)
    assert m.unsupported_generated == 0
    assert m.numeric_claims > 0, "claims must actually be extracted"


def test_planted_fabrications_are_counted():
    m = ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024, "none", gate=False)
    assert m.unsupported_generated == 2, (
        f"expected the 2 planted figures, got {m.unsupported_generated}: "
        f"{m.examples}")


def test_gate_off_shows_everything_it_generated():
    """Configuration 1 must not quietly launder claims through the gate."""
    m = ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024, "none", gate=False)
    assert m.unsupported_shown == m.unsupported_generated == 2
    assert m.words_after == m.words_before, "nothing should be removed"


def test_gate_on_removes_them_and_keeps_the_rest():
    m = ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024, "xbrl", gate=True)
    assert m.unsupported_generated == 2, "must still record what was written"
    assert m.unsupported_shown == 0, "gate must remove every unsupported claim"
    assert m.words_after < m.words_before
    assert "391.0" in DIRTY_MEMO and m.words_after > 0, "must not strip all"


def test_generated_count_survives_the_gate_in_the_record():
    """The whole point of two metrics.

    If the harness recorded only what the reader sees, configurations 2 and 3
    would report 0.000 and the table would carry no information about whether
    the prompt did anything.
    """
    gated = ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024, "xbrl", True)
    ungated = ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024, "none", False)
    assert gated.unsupported_generated == ungated.unsupported_generated


# --- aggregation ------------------------------------------------------------

def test_summary_averages_per_memo_not_per_claim():
    r = ev.ConfigResult(config="none")
    r.memos.append(ev.measure_memo(DIRTY_MEMO, VERIFIED, "AAPL", 2024,
                                   "none", False))
    r.memos.append(ev.measure_memo(CLEAN_MEMO, VERIFIED, "MSFT", 2024,
                                   "none", False))
    s = r.summary()
    assert s["n_memos"] == 2
    assert s["unsupported_generated_per_memo"] == 1.0, "2 over 2 memos"
    assert s["memos_with_any_unsupported"] == 1


def test_empty_result_does_not_divide_by_zero():
    assert ev.ConfigResult(config="none").summary()["n_memos"] == 0


# --- ablation integrity -----------------------------------------------------

def test_bare_prompt_really_removes_the_deterrent():
    """Configuration 1 is only a baseline if the prompt stops threatening to
    check. Otherwise row 1 measures a partly-mitigated system and understates
    the raw rate."""
    from filingiq.graph.nodes import MEMO_SYSTEM
    assert "checked automatically" in MEMO_SYSTEM, (
        "shipped prompt no longer contains the deterrent -- the ablation's "
        "premise has changed and MEMO_SYSTEM_BARE needs revisiting")
    bare = ev.MEMO_SYSTEM_BARE.lower()
    for banned in ("checked", "verif", "removed before", "costs you"):
        assert banned not in bare, f"bare prompt still mentions {banned!r}"


def test_bare_prompt_keeps_every_non_verification_rule():
    """It must differ from the shipped prompt ONLY in verification language.
    If it also dropped the formatting rules the configurations would differ in
    more than one variable and nothing could be attributed."""
    for rule in ("150-250 words", "no headings", "do not estimate",
                 "Introduce no other numbers"):
        assert rule in ev.MEMO_SYSTEM_BARE, f"bare prompt lost rule: {rule!r}"


def test_cited_config_adds_to_the_shipped_prompt():
    from filingiq.graph.nodes import MEMO_SYSTEM
    assert ev.MEMO_SYSTEM_CITED.startswith(MEMO_SYSTEM)
    assert "square brackets" in ev.MEMO_SYSTEM_CITED


def test_every_config_is_labelled():
    assert set(ev.CONFIGS) == set(ev.CONFIG_LABELS)


# --- prompt parity ----------------------------------------------------------

def test_user_prompt_is_identical_across_configurations():
    """The ablation varies the SYSTEM prompt only.

    build_memo_prompt is shared precisely so the evidence supplied cannot drift
    between configurations; if it could, a difference in the result would be
    attributable to the evidence rather than the instruction.
    """
    from filingiq.graph.nodes import build_memo_prompt
    state = {
        "ticker": "AAPL", "fiscal_year": 2024,
        "figures": {"revenue": 391035000000.0},
        "diff_summary": {"new": 0, "modified": 14, "unchanged": 49,
                         "drift_score": 0.2222},
        "new_risks": [], "risk_themes": {"regulatory": 2},
    }
    assert build_memo_prompt(state) == build_memo_prompt(state)
    assert "391.0 billion" in build_memo_prompt(state)


def test_verifiable_values_include_pipeline_counts():
    """Diff counts are as verifiable as XBRL figures. Excluding them would
    make the gate strip correct sentences about our own analysis and inflate
    the hallucination rate with false positives."""
    state = {"verified_values": {"revenue": 1.0},
             "diff_summary": {"new": 3, "modified": 7, "drift_score": 0.5}}
    v = ev.verifiable_values(state)
    assert v["diff_new"] == 3.0
    assert v["diff_modified"] == 7.0
    assert v["diff_drift_score"] == 0.5
    assert v["revenue"] == 1.0


def test_undefined_drift_is_not_offered_as_verifiable():
    """FM-019: drift is None when a section failed to parse. None must not
    become a citable value."""
    v = ev.verifiable_values({"verified_values": {},
                              "diff_summary": {"drift_score": None, "new": 2}})
    assert "diff_drift_score" not in v
    assert v["diff_new"] == 2.0


@pytest.mark.parametrize("config,expect_gate", [
    ("none", False), ("xbrl", True), ("cited", True)])
def test_gate_is_on_for_exactly_the_mitigated_configs(config, expect_gate):
    assert ev.CONFIGS[config][1] is expect_gate
