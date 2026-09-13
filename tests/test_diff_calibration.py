"""Tests for the diff-threshold calibration set and harness.

The embedding run itself needs sentence-transformers and a model download, so
it is not part of CI. What is tested here is everything that decides whether
the calibration MEANS anything: label integrity, that the harness reproduces
the production alignment, and that the reported thresholds match diff.py.

A calibration whose labels are wrong produces a confident, precise, wrong
recommendation -- the same failure shape as FM-019, one layer up.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.analysis import calibration_set as cs  # noqa: E402
from filingiq.analysis.diff import (  # noqa: E402
    MODIFIED_THRESHOLD, UNCHANGED_THRESHOLD,
)

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
           / "16_calibrate_diff.py")


def _load():
    spec = importlib.util.spec_from_file_location("calibrate_diff", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cal = _load()


# --- label integrity ------------------------------------------------------

def test_every_pair_carries_one_of_the_three_statuses():
    """The labels must be exactly the statuses diff_sections can emit.
    A fourth label would silently never be predicted and drag accuracy down
    for a reason unrelated to the thresholds."""
    assert {p.label for p in cs.build_pairs()} <= set(cal.LABELS)


def test_unchanged_pairs_are_genuinely_the_same_risk():
    """An 'unchanged' label must mean carried forward, not merely similar.
    Either verbatim, or the prior text still substantially present."""
    for p in cs.build_pairs():
        if p.label != "unchanged":
            continue
        assert p.prior, f"{p.topic}: unchanged pair needs a prior"
        if p.transformation == "verbatim":
            assert p.current == p.prior, f"{p.topic} claims verbatim but differs"
        else:
            # A refresh edits a little; it must not rewrite the factor.
            shared = len(set(p.prior.split()) & set(p.current.split()))
            assert shared / len(set(p.prior.split())) > 0.85, (
                f"{p.topic}: labelled a refresh but shares only "
                f"{shared} words with the prior")


def test_modified_pairs_are_actually_rewritten():
    """A 'modified' pair that is nearly verbatim would be mislabelled, and
    would pull the modified/unchanged boundary in the wrong direction."""
    for p in cs.build_pairs():
        if p.label != "modified":
            continue
        assert p.current != p.prior
        shared = len(set(p.prior.split()) & set(p.current.split()))
        assert shared / len(set(p.prior.split())) < 0.7, (
            f"{p.topic}: labelled a rewrite but barely changed")


def test_new_factors_have_no_prior_counterpart():
    for p in cs.build_pairs():
        if p.label == "new":
            assert p.prior == "", f"{p.topic}: a new risk cannot have a prior"
            assert p.topic not in cs.PRIOR_FACTORS


def test_new_topics_are_absent_from_the_prior_section():
    """If a 'new' topic were also in the prior year it would legitimately match
    and the label would be wrong."""
    prior_topics = {t for t, _ in cs.prior_year_section()}
    for topic, _, label in cs.current_year_section():
        if label == "new":
            assert topic not in prior_topics


def test_all_three_classes_are_represented():
    labels = [lbl for _, _, lbl in cs.current_year_section()]
    for expected in cal.LABELS:
        assert labels.count(expected) >= 3, (
            f"only {labels.count(expected)} {expected} examples -- too few to "
            f"place a threshold")


def test_prior_section_contains_every_modified_and_unchanged_topic():
    """Otherwise the best match would be some unrelated factor and the score
    would not measure what the label claims."""
    prior = {t for t, _ in cs.prior_year_section()}
    for topic, _, label in cs.current_year_section():
        if label in ("unchanged", "modified"):
            assert topic in prior, f"{topic} has no prior-year counterpart"


# --- harness logic --------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (1.00, "unchanged"), (0.96, "unchanged"), (0.95, "unchanged"),
    (0.94, "modified"), (0.80, "modified"),
    (0.79, "new"), (0.10, "new"),
])
def test_classify_matches_diff_sections_banding(score, expected):
    """The harness must band exactly as diff_sections does, thresholds
    inclusive at the lower edge, or it would be calibrating something the
    pipeline does not do."""
    assert cal.classify(score, UNCHANGED_THRESHOLD, MODIFIED_THRESHOLD) == expected


def test_accuracy_is_computed_over_all_rows():
    rows = [{"true": "unchanged", "score": 0.99},
            {"true": "modified", "score": 0.85},
            {"true": "new", "score": 0.10},
            {"true": "new", "score": 0.99}]      # deliberately wrong
    assert cal.accuracy(rows, 0.95, 0.80) == 0.75


def test_per_class_reports_precision_recall_and_support():
    rows = [{"true": "unchanged", "score": 0.99},
            {"true": "unchanged", "score": 0.50},   # missed
            {"true": "new", "score": 0.10}]
    m = cal.per_class(rows, 0.95, 0.80)
    assert m["unchanged"]["recall"] == 0.5
    assert m["unchanged"]["precision"] == 1.0
    assert m["unchanged"]["support"] == 2
    assert m["new"]["recall"] == 1.0


def test_per_class_handles_a_class_that_is_never_predicted():
    """precision is undefined, not zero, when nothing was predicted -- zero
    would read as 'the model got them all wrong'."""
    rows = [{"true": "unchanged", "score": 0.99}]
    assert cal.per_class(rows, 0.95, 0.80)["new"]["precision"] is None
