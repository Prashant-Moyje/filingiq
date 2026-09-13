"""Tests for the walk-forward harness.

The critical property is not accuracy -- it is CALIBRATION. A harness that
reports signal in pure noise makes every downstream number worthless, and a
harness that cannot detect a planted signal would hide a real finding. Both
directions are tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from filingiq.models.train import (  # noqa: E402
    FEATURE_SETS, permutation_test, walk_forward,
)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
THEMES = ["supply_chain", "cybersecurity", "regulatory", "litigation", "macro",
          "talent", "climate", "concentration", "liquidity", "ai_technology"]
FUNDS = ["net_margin", "operating_margin", "leverage", "roe", "asset_turnover",
         "cash_ratio", "rnd_intensity", "reporting_lag_days"]


def make_df(n_per_year=26, seed=0, signal=0.0, nan_fraction=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in YEARS:
        for i in range(n_per_year):
            r = {"ticker": f"T{i}", "fiscal_year": y,
                 "drift_score": rng.random(),
                 "n_new": rng.integers(0, 10), "n_modified": rng.integers(0, 40),
                 "n_removed": rng.integers(0, 30),
                 "n_total_risks": rng.integers(20, 90),
                 "removal_rate": rng.random(), "new_rate": rng.random(),
                 "modified_rate": rng.random()}
            r.update({f"theme_{t}": rng.random() for t in THEMES})
            r.update({f: rng.normal() for f in FUNDS})
            rows.append(r)
    df = pd.DataFrame(rows)
    df["excess_return"] = (signal * df["drift_score"]
                           + rng.normal(0, 0.15, len(df)))
    if nan_fraction:
        mask = rng.random(len(df)) < nan_fraction
        df.loc[mask, "operating_margin"] = np.nan
    return df


# --- calibration -----------------------------------------------------------

def test_no_signal_found_in_pure_noise():
    """The single most important property. A harness that finds signal in
    noise makes every number it produces worthless."""
    p = permutation_test(make_df(seed=1), "combined", n_permutations=60)
    assert p["p_value"] > 0.05, f"found signal in noise (p={p['p_value']})"


def test_planted_signal_is_detected():
    p = permutation_test(make_df(seed=2, signal=0.5), "combined",
                         n_permutations=60)
    assert p["p_value"] < 0.05, "failed to detect a planted signal"


def test_null_distribution_is_wide_at_this_sample_size():
    """A mean IC of 0.10+ arises by chance on ~26-row test folds. Reporting a
    point estimate without this context would be misleading."""
    p = permutation_test(make_df(seed=3), "combined", n_permutations=80)
    assert p["null_p95_abs_ic"] > 0.05, (
        "null distribution implausibly tight -- check the fold structure")


# --- walk-forward mechanics ------------------------------------------------

def test_training_never_includes_the_test_year_or_later():
    r = walk_forward(make_df(), "combined")
    for f in r.folds:
        assert max(f.train_years) < f.test_year, "trained on the future"


def test_folds_expand_forwards():
    r = walk_forward(make_df(), "combined")
    sizes = [f.n_train for f in r.folds]
    assert sizes == sorted(sizes), "training window should grow, not shuffle"


def test_first_year_is_never_a_test_fold():
    r = walk_forward(make_df(), "combined")
    assert min(f.test_year for f in r.folds) > min(YEARS)


def test_missing_values_do_not_break_the_fit():
    """Banks lack operating_income; retailers lack R&D. Imputation happens
    inside the fold, so this must run without leaking or crashing."""
    r = walk_forward(make_df(nan_fraction=0.4), "combined")
    assert r.folds and all(np.isfinite(f.rmse) for f in r.folds)


def test_all_feature_sets_are_runnable():
    for name in FEATURE_SETS:
        assert walk_forward(make_df(), name).folds, f"{name} produced no folds"


def test_baseline_rmse_uses_the_training_mean():
    """The comparison must be against a constant predictor, not zero."""
    r = walk_forward(make_df(), "fundamentals")
    assert all(f.baseline_rmse > 0 for f in r.folds)


def test_summary_is_empty_without_folds():
    tiny = make_df(n_per_year=2)
    assert walk_forward(tiny, "combined", min_train=1000).summary() == {}


# --- regression guards -----------------------------------------------------
#
# Each of these covers a defect that produced a PLAUSIBLE wrong number rather
# than a crash -- the class that reaches a report and does not reproduce.

def test_permutation_p_value_can_never_be_zero():
    """A finite permutation test cannot support p = 0.0.

    Without the (n+1) correction the estimator returns exactly 0.0 whenever no
    shuffle beats the observed statistic, which on a strongly planted signal is
    every shuffle. p=0.0 is an infinitely strong claim from 60 samples.
    """
    p = permutation_test(make_df(seed=7, signal=5.0), "combined",
                         n_permutations=60)
    assert p["p_value"] > 0.0, "reported p = 0.0 from a finite permutation test"
    assert p["p_value"] >= 1 / (p["n_permutations"] + 1) - 1e-12
    assert p["p_value"] < 0.05, "correction must not destroy real detection"


def test_permutation_null_uses_a_constant_sample_size():
    """Rows unusable in the observed run must not re-enter via the shuffle.

    Permuting a target column that still contains NaN relocates those NaNs on
    every draw, so each permutation is scored on a different subset and the
    null describes varying sample sizes rather than varying order.
    """
    df = make_df(seed=11)
    df.loc[df.index[:12], "excess_return"] = np.nan

    p = permutation_test(df, "combined", n_permutations=40)
    assert not np.isnan(p["p_value"])
    # With a constant sample the null spread stays in a sane range; a null
    # built on wobbling n inflates it.
    assert p["null_std_ic"] < 0.5


def test_coefficients_are_kept_per_fold_not_overwritten():
    """A single flat dict silently reported only the last fold's coefficients.

    Coefficients fitted on the first fold's training rows and on the last
    fold's are different objects, and the spread between them is the honest
    signal about stability at this sample size.
    """
    r = walk_forward(make_df(), "combined", model="ridge")
    assert len(r.coefficients) == len(r.folds), (
        "expected one coefficient set per fold")
    assert set(r.coefficients) == {f.test_year for f in r.folds}
    for coefs in r.coefficients.values():
        assert isinstance(coefs, dict) and coefs
    assert r.last_fold_coefficients == r.coefficients[max(r.coefficients)]
