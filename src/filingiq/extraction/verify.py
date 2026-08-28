"""Verification against XBRL ground truth.

This module is the reason the whole project is credible. Every extracted figure
is compared arithmetically against the value the SEC's own XBRL data says that
filing reported. There is no LLM in this loop and no judgment call: the answer
is either within tolerance of the authoritative figure or it is not.

TOLERANCE BANDS, NOT A SINGLE THRESHOLD
---------------------------------------
Exact match is too strict -- filings round, and 391,035 million is not
bit-identical to 391,035,000,000 after floating point. Too loose and genuine
errors slip through. Reporting bands (exact / 0.5% / 2%) shows the shape of the
error distribution rather than collapsing it to one number.

ERROR TAXONOMY
--------------
A wrong answer is not just wrong; HOW it is wrong tells you what to fix:

  scale_error       off by exactly 1e3 / 1e6 / 1e9 -- the model read the right
                    number and missed the table header. A prompt problem.
  sign_error        right magnitude, wrong sign -- usually a loss or a
                    parenthesised negative misread.
  wrong_period      matches the prior-year comparative column instead of the
                    current year. A retrieval/context problem.
  wrong_metric      matches a DIFFERENT metric's true value -- the model found
                    the wrong line item.
  unexplained       none of the above.

Without this taxonomy "76% accuracy" is a dead end. With it, you know whether
to fix the prompt, the retrieval, or the chunking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

log = logging.getLogger(__name__)

TOLERANCE_BANDS = [("exact", 0.0), ("within_0.5pct", 0.005), ("within_2pct", 0.02)]
SCALE_RATIOS = {1e3: "thousands", 1e6: "millions", 1e9: "billions",
                1e-3: "thousands", 1e-6: "millions", 1e-9: "billions"}


@dataclass
class Verdict:
    metric: str
    extracted: float | None
    truth: float | None
    abs_error: float | None
    rel_error: float | None
    band: str            # 'exact' | 'within_0.5pct' | 'within_2pct' | 'wrong'
    error_type: str | None
    verified: bool
    abstained: bool

    @property
    def correct(self) -> bool:
        return self.band != "wrong" and self.verified


def _rel(extracted: float, truth: float) -> float:
    if truth == 0:
        return 0.0 if extracted == 0 else float("inf")
    return abs(extracted - truth) / abs(truth)


def _classify_error(extracted: float, truth: float,
                    all_truth: dict[str, float], metric: str) -> str:
    if truth != 0:
        ratio = extracted / truth
        for factor, name in SCALE_RATIOS.items():
            if abs(ratio - factor) / factor < 0.01:
                return f"scale_error_{name}"
    if truth != 0 and _rel(-extracted, truth) < 0.02:
        return "sign_error"
    # Did it match some OTHER metric's true value?
    for other, val in all_truth.items():
        if other == metric or val is None or val == 0:
            continue
        if _rel(extracted, val) < 0.005:
            return f"wrong_metric_matched_{other}"
    return "unexplained"


def verify_figure(metric: str, extracted: float | None, truth: float | None,
                  all_truth: dict[str, float] | None = None,
                  prior_year_truth: float | None = None,
                  errored: bool = False) -> Verdict:
    """errored=True means the CALL failed, not that the model declined.

    These look identical downstream -- both produce no value -- but they mean
    opposite things. A declined answer is the system working; a failed call is
    the system broken. Conflating them let a rate limit masquerade as a
    cautious model and corrupt an entire results file. See FM-016.
    """
    all_truth = all_truth or {}

    if errored:
        return Verdict(metric, None, truth, None, None, "errored",
                       "call_failed", verified=False, abstained=False)

    if extracted is None:
        return Verdict(metric, None, truth, None, None, "abstained",
                       None, verified=False, abstained=True)
    if truth is None:
        # No ground truth for this metric -- cannot score it either way.
        return Verdict(metric, extracted, None, None, None, "no_ground_truth",
                       None, verified=False, abstained=False)

    abs_err = abs(extracted - truth)
    rel_err = _rel(extracted, truth)

    # WRONG-PERIOD CHECK RUNS FIRST, BEFORE TOLERANCE BANDING.
    #
    # Apple's FY2023 revenue ($383.3B) is 2% from FY2024 ($391.0B). A model that
    # reads the prior-year comparative column therefore lands INSIDE the 2%
    # tolerance band and would be scored correct. Slow-growing companies make
    # this worse, not better -- the closer the two years, the more certainly a
    # wrong-period read is absolved.
    #
    # So: if the value matches the prior year more closely than the current
    # year, it is a wrong-period error regardless of tolerance. Reading the
    # wrong column is wrong even when the columns happen to be similar.
    if prior_year_truth is not None:
        prior_rel = _rel(extracted, prior_year_truth)
        if prior_rel < rel_err and prior_rel < 0.005:
            return Verdict(metric, extracted, truth, abs_err, rel_err, "wrong",
                           "wrong_period_prior_year", verified=False,
                           abstained=False)

    band = "wrong"
    for name, tol in TOLERANCE_BANDS:
        # The absolute-epsilon shortcut exists because balance-sheet figures
        # are reported to the dollar and float arithmetic on 1e11 values loses
        # cents. It MUST be gated on magnitude: applied unconditionally, a
        # diluted EPS of 5.13 would score as an exact match against a true
        # 6.13, since |5.13 - 6.13| < 1.0. Per-share figures are the one place
        # a difference of 1.0 is enormous.
        exact_by_epsilon = (tol == 0.0 and abs_err < 1.0 and abs(truth) >= 1e6)
        if rel_err <= tol or exact_by_epsilon:
            band = name
            break

    error_type = None
    if band == "wrong":
        error_type = _classify_error(extracted, truth, all_truth, metric)

    return Verdict(metric, extracted, truth, abs_err, rel_err, band,
                   error_type, verified=(band != "wrong"), abstained=False)


def summarise(verdicts: Iterable[Verdict]) -> dict:
    """Accuracy, abstention and the error breakdown, in one place."""
    v = list(verdicts)
    scorable = [x for x in v
                if x.band not in ("abstained", "no_ground_truth", "errored")]
    abstained = [x for x in v if x.abstained]
    errored = [x for x in v if x.band == "errored"]

    out: dict = {
        "n_total": len(v),
        "n_scorable": len(scorable),
        "n_abstained": len(abstained),
        "n_errored": len(errored),
        "abstention_rate": round(len(abstained) / max(len(v), 1), 3),
        "error_rate": round(len(errored) / max(len(v), 1), 3),
    }
    # Bands are cumulative: 'within_2pct' counts exact and 0.5% matches too.
    band_order = [n for n, _ in TOLERANCE_BANDS]
    for idx, name in enumerate(band_order):
        hits = sum(1 for x in scorable
                   if x.band in band_order and band_order.index(x.band) <= idx)
        out[name] = round(hits / max(len(scorable), 1), 3)

    errs: dict[str, int] = {}
    for x in scorable:
        if x.error_type:
            errs[x.error_type] = errs.get(x.error_type, 0) + 1
    out["error_types"] = dict(sorted(errs.items(), key=lambda kv: -kv[1]))

    rels = [x.rel_error for x in scorable
            if x.rel_error is not None and x.rel_error != float("inf")]
    out["mean_abs_rel_error"] = round(sum(rels) / len(rels), 5) if rels else None
    out["median_abs_rel_error"] = (round(sorted(rels)[len(rels) // 2], 5)
                                   if rels else None)
    return out


# ---------------------------------------------------------------------------
# INTERNAL CONSISTENCY
#
# XBRL verification is only possible because this project happens to have
# ground truth. In production you generally do not: you are extracting from a
# filing precisely because nobody has structured it yet.
#
# Accounting identities give you a verification signal that needs no external
# reference at all. Financial statements are internally constrained, and those
# constraints are exactly what a wrong-table error violates.
#
# The motivating case: on JPMorgan filings the model read a SEGMENT balance
# sheet instead of the consolidated one, returning total assets of $568B
# against a true $3.7T. Consistently, across four filings -- a reliable model
# reading the wrong table. Equity was extracted correctly every time, so
#
#     assets != liabilities + equity
#
# catches it immediately, using nothing but the model's own outputs.
#
# This generalises where XBRL verification does not, which makes it the more
# valuable check even though it catches less.
# ---------------------------------------------------------------------------

IDENTITY_TOLERANCE = 0.02   # 2%: statements round, and minority interests vary


@dataclass
class ConsistencyViolation:
    rule: str
    detail: str
    severity: str            # 'error' | 'warning'
    metrics_involved: list[str]


def check_accounting_identities(values: dict[str, float | None]) -> list[ConsistencyViolation]:
    """Check extracted figures against accounting identities.

    Only checks what is present -- a missing figure is not a violation, it is
    simply an unchecked constraint.
    """
    out: list[ConsistencyViolation] = []

    assets = values.get("total_assets")
    liabilities = values.get("total_liabilities")
    equity = values.get("stockholders_equity")

    # A = L + E
    if None not in (assets, liabilities, equity):
        expected = liabilities + equity
        if assets and abs(assets - expected) / abs(assets) > IDENTITY_TOLERANCE:
            out.append(ConsistencyViolation(
                rule="balance_sheet_identity",
                detail=(f"assets {assets:,.0f} != liabilities {liabilities:,.0f} "
                        f"+ equity {equity:,.0f} = {expected:,.0f} "
                        f"(off by {abs(assets - expected) / abs(assets):.1%})"),
                severity="error",
                metrics_involved=["total_assets", "total_liabilities",
                                  "stockholders_equity"],
            ))

    # Equity cannot exceed assets.
    if None not in (assets, equity) and equity > assets:
        out.append(ConsistencyViolation(
            rule="equity_exceeds_assets",
            detail=f"equity {equity:,.0f} > assets {assets:,.0f}",
            severity="error",
            metrics_involved=["stockholders_equity", "total_assets"]))

    # Net income should not exceed revenue by a wide margin. Legitimate for a
    # company with large one-off gains, hence a warning rather than an error.
    revenue, net_income = values.get("revenue"), values.get("net_income")
    if None not in (revenue, net_income) and revenue > 0 and net_income > revenue * 1.05:
        out.append(ConsistencyViolation(
            rule="net_income_exceeds_revenue",
            detail=f"net income {net_income:,.0f} > revenue {revenue:,.0f}",
            severity="warning",
            metrics_involved=["net_income", "revenue"]))

    # Cash is a component of assets.
    cash = values.get("cash_and_equivalents")
    if None not in (cash, assets) and cash > assets:
        out.append(ConsistencyViolation(
            rule="cash_exceeds_assets",
            detail=f"cash {cash:,.0f} > assets {assets:,.0f}",
            severity="error",
            metrics_involved=["cash_and_equivalents", "total_assets"]))

    return out
