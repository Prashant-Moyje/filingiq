"""Walk-forward evaluation of disclosure features against market outcomes.

THE QUESTION
------------
Do LLM-derived disclosure features predict anything beyond fundamentals?

This has to be able to answer NO. A test that cannot fail is not a test, and
the honest expectation here is that it fails: 155 observations, an efficient
and heavily-analysed market, and a 90-day horizon. If a free text feature
predicted excess returns on this sample, the first suspicion should be a bug.

DESIGN CHOICES THAT MATTER
--------------------------

**Walk-forward, never K-fold.** Random folds put FY2024 observations in the
training set and FY2020 in test, which is training on the future. Each fold
here trains on years <= t and tests on year t+1 only.

**Ridge, not gradient boosting.** With ~40 features and ~130 training rows, a
boosted tree will fit the noise perfectly and generalise to nothing. A
regularised linear model is the honest choice at this sample size, and the
coefficient signs are interpretable. LightGBM is included for comparison so
the overfitting can be shown rather than asserted.

**Imputation inside the fold.** Bank filings lack `operating_income`; retailers
lack `rnd_expense`. Median imputation fitted on the full dataset would leak
test-period information into the training features. Fitted per fold, it does
not.

**A permutation test.** With small samples, the difference between feature sets
is usually noise. Shuffling the target and refitting many times gives the
distribution of scores achievable by chance, so "no signal" becomes a measured
statement rather than a shrug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FUNDAMENTAL_FEATURES = [
    "net_margin", "operating_margin", "leverage", "roe", "asset_turnover",
    "cash_ratio", "rnd_intensity", "reporting_lag_days",
]

DISCLOSURE_FEATURES = [
    "drift_score", "n_new", "n_modified", "n_removed", "n_total_risks",
    "removal_rate", "new_rate", "modified_rate",
]

THEME_FEATURES = [
    "theme_supply_chain", "theme_cybersecurity", "theme_regulatory",
    "theme_litigation", "theme_macro", "theme_talent", "theme_climate",
    "theme_concentration", "theme_liquidity", "theme_ai_technology",
]

FEATURE_SETS = {
    "fundamentals": FUNDAMENTAL_FEATURES,
    "disclosure": DISCLOSURE_FEATURES + THEME_FEATURES,
    "combined": FUNDAMENTAL_FEATURES + DISCLOSURE_FEATURES + THEME_FEATURES,
}


@dataclass
class FoldResult:
    train_years: list[int]
    test_year: int
    n_train: int
    n_test: int
    ic: float                 # Spearman rank correlation, prediction vs actual
    auc: float                # AUC for the sign of excess return
    rmse: float
    baseline_rmse: float      # predicting the TRAINING mean

    @property
    def rmse_improvement(self) -> float:
        return (self.baseline_rmse - self.rmse) / self.baseline_rmse


@dataclass
class WalkForwardResult:
    feature_set: str
    model: str
    folds: list[FoldResult] = field(default_factory=list)
    # {test_year: {feature: coefficient}} -- one entry PER FOLD. An earlier
    # version stored a single flat {feature: coefficient} dict that every fold
    # overwrote, so it reported the last fold's coefficients while reading as
    # though it described the model. Coefficients fitted on 52 rows and on 129
    # rows are different objects; collapsing them hides the instability that is
    # the most interesting thing about them at this sample size.
    coefficients: dict[int, dict] = field(default_factory=dict)

    @property
    def last_fold_coefficients(self) -> dict:
        """Coefficients from the final (largest-training-set) fold."""
        return self.coefficients[max(self.coefficients)] if self.coefficients else {}

    def summary(self) -> dict:
        if not self.folds:
            return {}
        ics = [f.ic for f in self.folds if not np.isnan(f.ic)]
        aucs = [f.auc for f in self.folds if not np.isnan(f.auc)]
        imps = [f.rmse_improvement for f in self.folds]
        return {
            "feature_set": self.feature_set,
            "model": self.model,
            "n_folds": len(self.folds),
            "n_test_total": sum(f.n_test for f in self.folds),
            "mean_ic": round(float(np.mean(ics)), 4) if ics else float("nan"),
            "std_ic": round(float(np.std(ics)), 4) if ics else float("nan"),
            "mean_auc": round(float(np.mean(aucs)), 4) if aucs else float("nan"),
            "mean_rmse_improvement": round(float(np.mean(imps)), 4),
        }


def _prepare(df: pd.DataFrame, features: list[str], target: str):
    cols = [c for c in features if c in df.columns]
    missing = set(features) - set(cols)
    if missing:
        log.warning("features absent from the frame: %s", sorted(missing))
    work = df[cols + [target, "fiscal_year", "ticker"]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].replace([np.inf, -np.inf], np.nan)
    return work, cols


def walk_forward(df: pd.DataFrame, feature_set: str = "combined",
                 target: str = "excess_return", model: str = "ridge",
                 alpha: float = 10.0, min_train: int = 40) -> WalkForwardResult:
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score

    features = FEATURE_SETS[feature_set]
    work, cols = _prepare(df, features, target)
    years = sorted(work["fiscal_year"].unique())
    result = WalkForwardResult(feature_set=feature_set, model=model)

    for i in range(1, len(years)):
        test_year = years[i]
        train_years = years[:i]
        tr = work[work["fiscal_year"].isin(train_years)]
        te = work[work["fiscal_year"] == test_year]
        if len(tr) < min_train or len(te) < 5:
            continue

        # Impute and scale using TRAINING statistics only.
        med = tr[cols].median()
        Xtr = tr[cols].fillna(med)
        Xte = te[cols].fillna(med)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
        Xtr = ((Xtr - mu) / sd).values
        Xte = ((Xte - mu) / sd).values
        ytr, yte = tr[target].values, te[target].values

        if model == "ridge":
            est = Ridge(alpha=alpha)
        elif model == "lightgbm":
            from lightgbm import LGBMRegressor
            est = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                num_leaves=7, min_child_samples=10,
                                verbose=-1, random_state=0)
        else:
            raise ValueError(f"unknown model {model!r}")

        est.fit(Xtr, ytr)
        pred = est.predict(Xte)

        ic = float("nan")
        if len(np.unique(pred)) > 1 and len(yte) > 2:
            ic = float(spearmanr(pred, yte).correlation)

        auc = float("nan")
        labels = (yte > 0).astype(int)
        if 0 < labels.sum() < len(labels):
            auc = float(roc_auc_score(labels, pred))

        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        baseline = float(np.sqrt(np.mean((ytr.mean() - yte) ** 2)))

        result.folds.append(FoldResult(
            train_years=[int(y) for y in train_years], test_year=int(test_year),
            n_train=len(tr), n_test=len(te), ic=ic, auc=auc,
            rmse=rmse, baseline_rmse=baseline))

        if model == "ridge":
            result.coefficients[int(test_year)] = dict(zip(cols, est.coef_))

    return result


def permutation_test(df: pd.DataFrame, feature_set: str = "combined",
                     target: str = "excess_return", n_permutations: int = 200,
                     seed: int = 0) -> dict:
    """How good a mean IC is achievable by chance on this data?

    Shuffles the target WITHIN each year, preserving the year structure and
    the target's distribution while destroying any real relationship. If the
    observed IC sits comfortably inside this null distribution, the model has
    found nothing -- and that is a measurement, not an excuse.
    """
    rng = np.random.default_rng(seed)
    observed = walk_forward(df, feature_set, target).summary().get("mean_ic")
    if observed is None or np.isnan(observed):
        return {"observed_ic": float("nan"), "p_value": float("nan")}

    # Drop unusable targets BEFORE shuffling. Permuting a column that still
    # contains NaN moves those NaNs onto different rows each time, so
    # walk_forward's dropna() removes a DIFFERENT set of rows per permutation
    # and the null distribution is built on varying sample sizes. The null must
    # differ from the observed run in the target's ORDER and nothing else.
    df = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()

    null_ics = []
    for _ in range(n_permutations):
        shuffled = df.copy()
        shuffled[target] = (
            shuffled.groupby("fiscal_year")[target]
            .transform(lambda s: rng.permutation(s.values)))
        s = walk_forward(shuffled, feature_set, target).summary().get("mean_ic")
        if s is not None and not np.isnan(s):
            null_ics.append(s)

    if not null_ics:
        return {"observed_ic": observed, "p_value": float("nan")}

    null_ics = np.array(null_ics)
    # Two-sided: how often does chance produce an |IC| at least this large?
    #
    # The +1 in numerator and denominator (Phipson & Smyth 2010) counts the
    # OBSERVED arrangement as one of the permutations, which it is. Without it
    # the estimator can return p = 0.0 -- a claim no finite permutation test
    # can support, and the exact overclaim this layer exists to prevent. The
    # floor becomes 1/(n+1): with 200 shuffles, p >= 0.005.
    n_at_least = int(np.sum(np.abs(null_ics) >= abs(observed)))
    p = (n_at_least + 1) / (len(null_ics) + 1)
    return {
        "observed_ic": round(observed, 4),
        "null_mean_ic": round(float(null_ics.mean()), 4),
        "null_std_ic": round(float(null_ics.std()), 4),
        "null_p95_abs_ic": round(float(np.percentile(np.abs(null_ics), 95)), 4),
        "p_value": round(p, 4),
        "n_permutations": len(null_ics),
    }
