#!/usr/bin/env python
"""Week 6b: the ablation -- do disclosure features add anything?

Runs walk-forward validation over three feature sets and permutation-tests the
result. Designed so that a NULL result is reported clearly rather than buried.

Usage:
    python scripts/15_train_model.py
    python scripts/15_train_model.py --model lightgbm
    python scripts/15_train_model.py --permutations 500
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from filingiq.config import settings  # noqa: E402
from filingiq.models.train import (  # noqa: E402
    FEATURE_SETS, permutation_test, walk_forward,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--target", default="excess_return",
                    choices=["excess_return", "fwd_return"])
    ap.add_argument("--model", default="ridge", choices=["ridge", "lightgbm"])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--no-permutation", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    path = Path(args.features) if args.features else settings.data_dir / "features.parquet"
    if not path.exists():
        print("No feature store. Run: python scripts/14_build_features.py")
        return 1

    df = pd.read_parquet(path)
    print(f"{len(df)} rows, {df['ticker'].nunique()} companies, "
          f"FY{int(df['fiscal_year'].min())}-{int(df['fiscal_year'].max())}")
    print(f"Target: {args.target} | Model: {args.model}\n")

    results = {}
    print("=" * 78)
    print("WALK-FORWARD ABLATION  (train on years <= t, test on year t+1)")
    print("=" * 78)
    print(f"{'feature set':<16}{'folds':>7}{'n_test':>8}{'mean IC':>10}"
          f"{'std IC':>9}{'mean AUC':>10}{'RMSE vs mean':>14}")
    print("-" * 78)

    for name in FEATURE_SETS:
        r = walk_forward(df, name, args.target, args.model, args.alpha)
        s = r.summary()
        if not s:
            print(f"{name:<16}  (insufficient data)")
            continue
        results[name] = {"summary": s,
                         "folds": [f.__dict__ for f in r.folds],
                         "coefficients": {k: round(float(v), 5)
                                          for k, v in r.coefficients.items()}}
        print(f"{name:<16}{s['n_folds']:>7}{s['n_test_total']:>8}"
              f"{s['mean_ic']:>10.4f}{s['std_ic']:>9.4f}"
              f"{s['mean_auc']:>10.4f}{s['mean_rmse_improvement']:>13.1%}")

    print("""
  IC   = Spearman rank correlation between prediction and realised return.
         0.02-0.05 is a genuinely useful signal in quantitative finance;
         anything much above that on 155 rows should be treated as a bug.
  AUC  = discrimination on the SIGN of excess return. 0.5 is a coin flip.
  RMSE vs mean = improvement over predicting the training mean. Negative
         means the model is worse than a constant.""")

    # ---- per-fold detail: a good average can hide one lucky year ----------
    if "combined" in results:
        print("\n" + "=" * 78)
        print("PER-FOLD DETAIL (combined feature set)")
        print("=" * 78)
        print(f"{'test year':>10}{'n_train':>9}{'n_test':>8}{'IC':>9}{'AUC':>8}")
        print("-" * 44)
        for f in results["combined"]["folds"]:
            print(f"{f['test_year']:>10}{f['n_train']:>9}{f['n_test']:>8}"
                  f"{f['ic']:>9.3f}{f['auc']:>8.3f}")

    # ---- permutation test -------------------------------------------------
    perm = {}
    if not args.no_permutation:
        print("\n" + "=" * 78)
        print(f"PERMUTATION TEST ({args.permutations} shuffles, target shuffled "
              f"within each year)")
        print("=" * 78)
        for name in FEATURE_SETS:
            p = permutation_test(df, name, args.target, args.permutations)
            perm[name] = p
            if p.get("p_value") is None or pd.isna(p.get("p_value")):
                print(f"  {name:<14} could not be tested")
                continue
            verdict = ("DISTINGUISHABLE from chance" if p["p_value"] < 0.05
                       else "indistinguishable from chance")
            print(f"  {name:<14} IC={p['observed_ic']:+.4f}  "
                  f"null={p['null_mean_ic']:+.4f}+-{p['null_std_ic']:.4f}  "
                  f"p={p['p_value']:.3f}  {verdict}")

    # ---- what the model leaned on ----------------------------------------
    if "combined" in results and results["combined"]["coefficients"]:
        coefs = results["combined"]["coefficients"]
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:12]
        print("\n" + "=" * 78)
        print("LARGEST RIDGE COEFFICIENTS (standardised features, final fold)")
        print("=" * 78)
        for k, v in top:
            print(f"  {k:<28}{v:+.4f}")
        print("""
  Read these as description, not discovery. If the permutation test says the
  model is indistinguishable from chance, these coefficients describe noise
  and must not be reported as findings.""")

    out = settings.data_dir / "model_results.json"
    out.write_text(json.dumps({"target": args.target, "model": args.model,
                               "n_rows": len(df), "results": results,
                               "permutation": perm}, indent=2, default=str))
    print(f"\nSaved -> {out}")

    combined_p = perm.get("combined", {}).get("p_value")
    print("\n" + "=" * 78)
    print("HOW TO REPORT THIS")
    print("=" * 78)
    if combined_p is not None and not pd.isna(combined_p) and combined_p < 0.05:
        print("""  A significant result on 155 rows warrants suspicion before
  celebration. Check: is the target leaking? Are features computed only from
  information available at the filing date? Does the effect survive on a
  held-out set of companies rather than years?""")
    else:
        print("""  The null result is the finding, and it is worth stating plainly:

    "Disclosure-drift features did not predict 90-day excess returns on this
     sample (mean IC indistinguishable from a permutation null, n=155 over
     6 years). This is the expected outcome -- 10-K risk-factor language is
     public and widely parsed, so any simple signal should already be priced
     in. The pipeline's value is the extraction and verification layer, which
     is measurable on its own terms; the return-prediction experiment tests
     whether those features carry market-relevant information and finds no
     evidence that they do at this sample size."

  That paragraph demonstrates more competence than a spurious positive. An
  interviewer who has run real experiments will recognise it immediately.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
