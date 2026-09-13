#!/usr/bin/env python
"""Calibrate the diff thresholds against labelled risk-factor pairs.

UNCHANGED_THRESHOLD = 0.95 and MODIFIED_THRESHOLD = 0.80 were asserted, never
fitted. This measures what they actually do.

The alignment is the real one: each current-year factor is scored against its
BEST match in the whole prior-year section (argmax), exactly as
diff_sections does. Scoring isolated pairs would flatter the system -- a
genuinely new risk is not compared with an unrelated factor, it is compared
with whichever prior factor happens to resemble it most, and that is the
number the threshold has to survive.

Usage:
    python scripts/16_calibrate_diff.py
    python scripts/16_calibrate_diff.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from filingiq.analysis.calibration_set import (  # noqa: E402
    current_year_section, prior_year_section,
)
from filingiq.analysis.diff import (  # noqa: E402
    MODIFIED_THRESHOLD, UNCHANGED_THRESHOLD,
)

LABELS = ("unchanged", "modified", "new")


def classify(score: float, unchanged_t: float, modified_t: float) -> str:
    if score >= unchanged_t:
        return "unchanged"
    if score >= modified_t:
        return "modified"
    return "new"


def best_scores(embedder):
    """Max cosine of each current factor against the whole prior section."""
    current = current_year_section()
    prior = prior_year_section()
    cur_vecs = embedder.embed_passages([t for _, t, _ in current],
                                       show_progress=False)
    pri_vecs = embedder.embed_passages([t for _, t in prior],
                                       show_progress=False)
    sim = cur_vecs @ pri_vecs.T          # embedder L2-normalises, so dot == cos
    rows = []
    for i, (topic, _, label) in enumerate(current):
        j = int(np.argmax(sim[i]))
        rows.append({"topic": topic, "true": label,
                     "best_match": prior[j][0], "score": float(sim[i, j])})
    return rows


def accuracy(rows, unchanged_t, modified_t) -> float:
    return sum(classify(r["score"], unchanged_t, modified_t) == r["true"]
               for r in rows) / len(rows)


def per_class(rows, unchanged_t, modified_t) -> dict:
    out = {}
    for label in LABELS:
        tp = sum(1 for r in rows
                 if r["true"] == label
                 and classify(r["score"], unchanged_t, modified_t) == label)
        predicted = sum(1 for r in rows
                        if classify(r["score"], unchanged_t, modified_t) == label)
        actual = sum(1 for r in rows if r["true"] == label)
        out[label] = {
            "precision": round(tp / predicted, 3) if predicted else None,
            "recall": round(tp / actual, 3) if actual else None,
            "support": actual,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write full results here")
    args = ap.parse_args()

    from filingiq.retrieval.embedder import Embedder
    print("loading embedder ...")
    rows = best_scores(Embedder())

    print("\n" + "=" * 74)
    print("BEST-MATCH COSINE BY TRUE LABEL")
    print("=" * 74)
    print(f"{'true':>10}{'topic':>20}{'best prior match':>20}{'cosine':>10}")
    print("-" * 74)
    for label in LABELS:
        for r in [x for x in rows if x["true"] == label]:
            print(f"{label:>10}{r['topic']:>20}{r['best_match']:>20}"
                  f"{r['score']:>10.4f}")

    print("\n" + "=" * 74)
    print("SEPARATION")
    print("=" * 74)
    stats = {}
    for label in LABELS:
        s = [r["score"] for r in rows if r["true"] == label]
        stats[label] = {"n": len(s), "min": min(s), "max": max(s),
                        "mean": float(np.mean(s))}
        print(f"  {label:<10} n={len(s)}  min={min(s):.4f}  "
              f"mean={np.mean(s):.4f}  max={max(s):.4f}")

    gap_um = stats["unchanged"]["min"] - stats["modified"]["max"]
    gap_mn = stats["modified"]["min"] - stats["new"]["max"]
    print(f"\n  unchanged/modified margin: {gap_um:+.4f}"
          f"   {'separable' if gap_um > 0 else 'OVERLAPPING'}")
    print(f"  modified/new       margin: {gap_mn:+.4f}"
          f"   {'separable' if gap_mn > 0 else 'OVERLAPPING'}")

    # ---- shipped thresholds ------------------------------------------
    print("\n" + "=" * 74)
    print(f"SHIPPED THRESHOLDS  unchanged>={UNCHANGED_THRESHOLD}  "
          f"modified>={MODIFIED_THRESHOLD}")
    print("=" * 74)
    shipped_acc = accuracy(rows, UNCHANGED_THRESHOLD, MODIFIED_THRESHOLD)
    print(f"  accuracy {shipped_acc:.1%}")
    for label, m in per_class(rows, UNCHANGED_THRESHOLD, MODIFIED_THRESHOLD).items():
        print(f"    {label:<10} precision={m['precision']}  "
              f"recall={m['recall']}  n={m['support']}")
    print("\n  misclassified:")
    bad = [r for r in rows
           if classify(r["score"], UNCHANGED_THRESHOLD, MODIFIED_THRESHOLD)
           != r["true"]]
    for r in bad:
        got = classify(r["score"], UNCHANGED_THRESHOLD, MODIFIED_THRESHOLD)
        print(f"    {r['topic']:<20} true={r['true']:<10} got={got:<10} "
              f"cos={r['score']:.4f}")
    if not bad:
        print("    (none)")

    # ---- sweep --------------------------------------------------------
    grid_u = [round(x, 2) for x in np.arange(0.80, 0.995, 0.01)]
    grid_m = [round(x, 2) for x in np.arange(0.60, 0.96, 0.01)]
    scored = [(accuracy(rows, u, m), u, m)
              for u, m in product(grid_u, grid_m) if m < u]
    best_acc = max(s[0] for s in scored)
    ties = sorted({(u, m) for a, u, m in scored if a == best_acc})

    # When two classes separate cleanly, ANY threshold in the gap scores the
    # same on this set. Accuracy therefore cannot choose between 77 tied pairs,
    # and picking one by accuracy alone would overfit 10 points. The midpoint
    # of each gap is the max-margin choice: it is the threshold furthest from
    # the nearest labelled example on either side, so it is the one most likely
    # to survive text this set does not contain.
    mid_u = (stats["unchanged"]["min"] + stats["modified"]["max"]) / 2
    mid_m = (stats["modified"]["min"] + stats["new"]["max"]) / 2

    print("\n" + "=" * 74)
    print("THRESHOLD SWEEP")
    print("=" * 74)
    print(f"  shipped        {shipped_acc:>6.1%} at unchanged>="
          f"{UNCHANGED_THRESHOLD}  modified>={MODIFIED_THRESHOLD}")
    print(f"  best on grid   {best_acc:>6.1%} ({len(ties)} tied pairs -- "
          f"accuracy cannot choose between them)")
    print(f"  max-margin            at unchanged>={mid_u:.2f}  "
          f"modified>={mid_m:.2f}")
    print(f"     accuracy {accuracy(rows, round(mid_u, 2), round(mid_m, 2)):.1%}")

    print("\n  HOW MUCH ROOM EACH SHIPPED THRESHOLD HAS")
    print(f"    0.95 sits {UNCHANGED_THRESHOLD - stats['unchanged']['min']:+.4f} "
          f"from the lowest 'unchanged' example ({stats['unchanged']['min']:.4f})")
    print(f"    0.80 sits {MODIFIED_THRESHOLD - stats['modified']['min']:+.4f} "
          f"from the lowest 'modified' example ({stats['modified']['min']:.4f})")
    print("    A positive number means the threshold is ABOVE a case it is")
    print("    supposed to admit -- that case is being misclassified now.")
    best = (best_acc, round(mid_u, 2), round(mid_m, 2))

    if args.json:
        args.json.write_text(json.dumps(
            {"rows": rows, "stats": stats,
             "shipped": {"unchanged": UNCHANGED_THRESHOLD,
                         "modified": MODIFIED_THRESHOLD,
                         "accuracy": shipped_acc},
             "best": {"unchanged": best[1], "modified": best[2],
                      "accuracy": best_acc}},
            indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.json}")

    print("""
  These labels are ground truth BY CONSTRUCTION, not human judgments on real
  filings. A transformation written here is not drawn from the same
  distribution as one written by a securities lawyer. Treat the result as a
  calibrated starting point with a measured basis -- which is more than the
  shipped values had -- not as a substitute for labelled real filings.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
