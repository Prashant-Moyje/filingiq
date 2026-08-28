#!/usr/bin/env python
"""Inspect extraction results: what abstained, what failed, and why.

The summary table tells you the score. This tells you what to fix.

Three questions it answers:

  1. WHICH metrics abstain? If the same metric abstains on every filing, the
     problem is retrieval or the metric definition -- not the model. Abstention
     concentrated in one metric is a fixable bug; abstention spread evenly is
     genuine uncertainty.

  2. WHAT did the model actually read? For every wrong or abstained figure,
     the retrieved chunk and the model's quote are shown. Reading five of these
     is worth more than any amount of prompt speculation.

  3. WHERE does the time and money go? Per-metric latency and tokens, so
     optimisation targets the expensive metrics rather than being applied
     uniformly.

Usage:
    python scripts/12_inspect_extractions.py
    python scripts/12_inspect_extractions.py --show-abstained
    python scripts/12_inspect_extractions.py --metric operating_cash_flow
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings  # noqa: E402
from filingiq.storage import db       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--metric", help="drill into one metric")
    ap.add_argument("--show-abstained", action="store_true",
                    help="print the retrieved context for abstentions")
    ap.add_argument("--max-context", type=int, default=700)
    args = ap.parse_args()

    path = Path(args.results) if args.results else settings.data_dir / "extraction_results.json"
    if not path.exists():
        print("No results. Run scripts/10_extract.py first.")
        return 1

    data = json.loads(path.read_text())
    rows = data["rows"]
    if args.metric:
        rows = [r for r in rows if r["metric"] == args.metric]

    # --- per-metric outcome table ----------------------------------------
    by_metric: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "exact": 0, "close": 0, "wrong": 0, "abstained": 0,
                 "no_gt": 0})
    for r in rows:
        m = by_metric[r["metric"]]
        m["n"] += 1
        band = r["band"]
        if band == "abstained":
            m["abstained"] += 1
        elif band == "no_ground_truth":
            m["no_gt"] += 1
        elif band == "exact":
            m["exact"] += 1
        elif band == "wrong":
            m["wrong"] += 1
        else:
            m["close"] += 1

    print("=" * 82)
    print("PER-METRIC OUTCOMES")
    print("=" * 82)
    print(f"{'metric':<24}{'n':>4}{'exact':>7}{'close':>7}{'wrong':>7}"
          f"{'abstain':>9}{'no_gt':>7}   verdict")
    print("-" * 82)
    for metric, m in sorted(by_metric.items(),
                            key=lambda kv: -kv[1]["abstained"]):
        scored = m["exact"] + m["close"] + m["wrong"]
        if m["abstained"] == m["n"]:
            verdict = "ALWAYS abstains -> retrieval or definition"
        elif m["abstained"] > m["n"] / 2:
            verdict = "mostly abstains -> check retrieval"
        elif m["wrong"] > 0:
            verdict = "errors -> read the quotes"
        elif scored and m["exact"] == scored:
            verdict = "clean"
        else:
            verdict = ""
        print(f"{metric:<24}{m['n']:>4}{m['exact']:>7}{m['close']:>7}"
              f"{m['wrong']:>7}{m['abstained']:>9}{m['no_gt']:>7}   {verdict}")

    # --- failures with evidence ------------------------------------------
    failures = [r for r in rows if r["band"] == "wrong"]
    if failures:
        print("\n" + "=" * 82)
        print("WRONG ANSWERS -- with the model's own quote")
        print("=" * 82)
        for r in failures:
            print(f"\n{r['ticker']} FY{r['fiscal_year']} {r['metric']} "
                  f"[{r['error_type']}]")
            print(f"  extracted : {r['extracted']:,.2f}" if r['extracted']
                  else "  extracted : None")
            print(f"  truth     : {r['truth']:,.2f}" if r['truth']
                  else "  truth     : None")
            if r["extracted"] and r["truth"]:
                print(f"  ratio     : {r['extracted'] / r['truth']:.6f}")
            print(f"  as stated : {r['value_as_stated']} (scale={r['scale']})")
            print(f"  quote     : {r['quote']}")

    # --- abstentions with the context the model was given ----------------
    abstained = [r for r in rows if r["band"] == "abstained"]
    if abstained:
        print("\n" + "=" * 82)
        print(f"ABSTENTIONS ({len(abstained)})")
        print("=" * 82)
        for r in abstained:
            print(f"  {r['ticker']} FY{r['fiscal_year']:<6} {r['metric']:<24}"
                  f"truth={r['truth'] if r['truth'] is not None else 'no ground truth'}")

        if args.show_abstained:
            print("\n" + "-" * 82)
            print("WHAT WAS RETRIEVED for the first few abstentions")
            print("-" * 82)
            print("If the figure IS present in this text, the prompt is at fault.")
            print("If it is NOT, retrieval is at fault. That distinction decides")
            print("what to fix, and guessing it wrong costs a day.\n")
            with db.connect(read_only=True) as con:
                for r in abstained[:4]:
                    ids = r.get("retrieved_chunk_ids") or []
                    if not ids:
                        print(f"\n### {r['ticker']} FY{r['fiscal_year']} "
                              f"{r['metric']}: no retrieval record "
                              "(re-run scripts/10_extract.py to capture it)")
                        continue
                    ph = ",".join("?" for _ in ids)
                    hits = con.execute(
                        f"""SELECT item, substr(raw_text, 1, ?) FROM chunks
                            WHERE chunk_id IN ({ph})""",
                        [args.max_context] + list(ids),
                    ).fetchall()
                    print(f"\n### {r['ticker']} FY{r['fiscal_year']} "
                          f"{r['metric']} (truth={r['truth']})")
                    for item, text in hits:
                        print(f"  [Item {item}] {text[:args.max_context]}...")

    # --- cost and latency -------------------------------------------------
    usage = data.get("usage", {})
    print("\n" + "=" * 82)
    print("COST AND LATENCY")
    print("=" * 82)
    print(f"  model        : {data.get('model')}")
    print(f"  calls        : {usage.get('calls')}")
    print(f"  tokens in/out: {usage.get('tokens_in', 0):,} / "
          f"{usage.get('tokens_out', 0):,}")
    if usage.get("calls"):
        print(f"  avg input    : {usage.get('tokens_in', 0)//usage['calls']:,} "
              f"tokens/call")
    print(f"  cost         : ${usage.get('cost_usd', 0):.4f}")
    print(f"  latency      : p50={usage.get('p50_ms')}ms  p95={usage.get('p95_ms')}ms")
    print("""
  If latency is high on a reasoning model, the input size is usually the
  driver. Fewer or shorter excerpts cut both cost and time; whether that
  costs accuracy is an ablation worth running (--k 4 vs --k 6 vs --k 8).
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
