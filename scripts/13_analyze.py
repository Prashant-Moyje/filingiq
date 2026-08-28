#!/usr/bin/env python
"""Week 5: run the analysis graph and produce verified analyst memos.

Usage:
    python scripts/13_analyze.py --tickers AAPL --years 2024
    python scripts/13_analyze.py --tickers AAPL MSFT --years 2023 2024
    python scripts/13_analyze.py --dry-run     # diff + taxonomy only, no LLM
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings                    # noqa: E402
from filingiq.graph.build import analyse_filing, build_graph  # noqa: E402
from filingiq.graph.state import Budget                 # noqa: E402
from filingiq.llm.router import ModelRouter             # noqa: E402
from filingiq.retrieval.embedder import Embedder        # noqa: E402
from filingiq.storage import db                         # noqa: E402


def _quiet():
    for n in ("httpx", "httpcore", "huggingface_hub", "urllib3",
              "sentence_transformers", "transformers", "groq"):
        logging.getLogger(n).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--dry-run", action="store_true",
                    help="deterministic nodes only; no LLM calls, no cost")
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    _quiet()

    router = ModelRouter()
    if not args.dry_run:
        ok, msg = router.healthcheck()
        if not ok:
            print(f"LLM preflight failed: {msg}")
            print("Use --dry-run to exercise the deterministic nodes only.")
            return 1
        print(f"Preflight OK: {msg}")
        router.usage = type(router.usage)()

    embedder = Embedder()

    with db.connect(read_only=True) as con:
        sql = """SELECT accession, ticker, fiscal_year FROM filings
                 WHERE local_path IS NOT NULL"""
        params: list = []
        if args.tickers:
            sql += " AND ticker IN ({})".format(",".join("?" for _ in args.tickers))
            params += [t.upper() for t in args.tickers]
        if args.years:
            sql += " AND fiscal_year IN ({})".format(",".join("?" for _ in args.years))
            params += args.years
        sql += " ORDER BY ticker, fiscal_year"
        filings = con.execute(sql, params).fetchall()

        if not filings:
            print("No filings match.")
            return 1

        # In dry-run the memo node is unreachable, since the budget is zero.
        budget_template = (Budget(max_llm_calls=0, max_tokens=0) if args.dry_run
                           else Budget(max_llm_calls=args.max_calls,
                                       max_tokens=args.max_tokens))
        graph = build_graph(con, embedder, router)
        results = []

        for accession, ticker, fy in filings:
            print(f"\n{'=' * 74}\n{ticker} FY{fy}\n{'=' * 74}")
            state = analyse_filing(graph, ticker, fy, accession,
                                   budget=budget_template.model_copy())

            d = state.get("diff_summary", {})
            if d:
                print(f"  Risk disclosure: {d['new']} new, {d['modified']} modified, "
                      f"{d['unchanged']} unchanged, {d['removed']} removed")
                print(f"  Drift score    : {d['drift_score']}")
            if d and not d.get("has_baseline", True):
                print(f"  ! {d.get('note')}")
            if state.get("risk_themes"):
                print("  New-risk themes: "
                      + ", ".join(f"{k} ({v})" for k, v in state['risk_themes'].items()))
            if state.get("modified_themes"):
                print("  Modified themes: "
                      + ", ".join(f"{k} ({v})"
                                  for k, v in list(state['modified_themes'].items())[:6]))

            cr = state.get("claim_report")
            if cr:
                print(f"\n  Claim verification: {cr['supported']}/"
                      f"{cr['numeric_claims']} numeric claims supported")
                if cr["unsupported"]:
                    print(f"  REMOVED as unverifiable: {cr['unsupported_examples']}")

            if state.get("memo_verified"):
                print(f"\n  --- VERIFIED MEMO ---\n")
                for line in state["memo_verified"].split(". "):
                    if line.strip():
                        print(f"  {line.strip()}.")

            for w in state.get("warnings", []):
                print(f"  ! {w}")
            for e in state.get("errors", []):
                print(f"  ERROR: {e}")

            results.append({
                "ticker": ticker, "fiscal_year": fy, "accession": accession,
                "diff_summary": d, "risk_themes": state.get("risk_themes"),
                "modified_themes": state.get("modified_themes"),
                "claim_report": {k: v for k, v in (cr or {}).items()
                                 if k != "annotated"},
                "memo_verified": state.get("memo_verified"),
                "trace": state.get("trace"),
            })

    out = settings.data_dir / "analysis_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'=' * 74}")
    total_claims = sum(r["claim_report"].get("numeric_claims", 0) for r in results)
    total_bad = sum(r["claim_report"].get("unsupported", 0) for r in results)
    if total_claims:
        print(f"HALLUCINATION GATE: {total_bad}/{total_claims} numeric claims "
              f"unsupported ({total_bad/total_claims:.1%}) across "
              f"{len(results)} memos")
    if not args.dry_run:
        u = router.usage.summary()
        print(f"LLM: {u['calls']} calls, ${u['cost_usd']:.4f}, "
              f"{u['tokens_in']:,} in / {u['tokens_out']:,} out")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
