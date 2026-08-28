#!/usr/bin/env python
"""Week 4: extract financial figures and verify every one against XBRL.

This produces the headline metric: extraction accuracy against objective
ground truth, with an error taxonomy that says what to fix.

Usage:
    python scripts/10_extract.py --tickers AAPL --years 2024        # start small
    python scripts/10_extract.py                                    # all filings
    python scripts/10_extract.py --provider ollama --model qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings                              # noqa: E402
from filingiq.extraction.agent import ExtractionAgent             # noqa: E402
from filingiq.extraction.schema import METRIC_DEFINITIONS         # noqa: E402
from filingiq.extraction.verify import summarise, verify_figure   # noqa: E402
from filingiq.llm.router import ModelRouter                       # noqa: E402
from filingiq.retrieval.embedder import Embedder                  # noqa: E402
from filingiq.retrieval.hybrid import HybridRetriever             # noqa: E402
from filingiq.retrieval.store import LexicalStore, VectorStore    # noqa: E402
from filingiq.storage import db                                   # noqa: E402


def _quiet() -> None:
    for n in ("httpx", "httpcore", "huggingface_hub", "urllib3",
              "sentence_transformers", "transformers", "groq"):
        logging.getLogger(n).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--metrics", nargs="*", default=list(METRIC_DEFINITIONS))
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rps", type=float, default=0.5)
    ap.add_argument("--list-models", action="store_true",
                    help="show models your API key can currently reach, then exit")
    ap.add_argument("--auto-model", action="store_true",
                    help="pick the best available model automatically")
    ap.add_argument("--no-group", action="store_true",
                    help="one call per metric (10x tokens; for blame attribution)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract filings already checkpointed")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    _quiet()

    router = ModelRouter(provider=args.provider, model=args.model,
                         requests_per_second=args.rps)
    if args.list_models:
        models = router.list_models()
        if not models:
            print("Could not list models. Check GROQ_API_KEY in .env.")
            return 1
        print(f"{len(models)} models available to this key:\n")
        for m in models:
            print(f"  {m}")
        return 0

    if args.auto_model:
        picked = router.autoselect_model()
        if picked:
            print(f"Auto-selected model: {picked}")
            router.model = picked

    print(f"Provider: {router.provider} | Model: {router.model}")
    ok, msg = router.healthcheck()

    # A dead model name is recoverable: ask the provider what it serves and
    # retry once. Better than making the user go read a deprecation page.
    if not ok and "model_not_found" in str(msg):
        print(f"Model '{router.model}' is unavailable (likely deprecated).")
        picked = router.autoselect_model()
        if picked:
            print(f"Retrying with '{picked}'...")
            router.model = picked
            ok, msg = router.healthcheck()

    if not ok:
        print(f"\nLLM PREFLIGHT FAILED: {msg}\n")
        print("  Check .env in the project root:")
        print("    LLM_PROVIDER=groq")
        print("    GROQ_API_KEY=gsk_...        (free at console.groq.com/keys)")
        print("    GROQ_MODEL=openai/gpt-oss-120b")
        print("\n  See what your key can reach:")
        print("    python scripts/10_extract.py --list-models")
        print("\n  For local Ollama instead: LLM_PROVIDER=ollama and "
              "`ollama serve` running.")
        print("\n  Aborting rather than producing a report from failed calls.")
        return 1
    print(f"Preflight OK: {msg}")
    router.usage = type(router.usage)()   # don't bill the healthcheck

    embedder = Embedder()
    vs = VectorStore(settings.data_dir / "qdrant", dim=embedder.dim)
    ls = LexicalStore.load(settings.data_dir / "bm25.pkl")
    agent = ExtractionAgent(HybridRetriever(vs, ls, embedder), router,
                            k=args.k, mode=args.mode)

    with db.connect(read_only=True) as con:
        sql = """SELECT accession, ticker, fiscal_year, period_of_report
                 FROM filings WHERE local_path IS NOT NULL"""
        params: list = []
        if args.tickers:
            sql += " AND ticker IN ({})".format(",".join("?" for _ in args.tickers))
            params += [t.upper() for t in args.tickers]
        if args.years:
            sql += " AND fiscal_year IN ({})".format(",".join("?" for _ in args.years))
            params += args.years
        sql += " ORDER BY ticker, fiscal_year"
        filings = con.execute(sql, params).fetchall()

        truth: dict[tuple[str, str], float] = {}
        prior: dict[tuple[str, int, str], float] = {}
        for acc, tic, fy, metric, val in con.execute(
                "SELECT accession, ticker, fiscal_year, metric, value FROM ground_truth"
        ).fetchall():
            truth[(acc, metric)] = val
            prior[(tic, fy, metric)] = val

    if not filings:
        print("No filings match. Run scripts/01_ingest.py first.")
        return 1

    # ---- CHECKPOINTING ---------------------------------------------------
    # A 429 mid-run previously destroyed six filings of completed work. Results
    # are now persisted per filing, and a resumed run skips what is already
    # done. Any pipeline whose unit of work costs money or minutes needs this;
    # discovering you need it after losing an hour is the expensive way.
    ckpt = settings.data_dir / "extraction_checkpoint.json"
    done: dict[str, list] = {}
    if ckpt.exists() and not args.force:
        done = json.loads(ckpt.read_text())
        if done:
            print(f"Resuming: {len(done)} filings already extracted "
                  f"(--force to redo)")

    pending = [f for f in filings if f[0] not in done]

    from filingiq.extraction.agent import METRIC_GROUPS
    calls_per_filing = (len(args.metrics) if args.no_group
                        else sum(1 for g in METRIC_GROUPS.values()
                                 if any(m in args.metrics for m in g)))
    n_calls = len(pending) * calls_per_filing
    est_tokens = n_calls * (3300 if args.no_group else 3800)

    print(f"{len(pending)} filings to process x {calls_per_filing} calls "
          f"= {n_calls} LLM calls")
    print(f"Estimated input tokens: ~{est_tokens:,}")
    if est_tokens > 180_000:
        print(f"\n  WARNING: Groq's free tier allows 200,000 tokens/day.")
        print(f"  This run needs ~{est_tokens:,}. Options:")
        print(f"    - run in batches: --tickers AAPL MSFT   (resumes safely)")
        print(f"    - fewer excerpts: --k 4")
        print(f"    - smaller model : --model openai/gpt-oss-20b")
    print()

    all_verdicts, rows = [], []
    for accession, rowlist in done.items():
        rows.extend(rowlist)

    for accession, ticker, fy, period_end in pending:
        try:
            res = agent.extract_filing(ticker, fy, accession, args.metrics,
                                       str(period_end or ""),
                                       grouped=not args.no_group)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            print(f"\n{ticker} FY{fy} failed: {msg[:160]}")
            if "rate_limit" in msg or "429" in msg:
                print(f"\n  Rate limit reached. {len(done)} filings are "
                      f"checkpointed and safe.")
                print("  Re-run the same command later to resume where it "
                      "stopped.")
            break
        filing_truth = {m: truth.get((accession, m)) for m in args.metrics}

        verdicts = []
        n_errored = 0
        for fig in res.figures:
            # A failed call is not an abstention. Marking it as one lets an
            # outage look like caution and silently corrupts the accuracy
            # denominator.
            errored = bool(fig.reasoning and fig.reasoning.startswith("llm error"))
            n_errored += errored
            v = verify_figure(
                fig.metric,
                fig.normalized_value if fig.is_usable() else None,
                truth.get((accession, fig.metric)),
                all_truth={k: v for k, v in filing_truth.items() if v is not None},
                prior_year_truth=prior.get((ticker, fy - 1, fig.metric)),
                errored=errored,
            )
            verdicts.append(v)
            rows.append({"accession": accession, "ticker": ticker, "fiscal_year": fy,
                         "metric": fig.metric, "extracted": v.extracted,
                         "truth": v.truth, "band": v.band, "error_type": v.error_type,
                         "confidence": fig.confidence, "scale": fig.scale,
                         "value_as_stated": fig.value_as_stated,
                         "chunk_id": fig.chunk_id, "quote": fig.quote,
                         "retrieved_chunk_ids": fig.retrieved_chunk_ids,
                         "reasoning": fig.reasoning,
                         "abs_error": v.abs_error, "rel_error": v.rel_error})
        # NEVER checkpoint a filing whose calls failed -- a resumed run would
        # skip it, treating a rate-limited filing as permanently complete and
        # baking the corruption into every future report.
        if n_errored:
            rows = [r for r in rows if r["accession"] != accession]
            print(f"  {ticker} FY{fy}: {n_errored} calls FAILED -- not "
                  f"checkpointed, results discarded")
            if n_errored >= len(res.figures) / 2:
                print("\n  Aborting: most calls are failing (likely a rate "
                      "limit).")
                print(f"  {len(done)} filings are safely checkpointed.")
                print("  Re-run the same command later to resume.")
                break
            continue

        done[accession] = [r for r in rows if r["accession"] == accession]
        ckpt.write_text(json.dumps(done, indent=1, default=str))

        s = summarise(verdicts)
        print(f"{ticker} FY{fy}: exact={s['exact']:.0%} within2%={s['within_2pct']:.0%} "
              f"abstained={s['n_abstained']}/{s['n_total']} cost=${res.cost_usd:.4f}")

    # ---- RETRIEVAL CEILING ----------------------------------------------
    # The single most useful diagnostic in a RAG pipeline: was the correct
    # answer even PRESENT in the context the model was given?
    #
    #   ceiling  = % of figures whose true value appeared in a retrieved chunk
    #   accuracy = % the model got right
    #
    # accuracy can never exceed ceiling. If accuracy is far below ceiling, fix
    # the prompt. If ceiling itself is low, fix retrieval -- no prompt work
    # will help, because the answer was never on the page.
    from filingiq.retrieval.evalset import value_variants

    present, checkable = 0, 0
    with db.connect(read_only=True) as con:
        for r in rows:
            if r["truth"] is None or not r.get("retrieved_chunk_ids"):
                continue
            checkable += 1
            variants = value_variants(r["truth"], r["metric"])
            if not variants:
                continue
            ph = ",".join("?" for _ in r["retrieved_chunk_ids"])
            clauses = " OR ".join("raw_text LIKE ?" for _ in variants)
            hit = con.execute(
                f"SELECT count(*) FROM chunks WHERE chunk_id IN ({ph}) AND ({clauses})",
                list(r["retrieved_chunk_ids"]) + [f"%{v}%" for v in variants],
            ).fetchone()[0]
            if hit:
                present += 1
    ceiling = present / max(checkable, 1)

    # Rebuild verdicts across everything, checkpointed and fresh alike.
    from filingiq.extraction.verify import Verdict as _V
    all_verdicts = [
        _V(metric=r["metric"], extracted=r["extracted"], truth=r["truth"],
           abs_error=r.get("abs_error"), rel_error=r.get("rel_error"),
           band=r["band"],
           error_type=r["error_type"], verified=(r["band"] not in ("wrong", "abstained")),
           abstained=(r["band"] == "abstained"))
        for r in rows]

    print("\n" + "=" * 74)
    print("RETRIEVAL CEILING -- was the answer even in the context?")
    print("=" * 74)
    print(f"  True value present in retrieved chunks: {present}/{checkable} "
          f"({ceiling:.1%})")
    print(f"  This is the maximum accuracy achievable without changing retrieval.")

    print("\n" + "=" * 74)
    print("EXTRACTION ACCURACY vs XBRL GROUND TRUTH")
    print("=" * 74)
    s = summarise(all_verdicts)
    print(f"  Figures attempted     : {s['n_total']}")
    print(f"  Abstained (found=false): {s['n_abstained']} ({s['abstention_rate']:.1%})")
    if s.get("n_errored"):
        print(f"  FAILED calls          : {s['n_errored']} "
              f"({s['error_rate']:.1%})  <-- not abstentions; results incomplete")
    print(f"  Scored                : {s['n_scorable']}")
    print(f"\n  Exact match           : {s['exact']:.1%}")
    print(f"  Within 0.5%           : {s['within_0.5pct']:.1%}")
    print(f"  Within 2%             : {s['within_2pct']:.1%}")
    print(f"\n  Median abs rel error  : {s['median_abs_rel_error']}")
    if s["error_types"]:
        print("\n  Error taxonomy (what to fix):")
        for k, v in s["error_types"].items():
            print(f"    {k:<34} {v}")
    u = router.usage.summary()
    print(f"\n  LLM calls={u['calls']} errors={u['errors']} "
          f"repairs={agent.repairs} schema_failures={agent.schema_failures}")
    print(f"  Tokens in/out: {u['tokens_in']:,}/{u['tokens_out']:,}")
    print(f"  Cost: ${u['cost_usd']:.4f} total, "
          f"${u['cost_usd']/max(len(filings),1):.4f}/filing")
    print(f"  Latency p50={u['p50_ms']}ms p95={u['p95_ms']}ms")

    # Accuracy must be measured over SCORABLE figures only. Dividing by all
    # attempted rows counts "no XBRL ground truth" as a failure, which
    # penalises exactly the companies whose filings differ most -- JPM has 10
    # metrics with no ground truth (banks do not report R&D and tag revenue
    # differently), so the naive denominator showed 50% for a company scoring
    # near 100% on everything actually checkable.
    by_ticker: dict = {}
    for r in rows:
        d = by_ticker.setdefault(r["ticker"],
                                 {"n": 0, "scored": 0, "ok": 0, "abst": 0,
                                  "no_gt": 0, "wrong": 0})
        d["n"] += 1
        band = r["band"]
        if band in ("exact", "within_0.5pct", "within_2pct"):
            d["ok"] += 1
            d["scored"] += 1
        elif band == "wrong":
            d["wrong"] += 1
            d["scored"] += 1
        elif band == "abstained":
            d["abst"] += 1
        elif band == "no_ground_truth":
            d["no_gt"] += 1

    print("\n  Per company -- accuracy over SCORABLE figures:")
    print(f"    {'':<6}{'correct':>9}{'scored':>8}{'accuracy':>10}"
          f"{'abstain':>9}{'no_gt':>7}")
    for tic, d in sorted(by_ticker.items()):
        acc_t = d["ok"] / d["scored"] if d["scored"] else float("nan")
        print(f"    {tic:<6}{d['ok']:>9}{d['scored']:>8}{acc_t:>9.0%}"
              f"{d['abst']:>9}{d['no_gt']:>7}")
    print("    (no_gt = XBRL has no value for that metric, so it cannot be "
          "scored either way)")

    acc = s["within_2pct"] * (s["n_scorable"] / max(s["n_total"], 1))
    print(f"\n  DIAGNOSIS: ceiling={ceiling:.1%}, end-to-end accuracy={acc:.1%}")
    if ceiling < 0.7:
        print("  -> RETRIEVAL is the bottleneck. Prompt changes cannot help;")
        print("     the figure was not in the context. Fix the query or k.")
    elif acc < ceiling * 0.95:
        gap = ceiling - acc
        print(f"  -> GENERATION is the bottleneck. The figure was present in")
        print(f"     {ceiling:.1%} of contexts but only {acc:.1%} were extracted")
        print(f"     correctly -- a {gap:.1%} gap that retrieval work cannot close.")
        print("     Fix the prompt, the model, or the grouping.")
    elif ceiling >= 0.99 and acc >= 0.99:
        print("  -> At ceiling with full accuracy. Widen the corpus before")
        print("     drawing conclusions; this is too clean to be the truth.")
    else:
        print("  -> Balanced. Raise the ceiling to raise accuracy.")

    out = settings.data_dir / "extraction_results.json"
    out.write_text(json.dumps({"summary": s, "usage": u, "model": router.model,
                               "retrieval_ceiling": ceiling,
                               "rows": rows}, indent=2, default=str))
    print(f"\nSaved -> {out}")
    print("""
WHAT TO DO WITH THE ERROR TAXONOMY
  scale_error_*      -> prompt problem: the model missed "(in millions)".
                        Try surfacing the table header more prominently.
  wrong_period_*     -> retrieval/context problem: it read the comparative
                        column. Consider passing the period end explicitly.
  wrong_metric_*     -> it found the wrong line item. Sharpen the definition.
  unexplained        -> read the `quote` field for those rows; that is where
                        the interesting failures live.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
