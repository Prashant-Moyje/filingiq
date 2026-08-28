#!/usr/bin/env python
"""Week 3b: the ablation table.

This script produces the numbers that go in EVALUATION.md and on your resume.
It runs the SAME queries through four retrieval configurations so the
comparison is apples to apples:

    dense    -- embeddings only
    sparse   -- BM25 only
    hybrid   -- RRF fusion of both
    rerank   -- hybrid + cross-encoder (optional, slow, usually best)

An ablation is far more persuasive than a single number. "Hybrid retrieval
achieves 0.87 Recall@5" invites the question "compared to what?". "Hybrid
lifted Recall@5 from 0.71 to 0.87, and reranking took it to 0.93" answers it.

Usage:
    python scripts/09_eval_retrieval.py
    python scripts/09_eval_retrieval.py --with-rerank
    python scripts/09_eval_retrieval.py --modes dense sparse --limit 40
"""
from __future__ import annotations

import argparse, json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings                        # noqa: E402
from filingiq.retrieval import evalset as evalset_mod       # noqa: E402
from filingiq.retrieval.embedder import Embedder            # noqa: E402
from filingiq.retrieval.hybrid import HybridRetriever, Reranker  # noqa: E402
from filingiq.retrieval.metrics import (                    # noqa: E402
    bootstrap_ci, evaluate, paired_bootstrap_test, per_query_scores,
)
from filingiq.retrieval.store import LexicalStore, VectorStore   # noqa: E402


def _quiet_libs() -> None:
    """Silence per-request HTTP logging from huggingface_hub and httpx."""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*",
                    default=["dense", "sparse", "hybrid"])
    ap.add_argument("--with-rerank", action="store_true")
    ap.add_argument("--reranker", default="fast",
                    choices=["fast", "base", "large"],
                    help="fast=MiniLM 22M (default), large=bge-v2-m3 568M")
    ap.add_argument("--rerank-candidates", type=int, default=25,
                    help="candidates passed to the cross-encoder")
    ap.add_argument("--limit", type=int, default=None,
                    help="stratified random sample (see --seed), NOT the first N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", default="hybrid",
                    help="mode each other mode is significance-tested against")
    ap.add_argument("--no-significance", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--evalset", default=None)
    ap.add_argument("--use-filters", action="store_true",
                    help="pre-filter by ticker + fiscal year before searching")
    args = ap.parse_args()
    _quiet_libs()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    eval_path = Path(args.evalset) if args.evalset else settings.data_dir / "evalset.json"
    if not eval_path.exists():
        print("No eval set. Run scripts/08_build_evalset.py first.")
        return 1
    queries = evalset_mod.load(eval_path)
    if args.limit and args.limit < len(queries):
        # The eval set is ordered: all numeric queries, then all qualitative.
        # Taking the first N therefore samples ONLY numeric queries -- an
        # unrepresentative slice that silently changes what is being measured.
        # Sample proportionally from each family instead, with a fixed seed so
        # runs remain comparable. See FM-009.
        import random
        rng = random.Random(args.seed)
        by_kind: dict[str, list] = {}
        for q in queries:
            by_kind.setdefault(q.kind, []).append(q)
        total = len(queries)
        sampled = []
        for kind, group in by_kind.items():
            take = max(1, round(args.limit * len(group) / total))
            sampled.extend(rng.sample(group, min(take, len(group))))
        queries = sampled[:args.limit]
        from collections import Counter
        print(f"Stratified sample of {len(queries)}: "
              f"{dict(Counter(q.kind for q in queries))} (seed={args.seed})")

    embedder = Embedder()
    vs = VectorStore(settings.data_dir / "qdrant", dim=embedder.dim)
    ls = LexicalStore.load(settings.data_dir / "bm25.pkl")
    reranker = None
    if args.with_rerank:
        from filingiq.retrieval.hybrid import RERANKER_MODELS
        reranker = Reranker(model_name=RERANKER_MODELS[args.reranker])
        print(f"Reranker: {reranker.model_name} "
              f"on {args.rerank_candidates} candidates/query")
    retriever = HybridRetriever(vs, ls, embedder, reranker)

    modes = list(args.modes)
    if args.with_rerank and "rerank" not in modes:
        modes.append("rerank")

    qrels = {q.query_id: set(q.relevant_chunks) for q in queries}

    # Recall@k is bounded above by k / |relevant|. If label sets are large,
    # recall cannot approach 1.0 no matter how good the retriever is, and
    # quoting it without this context is misleading.
    sizes = sorted(len(v) for v in qrels.values())
    med = sizes[len(sizes) // 2]
    print(f"Label sets: min={sizes[0]} median={med} max={sizes[-1]}")
    print(f"Recall@5 ceiling at median label size: {min(1.0, 5/med):.3f}")
    if args.use_filters:
        print("Metadata pre-filtering: ON (ticker + fiscal_year)")
    all_results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for mode in modes:
        print(f"\nRunning {mode} over {len(queries)} queries...")
        results: dict[str, list[str]] = {}
        t0 = time.time()
        for i, q in enumerate(queries, 1):
            # In production a question naming a company and year would be
            # routed with metadata filters rather than relying on the embedding
            # to encode "AAPL" and "2024". Measuring both shows how much of the
            # work is done by retrieval versus by routing.
            filters = ({"ticker": q.ticker, "fiscal_year": q.fiscal_year}
                       if args.use_filters else None)
            hits = retriever.search(q.question, k=args.k, mode=mode,
                                    filters=filters,
                                    rerank_candidates=args.rerank_candidates)
            results[q.query_id] = [h.chunk_id for h in hits]
            # Reranking is slow enough that silence looks like a hang.
            step = 5 if mode == "rerank" else 25
            if i % step == 0:
                rate = (time.time() - t0) / i
                eta = rate * (len(queries) - i)
                print(f"  {i}/{len(queries)}  ({rate:.2f}s/query, "
                      f"~{eta/60:.1f} min left)", flush=True)
        timings[mode] = time.time() - t0
        all_results[mode] = results

    # ---- overall table --------------------------------------------------
    print("\n" + "=" * 84)
    print("RETRIEVAL ABLATION -- all queries")
    print("=" * 84)
    header = f"{'mode':<10}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'hit@5':>8}{'nDCG@10':>10}{'MRR':>8}{'ms/q':>9}"
    print(header)
    print("-" * len(header))
    summary = {}
    for mode in modes:
        m = evaluate(all_results[mode], qrels)
        summary[mode] = m
        print(f"{mode:<10}{m['recall@1']:>8.3f}{m['recall@5']:>8.3f}"
              f"{m['recall@10']:>8.3f}{m['hit@5']:>8.3f}{m['ndcg@10']:>10.3f}"
              f"{m['mrr']:>8.3f}{1000*timings[mode]/len(queries):>9.0f}")

    # ---- split by relevance mode (they measure different things) --------
    for kind in ("numeric", "qualitative"):
        subset = [q.query_id for q in queries if q.kind == kind]
        if not subset:
            continue
        label = ("value-anchored (strict)" if kind == "numeric"
                 else "section-anchored (proxy)")
        print(f"\n{kind.upper()} queries only -- {label}  (n={len(subset)})")
        print(f"{'mode':<10}{'R@5':>8}{'hit@5':>8}{'MRR':>8}")
        print("-" * 34)
        for mode in modes:
            sub_res = {q: all_results[mode][q] for q in subset}
            sub_rel = {q: qrels[q] for q in subset}
            m = evaluate(sub_res, sub_rel)
            print(f"{mode:<10}{m['recall@5']:>8.3f}{m['hit@5']:>8.3f}{m['mrr']:>8.3f}")

    # ---- uncertainty ----------------------------------------------------
    # Tested per query FAMILY as well as pooled. Reranking helps numeric
    # queries and hurts qualitative ones by a similar magnitude, so the two
    # effects cancel in the aggregate and the pooled test reports "no
    # difference" for a system whose behaviour changed substantially. Pooling
    # over heterogeneous subgroups can reverse or erase a real effect --
    # always test the strata you believe differ.
    if not args.no_significance and len(modes) > 1:
        base = args.baseline if args.baseline in modes else modes[0]

        families = [("ALL", [q.query_id for q in queries])]
        for kind in ("numeric", "qualitative"):
            ids = [q.query_id for q in queries if q.kind == kind]
            if ids:
                families.append((kind.upper(), ids))

        for fam_name, ids in families:
            idset = set(ids)
            print("\n" + "=" * 84)
            print(f"SIGNIFICANCE [{fam_name}, n={len(ids)}] -- paired bootstrap "
                  f"vs '{base}'")
            print("=" * 84)
            sub_qrels = {q: qrels[q] for q in ids if qrels.get(q)}
            for metric, k in (("mrr", 5), ("hit", 5)):
                base_res = {q: all_results[base][q] for q in idset
                            if q in all_results[base]}
                base_scores = per_query_scores(base_res, sub_qrels, metric, k)
                if not base_scores:
                    continue
                m, lo, hi = bootstrap_ci(base_scores)
                label = "MRR" if metric == "mrr" else f"hit@{k}"
                print(f"\n{label}")
                print(f"  {base:<10} {m:.3f}  [{lo:.3f}, {hi:.3f}]   (baseline)")
                for mode in modes:
                    if mode == base:
                        continue
                    res = {q: all_results[mode][q] for q in idset
                           if q in all_results[mode]}
                    s = per_query_scores(res, sub_qrels, metric, k)
                    mm, mlo, mhi = bootstrap_ci(s)
                    tst = paired_bootstrap_test(s, base_scores)
                    sig = tst["lo"] > 0 or tst["hi"] < 0
                    verdict = ("SIGNIFICANT" if sig else "not distinguishable")
                    arrow = "" if not sig else (" better" if tst["diff"] > 0 else " worse")
                    print(f"  {mode:<10} {mm:.3f}  [{mlo:.3f}, {mhi:.3f}]   "
                          f"diff {tst['diff']:+.3f} "
                          f"[{tst['lo']:+.3f}, {tst['hi']:+.3f}] "
                          f"p={tst['p']:.3f}  {verdict}{arrow}")

        print("""
  A difference whose CI spans zero is not evidence of a better system.
  Note especially where a per-family result is significant but the pooled
  one is not: opposite-signed effects in different strata cancel when
  averaged, so the pooled table can hide a real behavioural change.""")

    out = settings.data_dir / "retrieval_eval.json"
    out.write_text(json.dumps(
        {"model": embedder.model_name, "k": args.k, "n_queries": len(queries),
         "timings_s": timings, "metrics": summary}, indent=2))
    print(f"\nSaved -> {out}")
    print("""
HOW TO READ THIS
  - Recall@k is capped by k / |relevant|. Check the ceiling printed above
    before quoting recall as though 1.0 were reachable.
  - hit@k and MRR are the honest headline metrics for this label design.
  - sparse tends to win on NUMERIC queries -- not because figures appear in
    questions (they do not), but because the context prefix gives BM25 exact
    high-IDF tokens: the ticker, the fiscal year, the item number.
  - dense tends to win on hit@k for QUALITATIVE queries, where paraphrase
    matters and no exact token is available.
  - hybrid should lead on MRR and nDCG. If it does not, report that.
  - Run WITH and WITHOUT --use-filters on the SAME eval set. Comparing runs
    built from different eval sets is not an ablation.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
