#!/usr/bin/env python
"""Week 3b: build the retrieval evaluation set from XBRL ground truth."""
from __future__ import annotations

import argparse, logging, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings                 # noqa: E402
from filingiq.retrieval import evalset               # noqa: E402
from filingiq.storage import db                      # noqa: E402


def _quiet_libs() -> None:
    """Silence per-request HTTP logging from huggingface_hub and httpx."""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-per-filing", type=int, default=4)
    args = ap.parse_args()
    _quiet_libs()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out) if args.out else settings.data_dir / "evalset.json"

    with db.connect(read_only=True) as con:
        if con.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
            print("No chunks. Run scripts/06_chunk.py first.")
            return 1
        numeric = evalset.build_numeric_queries(con, args.max_per_filing)
        qualitative = evalset.build_qualitative_queries(con)

    queries = numeric + qualitative
    evalset.save(queries, out)

    print("=" * 66)
    print(f"EVAL SET: {len(queries)} queries -> {out}")
    print("=" * 66)
    print(f"  value-anchored (strict, numeric)     : {len(numeric)}")
    print(f"  section-anchored (proxy, qualitative): {len(qualitative)}")
    print("\n  Relevant chunks per query:")
    for kind, qs in (("numeric", numeric), ("qualitative", qualitative)):
        if not qs:
            continue
        counts = [len(q.relevant_chunks) for q in qs]
        counts.sort()
        print(f"    {kind:<12} min={counts[0]}  median={counts[len(counts)//2]}  "
              f"max={counts[-1]}")
    # FM-008: a label set this large makes hit@k near-certain and caps
    # recall@k at k/|relevant|, so the query measures its labels rather than
    # the retriever. The threshold existed as a constant for weeks and was
    # read by nothing; this is the report it was always described as feeding.
    oversized = evalset.oversized_label_sets(queries)
    if oversized:
        print(f"\n  WARNING -- {len(oversized)} queries exceed "
              f"{evalset.LABEL_SET_WARN_THRESHOLD} relevant chunks:")
        for qid, n in oversized[:10]:
            print(f"    {qid:<44} {n} labels  (recall@5 capped at "
                  f"{5 / n:.3f})")
        if len(oversized) > 10:
            print(f"    ... and {len(oversized) - 10} more")
        print("  These flatter hit@k. Report hit@k and MRR, not recall@k, "
              "or tighten the label rule.")

    print("\n  By company:", dict(Counter(q.ticker for q in queries)))
    print("""
  SANITY CHECK -- read a few before trusting the numbers:
    python -c "import json;d=json.load(open(r'%s'));[print(q['question'],'->',len(q['relevant_chunks']),'chunks') for q in d[:8]]"

  A numeric query with 40+ relevant chunks usually means the figure is a
  common one (a round number appearing in many tables). Those queries are
  easy and will flatter your recall -- worth noting in EVALUATION.md.
""" % out)
    print("Next: python scripts/09_eval_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
