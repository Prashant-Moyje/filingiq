#!/usr/bin/env python
"""Week 3b: embed chunks and build the vector + BM25 indexes.

Usage:
    python scripts/07_embed.py
    python scripts/07_embed.py --model BAAI/bge-base-en-v1.5
    python scripts/07_embed.py --limit 500        # quick smoke test first
"""
from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filingiq.config import settings                       # noqa: E402
from filingiq.retrieval.embedder import Embedder           # noqa: E402
from filingiq.retrieval.store import LexicalStore, VectorStore  # noqa: E402
from filingiq.storage import db                            # noqa: E402


def _quiet_libs() -> None:
    """Silence per-request HTTP logging from huggingface_hub and httpx."""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    _quiet_libs()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    with db.connect(read_only=True) as con:
        sql = """SELECT chunk_id, accession, ticker, fiscal_year, item,
                        chunk_index, n_tokens, has_table, text, raw_text
                 FROM chunks ORDER BY ticker, fiscal_year, item, chunk_index"""
        if args.limit:
            sql += f" LIMIT {args.limit}"
        rows = con.execute(sql).fetchall()

    if not rows:
        print("No chunks. Run scripts/06_chunk.py first.")
        return 1

    print(f"Embedding {len(rows):,} chunks...")
    embedder = Embedder(model_name=args.model, batch_size=args.batch_size)
    texts = [r[8] for r in rows]        # context-prefixed text
    raws = [r[9] for r in rows]

    t0 = time.time()
    vectors = embedder.embed_passages(texts)
    elapsed = time.time() - t0
    print(f"Embedded in {elapsed:,.1f}s ({len(rows)/elapsed:,.1f}/s), "
          f"dim={vectors.shape[1]}, model={embedder.model_name}")

    payloads = [
        {"chunk_id": r[0], "accession": r[1], "ticker": r[2], "fiscal_year": r[3],
         "item": r[4], "chunk_index": r[5], "n_tokens": r[6], "has_table": r[7],
         "text": r[9]}
        for r in rows
    ]

    vs = VectorStore(settings.data_dir / "qdrant", dim=vectors.shape[1])
    vs.recreate()
    B = 256
    for i in range(0, len(rows), B):
        vs.upsert(list(range(i, min(i + B, len(rows)))),
                  vectors[i:i + B], payloads[i:i + B])
    print(f"Vector index: {vs.count():,} points")

    # BM25 indexes the CONTEXT-PREFIXED text, not the raw passage. The prefix
    # carries "AAPL | FY2024 | Item 7", and those are exactly the rare, exact
    # tokens a question supplies ("AAPL", "2024"). Indexing raw_text instead
    # made the ticker and year invisible to lexical search -- see FM-007.
    ls = LexicalStore()
    ls.build([r[0] for r in rows], texts, payloads)
    ls.save(settings.data_dir / "bm25.pkl")
    print(f"BM25 index: {len(ls.chunk_ids):,} documents")
    print("\nNext: python scripts/08_build_evalset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
