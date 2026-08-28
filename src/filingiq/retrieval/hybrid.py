"""Hybrid retrieval: RRF fusion plus optional cross-encoder reranking.

RECIPROCAL RANK FUSION
----------------------
Dense scores are cosine similarities in [0,1]; BM25 scores are unbounded and
corpus-dependent. They cannot be added, averaged, or compared -- a favourite
interview trap. Normalising them (min-max, z-score) is possible but fragile:
the scale shifts with every query and every corpus change.

RRF sidesteps the problem by discarding scores entirely and fusing RANKS:

    score(d) = sum over retrievers of  1 / (k + rank(d))

with k = 60 from Cormack et al. (2009). k dampens the top ranks so a single
retriever cannot dominate on its own confidence. It requires no tuning, no
normalisation, and no assumptions about score distributions -- which is why it
is the default in production systems.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from filingiq.retrieval.store import Hit

log = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[list[Hit]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    weights = list(weights) if weights else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranking, w in zip(rankings, weights):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + w / (k + rank)
            payloads.setdefault(hit.chunk_id, hit.payload)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [Hit(chunk_id=cid, score=s, payload=payloads[cid]) for cid, s in fused]


# Reranker options, cheapest first. On CPU the size difference is the whole
# story: bge-reranker-v2-m3 is 568M parameters (2.3 GB) and takes ~0.5 s per
# query-passage pair, so 60 queries x 50 candidates is a 25-minute run. MiniLM
# is 22M parameters and roughly 50x faster at a few points lower accuracy --
# the right default for iterating, with the large model reserved for a final
# reported number.
RERANKER_MODELS = {
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",     # 22M, ~90 MB
    "base": "BAAI/bge-reranker-base",                    # 278M, ~1.1 GB
    "large": "BAAI/bge-reranker-v2-m3",                  # 568M, ~2.3 GB
}


@dataclass
class Reranker:
    """Cross-encoder reranking.

    A bi-encoder embeds query and passage independently, so it never sees them
    together. A cross-encoder scores the PAIR -- far more accurate and far more
    expensive. Hence the standard pattern: retrieve N cheaply, rerank to k
    expensively. Usually the largest accuracy gain per unit of effort in a RAG
    pipeline, and also the largest latency cost.
    """
    model_name: str = RERANKER_MODELS["fast"]
    batch_size: int = 32
    _model: object | None = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int = 8) -> list[Hit]:
        if not hits:
            return []
        model = self._load()
        pairs = [(query, h.payload.get("text", "")) for h in hits]
        scores = model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=False)
        order = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
        return [Hit(chunk_id=h.chunk_id, score=float(s), payload=h.payload)
                for h, s in order[:top_k]]


class HybridRetriever:
    def __init__(self, vector_store, lexical_store, embedder,
                 reranker: Reranker | None = None):
        self.vs = vector_store
        self.ls = lexical_store
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, k: int = 10, candidates: int = 50,
               filters: dict | None = None, mode: str = "hybrid",
               rerank_candidates: int | None = None) -> list[Hit]:
        """mode: 'dense' | 'sparse' | 'hybrid' | 'rerank'.

        The modes exist so the evaluation harness can ablate them. Reporting
        'hybrid+rerank beats dense-only by N points' requires being able to run
        dense-only, and most implementations cannot.
        """
        # Validate before doing any work. Discovering a missing reranker only
        # after embedding and two retrievals wastes the expensive part of the
        # call and reports the error from the wrong place.
        if mode == "rerank" and self.reranker is None:
            raise ValueError("mode='rerank' requires a reranker")
        if mode not in {"dense", "sparse", "hybrid", "rerank"}:
            raise ValueError(f"unknown mode {mode!r}")

        if mode == "dense":
            qv = self.embedder.embed_queries([query])[0]
            return self.vs.search(qv, k=k, filters=filters)
        if mode == "sparse":
            return self.ls.search(query, k=k, filters=filters)

        qv = self.embedder.embed_queries([query])[0]
        dense = self.vs.search(qv, k=candidates, filters=filters)
        sparse = self.ls.search(query, k=candidates, filters=filters)
        fused = reciprocal_rank_fusion([dense, sparse])

        if mode == "rerank":
            # Reranking cost is linear in candidates. Going 50 -> 25 halves the
            # runtime; the recall lost is small because RRF already put the
            # good documents near the top.
            n = rerank_candidates or candidates
            return self.reranker.rerank(query, fused[:n], top_k=k)
        return fused[:k]
