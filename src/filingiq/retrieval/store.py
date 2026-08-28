"""Vector and lexical indexes.

QDRANT IN EMBEDDED MODE
-----------------------
`QdrantClient(path=...)` runs Qdrant as a local library with no Docker and no
server -- the whole index is a folder in `data/`. That keeps the project
runnable with one command on any machine, which matters when someone clones
your repo. Switching to a real server later is a one-line URL change.

WHY BM25 AS WELL
----------------
Dense embeddings are strong on paraphrase and weak on rare exact tokens. In
filings that weakness is acute: ticker symbols, "Item 7A", XBRL tags, and
specific figures like "391,035" are precisely the terms a user searches for and
precisely the ones a 384-dimensional vector blurs away. BM25 nails them.
Neither alone is sufficient, which is the case for hybrid retrieval.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

COLLECTION = "filings"


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, path: Path | str, dim: int, collection: str = COLLECTION):
        from qdrant_client import QdrantClient

        self.path = str(path)
        self.dim = dim
        self.collection = collection
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.path)

    def recreate(self) -> None:
        """Drop and rebuild the collection.

        `recreate_collection` was deprecated and then removed in qdrant-client
        1.19. Doing it explicitly works on every version and is clearer anyway.
        """
        from qdrant_client.models import Distance, VectorParams

        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
        except AttributeError:      # very old clients lack collection_exists
            try:
                self.client.delete_collection(self.collection)
            except Exception:       # noqa: BLE001 -- absent is fine
                pass
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def upsert(self, ids: list[int], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(id=i, vector=v.tolist(), payload=p)
                for i, v, p in zip(ids, vectors, payloads)
            ],
        )

    def search(self, vector: np.ndarray, k: int = 10,
               filters: dict | None = None) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qfilter = None
        if filters:
            qfilter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=val))
                for key, val in filters.items()
            ])
        # qdrant-client 1.19 removed `search()` in favour of `query_points()`.
        # Supporting both keeps the repo runnable for anyone who clones it with
        # a different pinned version -- a small kindness that costs six lines.
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection, query=vector.tolist(),
                limit=k, query_filter=qfilter, with_payload=True,
            ).points
        else:
            res = self.client.search(
                collection_name=self.collection, query_vector=vector.tolist(),
                limit=k, query_filter=qfilter, with_payload=True,
            )
        return [Hit(chunk_id=r.payload["chunk_id"], score=float(r.score),
                    payload=r.payload) for r in res]

    def count(self) -> int:
        return self.client.count(self.collection).count


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\.,%$-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that PRESERVES numeric formatting.

    A generic tokenizer splits "391,035" into "391" and "035", destroying the
    exact-match signal that is BM25's entire advantage here. Commas, periods,
    percent and dollar signs stay attached.
    """
    return _TOKEN_RE.findall(text.lower())


class LexicalStore:
    """BM25 index over the same chunks. Pure Python, no wheels to worry about."""

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.payloads: list[dict] = []
        self._bm25 = None

    def build(self, chunk_ids: list[str], texts: list[str],
              payloads: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        self.payloads = payloads
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 10,
               filters: dict | None = None) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("LexicalStore.build() has not been called")
        scores = self._bm25.get_scores(tokenize(query))

        idx = np.argsort(scores)[::-1]
        out: list[Hit] = []
        for i in idx:
            if scores[i] <= 0:
                break
            payload = self.payloads[i]
            if filters and any(payload.get(kk) != vv for kk, vv in filters.items()):
                continue
            out.append(Hit(chunk_id=self.chunk_ids[i], score=float(scores[i]),
                           payload=payload))
            if len(out) >= k:
                break
        return out

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"chunk_ids": self.chunk_ids, "payloads": self.payloads,
                         "bm25": self._bm25}, fh)

    @classmethod
    def load(cls, path: Path | str) -> "LexicalStore":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        store = cls()
        store.chunk_ids = data["chunk_ids"]
        store.payloads = data["payloads"]
        store._bm25 = data["bm25"]
        return store
