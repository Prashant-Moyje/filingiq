"""Embedding models.

MODEL CHOICE
------------
Default is `BAAI/bge-small-en-v1.5`: 133 MB, 384 dimensions, fast enough to
embed 6,000 chunks on CPU in a few minutes. `bge-base-en-v1.5` (438 MB, 768d)
is a real accuracy step up; `bge-m3` (2.2 GB) better again but slow on CPU.
Run the eval harness across all three and report the accuracy/cost frontier --
that comparison is worth more in an interview than picking the biggest model
and asserting it is best.

QUERY / PASSAGE ASYMMETRY
-------------------------
BGE and E5 are trained with an instruction prefix on QUERIES but not passages.
Omitting it costs several points of recall, and the failure is silent -- you
still get results, just worse ones. One of the most common unforced errors in
RAG, so it is handled here rather than left for callers to remember.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
    "nomic": "search_query: ",
}
PASSAGE_PREFIXES = {"bge": "", "e5": "passage: ", "gte": "", "nomic": "search_document: "}


def _family(model_name: str) -> str:
    n = model_name.lower()
    for fam in ("bge", "e5", "gte", "nomic"):
        if fam in n:
            return fam
    return "gte"


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int = 32):
        from filingiq.config import settings
        self.model_name = model_name or settings.embedding_model
        self.family = _family(self.model_name)
        self.batch_size = batch_size
        self._model = None
        self._device = device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading %s (first run downloads weights)", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dim(self) -> int:
        # sentence-transformers 6.x renamed this method; support both.
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(self.model, name, None)
            if fn is not None:
                return int(fn())
        raise AttributeError("cannot determine embedding dimension")

    def embed_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        p = PASSAGE_PREFIXES.get(self.family, "")
        return self._encode([p + t for t in texts] if p else texts, show_progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        p = QUERY_PREFIXES.get(self.family, "")
        return self._encode([p + t for t in texts] if p else texts, False)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)
