"""Retrieval tests that need no model download -- pure logic and metrics."""
from __future__ import annotations

import math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from filingiq.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from filingiq.retrieval.metrics import (  # noqa: E402
    hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank,
)
from filingiq.retrieval.store import Hit, tokenize  # noqa: E402
from filingiq.retrieval.evalset import value_variants  # noqa: E402


def H(cid, score=1.0):
    return Hit(chunk_id=cid, score=score, payload={"chunk_id": cid})


# --- metrics ---------------------------------------------------------------

def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["x"], {"a"}, 1) == 0.0


def test_recall_vs_hit_differ():
    """With 4 relevant and 1 found, recall is 0.25 but the user got an answer."""
    retrieved, relevant = ["a", "x", "y"], {"a", "b", "c", "d"}
    assert recall_at_k(retrieved, relevant, 3) == 0.25
    assert hit_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_rewards_higher_ranks():
    good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert good > bad


def test_ndcg_ideal_uses_min_of_relevant_and_k():
    """With 10 relevant items and k=2, finding 2 is a perfect result at k=2.
    Forgetting the min() would make this ~0.2 and understate every score."""
    relevant = {f"d{i}" for i in range(10)}
    assert ndcg_at_k(["d0", "d1"], relevant, 2) == pytest.approx(1.0)


def test_metrics_handle_empty_labels():
    assert math.isnan(recall_at_k(["a"], set(), 5))


# --- RRF -------------------------------------------------------------------

def test_rrf_ranks_consensus_above_single_retriever_favourite():
    """'b' is 2nd in both lists; 'a' is 1st in one and absent from the other.
    Agreement across retrievers is the signal RRF is designed to reward."""
    dense = [H("a"), H("b"), H("c")]
    sparse = [H("z"), H("b"), H("c")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk_id == "b"


def test_rrf_ignores_raw_score_magnitudes():
    """The whole point: BM25 scores are unbounded, cosine is [0,1]. Fusion
    must depend on RANK only."""
    a = [H("x", score=999.0), H("y", score=0.001)]
    b = [H("x", score=0.02), H("y", score=0.01)]
    fused = reciprocal_rank_fusion([a, b])
    assert [h.chunk_id for h in fused] == ["x", "y"]


def test_rrf_weights_shift_the_balance():
    dense = [H("d1"), H("d2")]
    sparse = [H("s1"), H("s2")]
    assert reciprocal_rank_fusion([dense, sparse], weights=[5.0, 1.0])[0].chunk_id == "d1"
    assert reciprocal_rank_fusion([dense, sparse], weights=[1.0, 5.0])[0].chunk_id == "s1"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[H("a")]], weights=[1.0, 2.0])


# --- tokenizer -------------------------------------------------------------

def test_tokenizer_preserves_financial_figures():
    """Splitting '391,035' into '391' and '035' destroys BM25's advantage."""
    assert "391,035" in tokenize("Net sales were 391,035 million")


def test_tokenizer_lowercases():
    assert "aapl" in tokenize("AAPL reported")


# --- eval set value variants ----------------------------------------------

def test_value_variants_cover_millions_and_billions():
    v = value_variants(391_035_000_000, "revenue")
    assert "391,035" in v      # as printed in a millions-scaled table
    assert "391.0" in v        # as printed in prose


def test_eps_variants_use_two_decimals():
    assert "6.13" in value_variants(6.13, "eps_diluted")


def test_value_variants_drop_tiny_ambiguous_numbers():
    """Two-digit strings would match everywhere and pollute the labels."""
    assert all(len(x.replace(",", "")) >= 3 for x in value_variants(1_234_000, "revenue"))


# --- client API compatibility ---------------------------------------------

def test_vector_store_roundtrip(tmp_path):
    """Guards the qdrant-client API surface we depend on. 1.19 removed
    `search()` and `recreate_collection()`; this catches the next such change
    at test time instead of mid-evaluation."""
    import numpy as np
    pytest.importorskip("qdrant_client")
    from filingiq.retrieval.store import VectorStore

    vs = VectorStore(tmp_path / "q", dim=8)
    vs.recreate()
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    payloads = [{"chunk_id": f"c{i}", "ticker": "AAPL" if i < 2 else "MSFT",
                 "text": f"chunk {i}"} for i in range(4)]
    vs.upsert(list(range(4)), vecs, payloads)

    assert vs.count() == 4
    hits = vs.search(vecs[0], k=2)
    assert hits[0].chunk_id == "c0"
    filtered = vs.search(vecs[0], k=4, filters={"ticker": "MSFT"})
    assert filtered and all(h.payload["ticker"] == "MSFT" for h in filtered)


def test_lexical_store_finds_exact_figures():
    pytest.importorskip("rank_bm25")
    from filingiq.retrieval.store import LexicalStore

    ls = LexicalStore()
    payloads = [{"chunk_id": f"c{i}"} for i in range(3)]
    ls.build(["c0", "c1", "c2"],
             ["Net sales were 391,035 million in fiscal 2024",
              "Gross margin percentage improved",
              "Risk factors relating to supply chain"], payloads)
    hits = ls.search("391,035", k=2)
    assert hits and hits[0].chunk_id == "c0"


def test_reranker_model_tiers_are_ordered_by_size():
    """Guards against silently defaulting to the 2.3 GB model, which turns a
    2-minute evaluation into a 25-minute one on CPU."""
    from filingiq.retrieval.hybrid import RERANKER_MODELS, Reranker
    assert set(RERANKER_MODELS) == {"fast", "base", "large"}
    assert Reranker().model_name == RERANKER_MODELS["fast"]


def test_rerank_mode_requires_a_reranker():
    from filingiq.retrieval.hybrid import HybridRetriever
    r = HybridRetriever(None, None, None, reranker=None)
    with pytest.raises(ValueError, match="requires a reranker"):
        r.search("q", mode="rerank")


# --- uncertainty quantification -------------------------------------------

def test_bootstrap_ci_brackets_the_mean():
    from filingiq.retrieval.metrics import bootstrap_ci
    scores = {f"q{i}": (i % 5) / 4 for i in range(80)}
    mean, lo, hi = bootstrap_ci(scores, n_resamples=500)
    assert lo <= mean <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_narrows_with_more_queries():
    """The whole point: small eval sets give wide intervals."""
    from filingiq.retrieval.metrics import bootstrap_ci
    small = {f"q{i}": (i % 2) for i in range(20)}
    large = {f"q{i}": (i % 2) for i in range(400)}
    _, lo_s, hi_s = bootstrap_ci(small, n_resamples=500)
    _, lo_l, hi_l = bootstrap_ci(large, n_resamples=500)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_paired_test_detects_a_real_difference():
    from filingiq.retrieval.metrics import paired_bootstrap_test
    a = {f"q{i}": 1.0 for i in range(60)}
    b = {f"q{i}": 0.0 for i in range(60)}
    r = paired_bootstrap_test(a, b, n_resamples=500)
    assert r["diff"] == pytest.approx(1.0)
    assert r["lo"] > 0


def test_paired_test_reports_noise_as_inconclusive():
    """Two systems differing only by coin-flip noise must NOT come back
    significant -- this is the guard against reporting a lucky sample."""
    import random
    from filingiq.retrieval.metrics import paired_bootstrap_test
    rng = random.Random(7)
    a = {f"q{i}": rng.random() for i in range(60)}
    b = {f"q{i}": rng.random() for i in range(60)}
    r = paired_bootstrap_test(a, b, n_resamples=1000)
    assert r["lo"] < 0 < r["hi"], "noise was reported as a significant difference"


def test_paired_test_handles_no_overlap():
    from filingiq.retrieval.metrics import paired_bootstrap_test
    r = paired_bootstrap_test({"a": 1.0}, {"b": 1.0}, n_resamples=100)
    assert r["n"] == 0
