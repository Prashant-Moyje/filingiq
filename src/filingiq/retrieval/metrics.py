"""Retrieval metrics.

Deliberately dependency-free and unit-tested, because these numbers go in
EVALUATION.md and on your resume. A subtly wrong nDCG that flatters your system
is worse than no metric at all -- and an interviewer who asks you to define it
will find out.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k.

    Note this is recall, not hit-rate: with 4 relevant chunks and 2 found,
    this returns 0.5 even though the user got a usable answer. Report hit_at_k
    alongside it when the practical question is "did we find anything useful".
    """
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if ANY relevant item is in the top k."""
    if not relevant:
        return float("nan")
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG.

    DCG uses log2(rank + 1), so rank 1 has discount log2(2) = 1. Ideal DCG is
    computed over min(len(relevant), k) items -- forgetting the min() inflates
    the ideal denominator and silently deflates every score.
    """
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(results: dict[str, list[str]], qrels: dict[str, set[str]],
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Aggregate metrics over a query set. Queries with no labels are skipped."""
    valid = [q for q in results if qrels.get(q)]
    if not valid:
        return {}
    out: dict[str, float] = {"n_queries": len(valid)}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(results[q], qrels[q], k) for q in valid) / len(valid)
        out[f"hit@{k}"] = sum(hit_at_k(results[q], qrels[q], k) for q in valid) / len(valid)
        out[f"ndcg@{k}"] = sum(ndcg_at_k(results[q], qrels[q], k) for q in valid) / len(valid)
    out["mrr"] = sum(reciprocal_rank(results[q], qrels[q]) for q in valid) / len(valid)
    return out


# ---------------------------------------------------------------------------
# Uncertainty
#
# Retrieval evaluations are routinely reported as bare point estimates, and on
# small query sets that is close to meaningless: on 182 queries hybrid beat
# dense on MRR (0.678 vs 0.648); on a 60-query subsample of the SAME system the
# ordering reversed (0.617 vs 0.663). Nothing changed except sample size.
#
# Every metric here is a mean over queries, so it has a sampling distribution
# and deserves an interval. The bootstrap is the right tool: no normality
# assumption, works for MRR and nDCG whose distributions are nowhere near
# Gaussian (MRR is spiky at 1, 0.5, 0.33, 0).
#
# Comparisons use the PAIRED bootstrap -- resampling queries, not systems --
# because both systems answer the same queries. Query difficulty is the
# dominant variance component, and pairing removes it.
# ---------------------------------------------------------------------------

def per_query_scores(results: dict[str, list[str]], qrels: dict[str, set[str]],
                     metric: str = "mrr", k: int = 5) -> dict[str, float]:
    """Per-query scores, needed for bootstrapping."""
    fns = {
        "mrr": lambda r, rel: reciprocal_rank(r, rel),
        "hit": lambda r, rel: hit_at_k(r, rel, k),
        "recall": lambda r, rel: recall_at_k(r, rel, k),
        "ndcg": lambda r, rel: ndcg_at_k(r, rel, k),
    }
    fn = fns[metric]
    return {q: fn(results[q], qrels[q]) for q in results if qrels.get(q)}


def bootstrap_ci(scores: dict[str, float], n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper) percentile bootstrap CI."""
    import random

    vals = [v for v in scores.values() if not math.isnan(v)]
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_resamples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return sum(vals) / n, lo, hi


def paired_bootstrap_test(a: dict[str, float], b: dict[str, float],
                          n_resamples: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap on the per-query difference a - b.

    Returns the mean difference, its CI, and a two-sided p-value. If the CI
    spans zero, the two systems are not distinguishable on this query set --
    which is the honest conclusion for most small-sample retrieval comparisons.
    """
    import random

    shared = [q for q in a if q in b
              and not math.isnan(a[q]) and not math.isnan(b[q])]
    if not shared:
        return {"n": 0, "diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}

    diffs = [a[q] - b[q] for q in shared]
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    boot = []
    for _ in range(n_resamples):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    lo, hi = boot[int(0.025 * n_resamples)], boot[int(0.975 * n_resamples) - 1]

    # Two-sided p: proportion of centred resamples at least as extreme.
    centred = [x - observed for x in boot]
    extreme = sum(1 for x in centred if abs(x) >= abs(observed))
    p = extreme / n_resamples

    return {"n": n, "diff": observed, "lo": lo, "hi": hi, "p": p}
