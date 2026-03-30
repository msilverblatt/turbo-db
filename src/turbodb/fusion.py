"""Score fusion methods for hybrid search.

All functions accept two ranked result lists (vector and BM25) as
``list[tuple[key, score]]`` sorted by descending score, and return a
fused list of ``(key, fused_score)`` truncated to *k*.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["fuse_dbsf", "fuse_rrf", "fuse_weighted"]

ResultList = list[tuple[Any, float]]


# ------------------------------------------------------------------
# Reciprocal Rank Fusion (Cormack et al.)
# ------------------------------------------------------------------

def fuse_rrf(
    vector_results: ResultList,
    bm25_results: ResultList,
    k: int,
    rrf_k: int = 60,
) -> ResultList:
    """Reciprocal Rank Fusion.

    ``RRF(d) = 1/(rrf_k + rank_vector(d)) + 1/(rrf_k + rank_bm25(d))``

    Score-agnostic — only uses rank ordering, not raw scores.
    *rrf_k* defaults to 60 per the original paper.
    """
    scores: dict[Any, float] = {}

    for rank, (key, _) in enumerate(vector_results, start=1):
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (key, _) in enumerate(bm25_results, start=1):
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ------------------------------------------------------------------
# Weighted Linear Combination (Convex Combination)
# ------------------------------------------------------------------

def _min_max_normalize(results: ResultList) -> ResultList:
    """Normalize scores to [0, 1] via min-max."""
    if not results:
        return results
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(key, 1.0) for key, _ in results]
    span = hi - lo
    return [(key, (s - lo) / span) for key, s in results]


def fuse_weighted(
    vector_results: ResultList,
    bm25_results: ResultList,
    k: int,
    alpha: float = 0.5,
) -> ResultList:
    """Weighted linear combination with min-max normalization.

    ``score(d) = alpha * norm_vector(d) + (1 - alpha) * norm_bm25(d)``

    *alpha* = 1.0 → pure vector; *alpha* = 0.0 → pure BM25.
    Per Kuzi et al. (ACM TOIS 2023), convex combination outperforms RRF
    when *alpha* is tuned.
    """
    norm_vector = dict(_min_max_normalize(vector_results))
    norm_bm25 = dict(_min_max_normalize(bm25_results))

    all_keys = set(norm_vector) | set(norm_bm25)
    scores: dict[Any, float] = {}
    for key in all_keys:
        scores[key] = alpha * norm_vector.get(key, 0.0) + (1 - alpha) * norm_bm25.get(key, 0.0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ------------------------------------------------------------------
# Distribution-Based Score Fusion (Qdrant-style)
# ------------------------------------------------------------------

def _dbsf_normalize(results: ResultList) -> ResultList:
    """Normalize using mean ± 3·stddev as bounds, clamped to [0, 1]."""
    if not results:
        return results
    scores = [s for _, s in results]
    if len(scores) == 1:
        return [(key, 1.0) for key, _ in results]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    stddev = math.sqrt(variance)
    if stddev == 0:
        return [(key, 1.0) for key, _ in results]
    lower = mean - 3 * stddev
    upper = mean + 3 * stddev
    span = upper - lower
    return [(key, max(0.0, min(1.0, (s - lower) / span))) for key, s in results]


def fuse_dbsf(
    vector_results: ResultList,
    bm25_results: ResultList,
    k: int,
) -> ResultList:
    """Distribution-Based Score Fusion.

    Normalizes each retriever's scores using ``mean ± 3·stddev`` as
    bounds (mapped to [0, 1]), then sums.  More robust than min-max
    against score outliers.
    """
    norm_vector = dict(_dbsf_normalize(vector_results))
    norm_bm25 = dict(_dbsf_normalize(bm25_results))

    all_keys = set(norm_vector) | set(norm_bm25)
    scores: dict[Any, float] = {}
    for key in all_keys:
        scores[key] = norm_vector.get(key, 0.0) + norm_bm25.get(key, 0.0)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
