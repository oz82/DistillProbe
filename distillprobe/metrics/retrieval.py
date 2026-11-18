"""Ranked-retrieval metrics on a dense (queries x docs) score matrix."""

from __future__ import annotations

import numpy as np


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (ties by lower index)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query nDCG@k for graded relevance.

    `scores` is (Q, D) system scores, `gains` is (Q, D) gold relevance grades
    (0 = not relevant). Queries without any relevant document get nDCG 0.
    """
    top = _top_k_indices(scores, k)
    discounts = 1.0 / np.log2(np.arange(2, top.shape[1] + 2))
    dcg = (np.take_along_axis(gains, top, axis=1) * discounts).sum(axis=1)
    ideal_sorted = -np.sort(-gains, axis=1)[:, : top.shape[1]]
    idcg = (ideal_sorted * discounts).sum(axis=1)
    out = np.zeros(scores.shape[0], dtype=np.float64)
    ok = idcg > 0
    out[ok] = dcg[ok] / idcg[ok]
    return out


def recall_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query fraction of relevant documents (gain > 0) retrieved in the top k."""
    top = _top_k_indices(scores, k)
    relevant = gains > 0
    hits = np.take_along_axis(relevant, top, axis=1).sum(axis=1)
    total = relevant.sum(axis=1)
    out = np.zeros(scores.shape[0], dtype=np.float64)
    ok = total > 0
    out[ok] = hits[ok] / total[ok]
    return out


def mrr_at_k(scores: np.ndarray, gains: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query reciprocal rank of the first relevant document within the top k."""
    top = _top_k_indices(scores, k)
    hit = np.take_along_axis(gains > 0, top, axis=1)
    out = np.zeros(scores.shape[0], dtype=np.float64)
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    out[any_hit] = 1.0 / (first[any_hit] + 1)
    return out
