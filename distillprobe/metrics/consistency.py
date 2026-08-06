from __future__ import annotations

import numpy as np


def difference_vectors(stems: np.ndarray, forms: np.ndarray) -> np.ndarray:
    """Per-pair difference vectors `forms - stems`.

    Inputs are (N, D) arrays; output is (N, D).
    """
    stems = np.asarray(stems)
    forms = np.asarray(forms)
    if stems.shape != forms.shape:
        raise ValueError(f"shape mismatch: {stems.shape} vs {forms.shape}")
    return forms - stems


def pairwise_consistency(diff_vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity over all unique (i, j) pairs of difference vectors.

    Returns NaN when fewer than 2 difference vectors are provided.
    """
    diffs = np.asarray(diff_vectors, dtype=np.float64)
    n = diffs.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normed = diffs / norms
    sim = normed @ normed.T
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())
