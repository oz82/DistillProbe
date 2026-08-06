from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Inputs are normalized internally for safety."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def stem_form_similarity(stems: np.ndarray, forms: np.ndarray) -> np.ndarray:
    """Per-pair cosine similarity between (stem, form). Returns shape (N,)."""
    stems = np.asarray(stems, dtype=np.float64)
    forms = np.asarray(forms, dtype=np.float64)
    if stems.shape != forms.shape:
        raise ValueError(f"shape mismatch: {stems.shape} vs {forms.shape}")
    s_n = np.linalg.norm(stems, axis=1)
    f_n = np.linalg.norm(forms, axis=1)
    denom = s_n * f_n
    denom[denom < 1e-12] = 1.0
    return ((stems * forms).sum(axis=1) / denom).astype(np.float64)
