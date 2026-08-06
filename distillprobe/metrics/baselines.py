from __future__ import annotations

import numpy as np

from .consistency import pairwise_consistency


def random_pair_baseline(
    word_pool: list[str],
    encode_fn,
    n_pairs: int,
    n_seeds: int = 5,
    base_seed: int = 42,
) -> dict[int, float]:
    """Per-seed difference-vector consistency over random word pairs.

    `encode_fn(word) -> np.ndarray` should return an L2-normalized vector.
    Returns a dict mapping `seed -> consistency`. Distribution stats can be
    computed by the caller (mean, CI).
    """
    if n_pairs < 2:
        raise ValueError("n_pairs must be >= 2 for a meaningful baseline")
    pool = list(dict.fromkeys(word_pool))
    if len(pool) < 4:
        raise ValueError(f"word_pool too small (need >=4 unique items, got {len(pool)})")

    cache: dict[str, np.ndarray] = {}

    def _vec(w: str) -> np.ndarray:
        v = cache.get(w)
        if v is None:
            v = np.asarray(encode_fn(w))
            cache[w] = v
        return v

    out: dict[int, float] = {}
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        diffs = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(pool), size=2, replace=False)
            a, b = _vec(pool[i]), _vec(pool[j])
            diffs.append(b - a)
        out[base_seed + s] = pairwise_consistency(np.stack(diffs, axis=0))
    return out
