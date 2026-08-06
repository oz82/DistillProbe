from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Encoder, l2_normalize


class FastTextEncoder(Encoder):
    """Encoder backed by Facebook fastText `.bin` (with subword n-grams)."""

    def __init__(self, path: str | Path):
        from gensim.models.fasttext import load_facebook_vectors

        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"fastText binary not found: {self.path}")
        self._kv = load_facebook_vectors(self.path)
        self._dim = int(self._kv.vector_size)

    @property
    def name(self) -> str:
        return f"fasttext:{Path(self.path).name}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        # Canonical fastText phrase/sentence vector: the mean of per-word
        # subword-composed vectors (whitespace tokenization). For a single word
        # this reduces exactly to that word's own subword vector, so single-word
        # probes (e.g. morphology) are unaffected; multi-word inputs (e.g. the
        # "X değil" negation phrase, polysemy definitions) get a proper bag-of-
        # words representation rather than one OOV vector for the whole string.
        # OOV tokens are handled via fastText character n-grams.
        tokens = text.split()
        if not tokens:
            return l2_normalize(np.zeros(self._dim, dtype=np.float32))
        if len(tokens) == 1:
            v = np.asarray(self._kv.get_vector(tokens[0]), dtype=np.float32)
        else:
            vecs = [np.asarray(self._kv.get_vector(t), dtype=np.float32) for t in tokens]
            v = np.mean(np.stack(vecs, axis=0), axis=0)
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        """Return fastText-style character n-grams (full word + char n-grams in [min_n, max_n]).

        FastTextKeyedVectors does not expose `get_subwords` (that lives on the FullModel
        class), so we recompute the standard fastText character n-gram set: word is
        wrapped in '<>' and split into every n-gram of length min_n..max_n inclusive.
        """
        min_n = int(getattr(self._kv, "min_n", 3))
        max_n = int(getattr(self._kv, "max_n", 6))
        ext = f"<{text}>"
        out: list[str] = [ext]  # the full-word token itself
        for n in range(min_n, max_n + 1):
            if n > len(ext):
                break
            for i in range(0, len(ext) - n + 1):
                out.append(ext[i : i + n])
        return out
