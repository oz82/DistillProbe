from __future__ import annotations

import string
from pathlib import Path

import numpy as np

from .base import Encoder, l2_normalize

# Punctuation skipped before averaging, as in the reference implementation.
_PUNCT = set(string.punctuation) | set("¡○¦–—”…’“。、？！「」（）：・，")


class StaticWordVecEncoder(Encoder):
    """Word-level static embeddings in word2vec text format, mean-pooled per sentence.

    Implements the inference recipe of Wada et al. (EMNLP 2025, "Static Word
    Embeddings for Sentence Semantic Representation"): words are split with a
    BERT-style pre-tokenizer, looked up case-sensitively then lower-cased, and an
    out-of-vocabulary word backs off to its longest subword prefix under the
    source Sentence Transformer's tokenizer. The sentence vector is the mean of
    the retained word vectors (punctuation dropped), L2-normalised.

    This is a non-Model2Vec static-distillation family (sentence-level PCA +
    knowledge distillation from a Sentence Transformer), used to test whether the
    loss spectrum generalises beyond Model2Vec.
    """

    def __init__(
        self,
        path: str | Path,
        word_tokenizer: str = "google-bert/bert-base-multilingual-cased",
        subword_tokenizer: str | None = "Alibaba-NLP/gte-base-en-v1.5",
    ):
        from transformers import AutoTokenizer

        self.path = str(path)
        self.word_tokenizer_id = word_tokenizer
        self.subword_tokenizer_id = subword_tokenizer
        self._word2vec, self._dim = self._load(Path(path))
        self._pre_tokenize = AutoTokenizer.from_pretrained(
            word_tokenizer
        ).backend_tokenizer.pre_tokenizer.pre_tokenize_str
        self._subword = (
            AutoTokenizer.from_pretrained(subword_tokenizer) if subword_tokenizer else None
        )

    @staticmethod
    def _load(path: Path) -> tuple[dict[str, np.ndarray], int]:
        word2vec: dict[str, np.ndarray] = {}
        dim: int | None = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").rstrip(" ").split(" ")
                if len(parts) == 2:  # optional "<vocab> <dim>" header
                    continue
                vec = np.asarray(parts[1:], dtype=np.float32)
                if dim is None:
                    dim = vec.size
                elif vec.size != dim:
                    continue
                word2vec[parts[0]] = vec
        if dim is None:
            raise ValueError(f"no vectors found in {path}")
        return word2vec, dim

    @property
    def name(self) -> str:
        return f"static_wordvec:{Path(self.path).name}"

    @property
    def dim(self) -> int:
        return self._dim

    def _lookup(self, word: str) -> np.ndarray | None:
        v = self._word2vec.get(word)
        if v is None:
            v = self._word2vec.get(word.lower())
        if v is not None or self._subword is None:
            return v
        subwords = self._subword.tokenize(word)
        while len(subwords) > 1:
            subwords = subwords[:-1]
            prefix = "".join(subwords).replace("##", "").replace("▁", "")
            v = self._word2vec.get(prefix)
            if v is None:
                v = self._word2vec.get(prefix.lower())
            if v is not None:
                return v
        return None

    def _words(self, text: str) -> list[str]:
        return [w for w, _ in self._pre_tokenize(text) if w not in _PUNCT]

    def encode(self, text: str) -> np.ndarray:
        vecs = [v for v in (self._lookup(w) for w in self._words(text)) if v is not None]
        if not vecs:
            return np.zeros(self._dim, dtype=np.float32)
        return l2_normalize(np.mean(vecs, axis=0).astype(np.float32))

    def tokenize(self, text: str) -> list[str]:
        return self._words(text)
