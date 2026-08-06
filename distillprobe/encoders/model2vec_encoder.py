from __future__ import annotations

import numpy as np

from .base import Encoder, l2_normalize


class Model2VecEncoder(Encoder):
    """Encoder backed by a Model2Vec StaticModel (HuggingFace hub or local)."""

    def __init__(self, model_id: str = "minishlab/potion-multilingual-128M"):
        from model2vec import StaticModel

        self.model_id = model_id
        self._model = StaticModel.from_pretrained(model_id)
        self._dim = int(self._model.dim)

    @property
    def name(self) -> str:
        return f"model2vec:{self.model_id}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        v = np.asarray(self._model.encode([text])[0], dtype=np.float32)
        return l2_normalize(v)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        v = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return (v / norms).astype(np.float32, copy=False)

    def tokenize(self, text: str) -> list[str]:
        enc = self._model.tokenizer.encode(text)
        # tokenizers.Encoding has .tokens attribute
        return list(enc.tokens)
