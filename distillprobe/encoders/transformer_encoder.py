from __future__ import annotations

import numpy as np

from .base import Encoder, l2_normalize


class TransformerEncoder(Encoder):
    """Encoder backed by a sentence-transformers SentenceTransformer.

    Uses the model's default pooling (typically mean for LaBSE / multilingual-e5)
    and emits L2-normalized vectors so cosine similarity is `dot`.
    """

    def __init__(self, model_id: str = "sentence-transformers/LaBSE", pool: str = "mean"):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.pool = pool
        self._model = SentenceTransformer(model_id)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return f"transformer:{self.model_id}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        v = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0]
        return np.asarray(v, dtype=np.float32)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64,
        )
        return np.asarray(v, dtype=np.float32)

    def tokenize(self, text: str) -> list[str]:
        return list(self._model.tokenizer.tokenize(text))
