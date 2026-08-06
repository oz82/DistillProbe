from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return v / ||v||_2; if ||v|| < eps, return v unchanged."""
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v.astype(np.float32, copy=False)
    return (v / norm).astype(np.float32, copy=False)


class Encoder(ABC):
    """Abstract base class for all encoders.

    Contract:
      - `name` is a stable identifier used in logs and run manifests
      - `dim` is the output dimension; constant for the lifetime of the instance
      - `encode(text)` returns a 1-D np.ndarray of shape (dim,), L2-normalized
      - `tokenize(text)` returns a list of sub-tokens / subwords for audit purposes
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[str]: ...

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Default batched encode. Concrete encoders may override for speed."""
        return np.stack([self.encode(t) for t in texts], axis=0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"
