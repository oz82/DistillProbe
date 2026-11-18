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
      - `role` optionally declares how a probe should treat the encoder
        ("student" | "reference" | "baseline"); set from the run config. When it is
        None, modules fall back to the historical name heuristics below.
    """

    role: str | None = None

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


_REFERENCE_NAME_MARKERS = ("transformer", "labse", "e5")


def is_reference_encoder(enc: Encoder) -> bool:
    """True for the contextual reference/teacher side of a probe.

    An explicit `role` wins; otherwise the encoder name is matched against the
    markers the canonical configs relied on (``transformer:``, LaBSE, mE5).
    """
    if enc.role is not None:
        return enc.role == "reference"
    lname = enc.name.lower()
    return any(m in lname for m in _REFERENCE_NAME_MARKERS)


def is_student_encoder(enc: Encoder) -> bool:
    """True for a distilled/static student whose retention is being measured.

    An explicit `role` wins; otherwise only Model2Vec encoders count (fastText is a
    baseline, not a student, in the canonical configs).
    """
    if enc.role is not None:
        return enc.role == "student"
    return "model2vec" in enc.name.lower()


def first_reference(encoders: dict[str, Encoder]) -> str | None:
    return next((n for n, e in encoders.items() if is_reference_encoder(e)), None)


def first_student(encoders: dict[str, Encoder]) -> str | None:
    return next((n for n, e in encoders.items() if is_student_encoder(e)), None)
