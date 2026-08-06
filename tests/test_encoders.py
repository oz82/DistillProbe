"""Encoder contract tests.

These tests run against a deterministic stub encoder so they don't require any
network or model download. Concrete encoders (Model2Vec, fastText) inherit the
same contract and are exercised by `scripts/smoke_test.py`.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from distillprobe.encoders.base import Encoder, l2_normalize


class StubEncoder(Encoder):
    DIM = 16

    @property
    def name(self) -> str:
        return "stub"

    @property
    def dim(self) -> int:
        return self.DIM

    def encode(self, text: str) -> np.ndarray:
        # Deterministic pseudo-vector from text hash.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.frombuffer(h[: self.DIM], dtype=np.uint8).astype(np.float32) - 127.5
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        return list(text)


@pytest.fixture
def enc() -> Encoder:
    return StubEncoder()


def test_dim_is_constant(enc):
    assert isinstance(enc.dim, int)
    assert enc.dim == StubEncoder.DIM


def test_encode_shape(enc):
    v = enc.encode("ev")
    assert v.shape == (enc.dim,)
    assert v.dtype == np.float32


def test_encode_is_l2_normalized(enc):
    v = enc.encode("evler")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_encode_is_deterministic(enc):
    a = enc.encode("kitap")
    b = enc.encode("kitap")
    np.testing.assert_array_equal(a, b)


def test_encode_many_matches_encode(enc):
    words = ["a", "ab", "abc"]
    expected = np.stack([enc.encode(w) for w in words], axis=0)
    got = enc.encode_many(words)
    np.testing.assert_allclose(got, expected, atol=1e-7)


def test_tokenize_returns_list(enc):
    toks = enc.tokenize("ev")
    assert isinstance(toks, list)
    assert all(isinstance(t, str) for t in toks)


def test_l2_normalize_handles_zero():
    z = np.zeros(4, dtype=np.float32)
    out = l2_normalize(z)
    np.testing.assert_array_equal(out, z)


# --- FastTextEncoder phrase mean-pooling (no real model needed) ------------------


class _MockKV:
    vector_size = 4

    def get_vector(self, w):
        table = {
            "a": np.array([1.0, 0.0, 0.0, 0.0]),
            "b": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        return table[w]


def _fasttext_with_mock():
    from distillprobe.encoders.fasttext_encoder import FastTextEncoder

    enc = FastTextEncoder.__new__(FastTextEncoder)  # bypass __init__ (no .bin load)
    enc.path = "mock"
    enc._kv = _MockKV()
    enc._dim = 4
    return enc


def test_fasttext_single_word_is_word_vector():
    enc = _fasttext_with_mock()
    v = enc.encode("a")
    np.testing.assert_allclose(v, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_fasttext_multiword_is_normalized_mean():
    enc = _fasttext_with_mock()
    v = enc.encode("a b")  # mean([1,0,0,0],[0,1,0,0]) = [.5,.5,0,0] -> L2 norm
    np.testing.assert_allclose(v, [0.70710678, 0.70710678, 0.0, 0.0], atol=1e-6)


def test_fasttext_empty_is_zero():
    enc = _fasttext_with_mock()
    v = enc.encode("   ")
    np.testing.assert_array_equal(v, np.zeros(4, dtype=np.float32))
