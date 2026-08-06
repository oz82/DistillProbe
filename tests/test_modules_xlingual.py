"""CSLS and hubness diagnostics for the cross-lingual retrieval probe."""

from __future__ import annotations

import numpy as np

from distillprobe.modules.xlingual import _csls_matrix, _retrieval_metrics


def _l2(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def test_perfect_alignment_p1_and_csls_are_one():
    rng = np.random.default_rng(0)
    v = _l2(rng.normal(size=(30, 16)).astype(np.float32))
    m = _retrieval_metrics(v, v)  # identical -> perfect retrieval
    assert m["p1"] == 1.0
    assert m["p1_csls"] == 1.0
    assert "hubness_skew" in m
    assert m["margin"] > 0.5


def test_csls_matrix_shape():
    rng = np.random.default_rng(1)
    v1 = _l2(rng.normal(size=(20, 8)).astype(np.float32))
    v2 = _l2(v1 + 0.05 * rng.normal(size=v1.shape).astype(np.float32))
    sim = v1 @ v2.T
    csls = _csls_matrix(sim, k=5)
    assert csls.shape == sim.shape
    # diagonal should remain the argmax for most near-parallel rows
    diag = np.arange(20)
    assert (csls.argmax(axis=1) == diag).mean() > 0.8


def test_hubness_skew_increases_with_a_hub():
    # Construct a target set where one column is a hub (close to everything).
    rng = np.random.default_rng(2)
    n, d = 25, 12
    v1 = _l2(rng.normal(size=(n, d)).astype(np.float32))
    v2 = v1.copy()
    # make target 0 a hub by pulling all sources slightly toward it
    hub = v2[0].copy()
    v1_hub = _l2(v1 + 0.9 * hub)
    m_clean = _retrieval_metrics(v1, v2)
    m_hub = _retrieval_metrics(v1_hub, v2)
    assert m_hub["hubness_skew"] >= m_clean["hubness_skew"]
