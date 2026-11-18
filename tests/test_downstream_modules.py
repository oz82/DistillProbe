"""Downstream clustering / retrieval modules and the ranked-retrieval metrics.

Run against a stub encoder whose vectors are fixed per text, so no model is loaded.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from distillprobe.encoders.base import Encoder, l2_normalize
from distillprobe.metrics import mrr_at_k, ndcg_at_k, recall_at_k


class _TableEncoder(Encoder):
    """Returns a fixed vector per text; unknown texts get a deterministic hash vector."""

    def __init__(self, table: dict[str, list[float]], name: str, role: str | None = None):
        self._table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        self._name = name
        self.role = role

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return 3

    def encode(self, text: str) -> np.ndarray:
        v = self._table.get(text)
        if v is None:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.normal(size=3).astype(np.float32)
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        return text.split()


# --- metrics -----------------------------------------------------------------------


def test_ndcg_recall_mrr_on_small_example():
    scores = np.array([[0.9, 0.8, 0.1], [0.2, 0.9, 0.3]])
    gains = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    nd = ndcg_at_k(scores, gains, k=3)
    # q0: relevant doc ranked 2nd -> 1/log2(3); q1: relevant doc ranked 3rd -> 1/log2(4)
    np.testing.assert_allclose(nd, [1 / np.log2(3), 0.5], atol=1e-9)
    np.testing.assert_allclose(recall_at_k(scores, gains, k=1), [0.0, 0.0])
    np.testing.assert_allclose(recall_at_k(scores, gains, k=2), [1.0, 0.0])
    np.testing.assert_allclose(mrr_at_k(scores, gains, k=3), [0.5, 1 / 3])


def test_ndcg_handles_graded_gains_and_no_relevant():
    scores = np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]])
    gains = np.array([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    nd = ndcg_at_k(scores, gains, k=3)
    dcg = 2 / np.log2(2) + 1 / np.log2(4)
    idcg = 2 / np.log2(2) + 1 / np.log2(3)
    np.testing.assert_allclose(nd, [dcg / idcg, 0.0], atol=1e-9)


# --- modules -----------------------------------------------------------------------


def _cfg(**extra):
    return SimpleNamespace(seed=0, bootstrap_n=200, extra=extra)


def test_clustering_module_perfect_vs_random(tmp_path):
    from distillprobe.modules.downstream_clustering import DownstreamClusteringModule

    sents = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
    labels = ["A"] * 6 + ["B"] * 6
    data = {"language": "xx", "version": "t", "source": "t",
            "sets": [{"sentences": sents, "labels": labels}] * 3}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    good = _TableEncoder({s: ([1, 0, 0] if s.startswith("a") else [0, 1, 0]) for s in sents},
                         "transformer:ref", role="reference")
    student = _TableEncoder({s: ([1, 0.1, 0] if s.startswith("a") else [0, 1, 0.1]) for s in sents},
                            "model2vec:student")
    res = DownstreamClusteringModule().run({"student": student, "ref": good}, str(path), _cfg())
    m = res.metrics
    assert m["per_encoder"]["ref"]["v_measure_mean"] == 1.0
    assert m["per_encoder"]["student"]["v_measure_mean"] == 1.0
    assert m["retention"]["student_over_ref"]["v_measure_retention"] == 1.0
    assert m["primary_reference"] == "ref"
    assert len(res.per_category) == 3 and res.per_category[0]["k"] == 2


def test_retrieval_module_scores_only_judged_queries(tmp_path):
    from distillprobe.modules.downstream_retrieval import DownstreamRetrievalModule

    data = {
        "language": "xx", "version": "t", "source": "t",
        "corpus": [{"id": "d1", "title": "", "text": "one"}, {"id": "d2", "text": "two"},
                   {"id": "d3", "text": "three"}],
        "queries": [{"id": "q1", "text": "Q1"}, {"id": "q2", "text": "Q2"}, {"id": "q3", "text": "Q3"}],
        "qrels": {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}},
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    perfect = _TableEncoder({"one": [1, 0, 0], "two": [0, 1, 0], "three": [0, 0, 1],
                             "Q1": [1, 0, 0], "Q2": [0, 0, 1]}, "transformer:ref", role="reference")
    # student ranks the relevant doc second for q2
    student = _TableEncoder({"one": [1, 0, 0], "two": [0, 1, 0], "three": [0, 0, 1],
                             "Q1": [1, 0, 0], "Q2": [0, 0.9, 0.5]}, "model2vec:student")
    res = DownstreamRetrievalModule().run({"student": student, "ref": perfect}, str(path), _cfg(k=2))
    m = res.metrics
    assert m["n_queries_judged"] == 2
    assert m["per_encoder"]["ref"]["ndcg@2"] == 1.0
    expected_student = (1.0 + 1 / np.log2(3)) / 2
    assert abs(m["per_encoder"]["student"]["ndcg@2"] - expected_student) < 1e-9
    assert abs(m["retention"]["student_over_ref"]["ndcg@2_retention"] - expected_student) < 1e-9
