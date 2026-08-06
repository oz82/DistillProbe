"""Golden-value tests for the metric library."""

from __future__ import annotations

import math

import numpy as np
import pytest

from distillprobe.metrics import (
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    cosine_similarity,
    difference_vectors,
    holm_bonferroni,
    pairwise_consistency,
    paired_wilcoxon,
    permutation_test,
    random_pair_baseline,
    rule_of_three,
    stem_form_similarity,
    wilson_ci,
)


# --- consistency -----------------------------------------------------------------


def test_pairwise_consistency_identical_diffs_is_one():
    diffs = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
    assert pairwise_consistency(diffs) == pytest.approx(1.0, abs=1e-9)


def test_pairwise_consistency_orthogonal_diffs_is_zero():
    diffs = np.eye(4)  # 4 mutually orthogonal vectors
    assert pairwise_consistency(diffs) == pytest.approx(0.0, abs=1e-9)


def test_pairwise_consistency_antiparallel_pair_is_minus_one():
    diffs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert pairwise_consistency(diffs) == pytest.approx(-1.0, abs=1e-9)


def test_pairwise_consistency_nan_for_singleton():
    diffs = np.array([[1.0, 0.0, 0.0]])
    assert math.isnan(pairwise_consistency(diffs))


def test_difference_vectors_basic():
    stems = np.array([[1.0, 0.0], [0.0, 1.0]])
    forms = np.array([[2.0, 0.0], [0.0, 3.0]])
    out = difference_vectors(stems, forms)
    np.testing.assert_array_equal(out, np.array([[1.0, 0.0], [0.0, 2.0]]))


def test_difference_vectors_shape_mismatch():
    with pytest.raises(ValueError):
        difference_vectors(np.zeros((2, 3)), np.zeros((3, 3)))


# --- similarity ------------------------------------------------------------------


def test_cosine_similarity_identical_is_one():
    v = np.array([0.5, 0.5, 0.5, 0.5])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero():
    a = np.zeros(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_stem_form_similarity_batched():
    stems = np.array([[1.0, 0.0], [0.0, 1.0]])
    forms = np.array([[1.0, 0.0], [1.0, 0.0]])
    sims = stem_form_similarity(stems, forms)
    np.testing.assert_allclose(sims, np.array([1.0, 0.0]), atol=1e-9)


# --- baselines -------------------------------------------------------------------


def test_random_pair_baseline_is_seeded_and_reproducible():
    rng = np.random.default_rng(7)
    pool = [f"w{i}" for i in range(20)]
    vecs = {w: rng.normal(size=8) for w in pool}
    vecs = {w: v / np.linalg.norm(v) for w, v in vecs.items()}

    def enc(w):
        return vecs[w]

    a = random_pair_baseline(pool, enc, n_pairs=10, n_seeds=3, base_seed=42)
    b = random_pair_baseline(pool, enc, n_pairs=10, n_seeds=3, base_seed=42)
    assert a == b
    assert set(a.keys()) == {42, 43, 44}


# --- stats -----------------------------------------------------------------------


def test_bootstrap_ci_contains_mean():
    arr = np.linspace(0, 1, 50)
    res = bootstrap_ci(arr, n=2000, seed=1)
    assert res.low <= res.mean <= res.high
    assert res.n == 50


def test_bootstrap_ci_singleton():
    res = bootstrap_ci([0.42], n=1000, seed=1)
    assert res.mean == res.low == res.high == pytest.approx(0.42)
    assert res.n == 1


def test_bootstrap_ci_empty():
    res = bootstrap_ci([], n=100, seed=1)
    assert math.isnan(res.mean) and res.n == 0


def test_cliffs_delta_signs():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    assert cliffs_delta(x, y) == pytest.approx(-1.0)
    assert cliffs_delta(y, x) == pytest.approx(1.0)
    assert cliffs_delta(x, x) == pytest.approx(0.0)


def test_paired_wilcoxon_runs():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    res = paired_wilcoxon(x, y)
    assert hasattr(res, "pvalue")


# --- multiple-comparison correction ---------------------------------------------


def test_holm_bonferroni_known_values():
    # Classic worked example: Holm is uniformly >= raw and <= Bonferroni.
    p = [0.01, 0.04, 0.03]
    adj = holm_bonferroni(p)
    # sorted: 0.01(*3)=0.03, 0.03(*2)=0.06, 0.04(*1)=0.04 -> monotone: 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)  # 0.03 raw, bumped by monotonicity
    assert adj[1] == pytest.approx(0.06)
    assert np.all(adj <= 1.0)


def test_holm_bonferroni_caps_at_one():
    adj = holm_bonferroni([0.5, 0.9])
    assert np.all(adj <= 1.0)


def test_benjamini_hochberg_monotone_and_bounded():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    adj = benjamini_hochberg(p)
    assert np.all(adj <= 1.0)
    assert np.all(adj >= np.asarray(p))  # adjusted never below raw
    # BH for largest p equals itself (m/m * p)
    assert adj[4] == pytest.approx(0.9)


def test_holm_vs_bh_ordering():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    holm = holm_bonferroni(p)
    bh = benjamini_hochberg(p)
    # Holm (FWER) is at least as conservative as BH (FDR) elementwise.
    assert np.all(holm >= bh - 1e-12)


# --- binomial CIs ----------------------------------------------------------------


def test_wilson_ci_zero_successes():
    # 0/98 negation flips: lower bound 0, upper bound small and < rule-of-three-ish
    res = wilson_ci(0, 98)
    assert res.low == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < res.high < 0.06
    assert res.mean == 0.0


def test_wilson_ci_contains_phat():
    res = wilson_ci(11, 98)
    assert res.low <= res.mean <= res.high
    assert res.mean == pytest.approx(11 / 98)


def test_rule_of_three():
    assert rule_of_three(98) == pytest.approx(3.0 / 98, rel=0.05)
    assert math.isnan(rule_of_three(0))


# --- permutation test ------------------------------------------------------------


def test_permutation_test_separated_groups():
    x = np.array([0.085, 0.091, 0.088])  # fusional gaps
    y = np.array([0.052, 0.070])          # agglutinative gaps
    res = permutation_test(x, y, n=5000, seed=1)
    assert res["observed"] > 0
    assert 0.0 < res["pvalue"] <= 1.0


def test_permutation_test_reproducible():
    x = [1.0, 2.0, 3.0]
    y = [1.5, 2.5]
    a = permutation_test(x, y, n=1000, seed=42)
    b = permutation_test(x, y, n=1000, seed=42)
    assert a["pvalue"] == b["pvalue"]
