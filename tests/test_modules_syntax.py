"""Control-task selectivity logic for the syntax V2 probe.

These tests validate the parameter-free LOO nearest-centroid classifier that
underlies the Hewitt & Liang (2019) control task: with real (separable) labels
accuracy is high, while with permuted labels it collapses to ~chance. The gap
is the selectivity reported by SyntaxModuleV2.
"""

from __future__ import annotations

import numpy as np

from distillprobe.modules.syntax import SyntaxModuleV2


def _two_cluster_features(n_per: int = 12, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=+2.0, scale=0.2, size=(n_per, dim))
    b = rng.normal(loc=-2.0, scale=0.2, size=(n_per, dim))
    features = np.vstack([a, b]).astype(np.float32)
    labels = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    return features, labels


def test_loo_accuracy_high_on_separable_labels():
    features, labels = _two_cluster_features()
    acc = SyntaxModuleV2._loo_nearest_centroid_accuracy(features, labels)
    assert acc > 0.95


def test_control_labels_collapse_to_chance():
    features, labels = _two_cluster_features()
    rng = np.random.default_rng(42)
    ctrl_accs = [
        SyntaxModuleV2._loo_nearest_centroid_accuracy(features, rng.permutation(labels))
        for _ in range(20)
    ]
    mean_ctrl = float(np.mean(ctrl_accs))
    # permuted labels destroy the representation-label link -> ~chance
    assert 0.3 < mean_ctrl < 0.7


def test_selectivity_is_positive_and_large():
    features, labels = _two_cluster_features()
    real = SyntaxModuleV2._loo_nearest_centroid_accuracy(features, labels)
    rng = np.random.default_rng(1)
    ctrl = np.mean(
        [
            SyntaxModuleV2._loo_nearest_centroid_accuracy(features, rng.permutation(labels))
            for _ in range(20)
        ]
    )
    selectivity = real - ctrl
    assert selectivity > 0.3
