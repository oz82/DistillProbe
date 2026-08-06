"""End-to-end smoke test for MorphologyModule using stub encoders."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from distillprobe.encoders.base import Encoder, l2_normalize
from distillprobe.modules import MorphologyModule


class StubEncoder(Encoder):
    """Deterministic 8-D encoder. `salt` lets us fabricate a second encoder."""

    def __init__(self, salt: str = "a"):
        self._name = f"stub:{salt}"
        self._salt = salt

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return 8

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256((self._salt + ":" + text).encode("utf-8")).digest()
        v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32) - 127.5
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        return list(text)


def _mini_dataset() -> dict:
    return {
        "language": "tr",
        "version": "test",
        "categories": [
            {
                "id": "plural_lAr",
                "label": "plural -lAr",
                "pairs": [
                    {"stem": "ev", "form": "evler"},
                    {"stem": "kitap", "form": "kitaplar"},
                    {"stem": "yol", "form": "yollar"},
                    {"stem": "araba", "form": "arabalar"},
                ],
            },
            {
                "id": "locative_DA",
                "label": "locative -DA",
                "pairs": [
                    {"stem": "ev", "form": "evde"},
                    {"stem": "okul", "form": "okulda"},
                    {"stem": "yol", "form": "yolda"},
                    {"stem": "park", "form": "parkta"},
                ],
            },
        ],
        "distractor_pool": [f"w{i}" for i in range(20)],
    }


def test_morphology_e2e_with_stub(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(_mini_dataset()), encoding="utf-8")

    encoders = {"e1": StubEncoder("a"), "e2": StubEncoder("b")}
    cfg = SimpleNamespace(seed=42, bootstrap_n=500)

    res = MorphologyModule().run(encoders=encoders, data_path=str(p), config=cfg)

    assert res.module == "morphology"

    # per-category structure
    assert len(res.per_category) == 2
    cat0 = res.per_category[0]
    assert cat0["id"] == "plural_lAr"
    assert cat0["n"] == 4
    for enc_name in encoders:
        assert f"consistency:{enc_name}" in cat0
        assert f"stem_form_sim_mean:{enc_name}" in cat0
        assert f"stem_form_sim_ci_low:{enc_name}" in cat0
        assert f"stem_form_sim_ci_high:{enc_name}" in cat0

    # tokenization audit: one row per (category, pair); 4 + 4 pairs total
    assert len(res.tokenization_audit) == 8
    audit0 = res.tokenization_audit[0]
    assert audit0["category"] == "plural_lAr"
    for enc_name in encoders:
        assert f"tokens:stem:{enc_name}" in audit0
        assert f"tokens:form:{enc_name}" in audit0

    # baselines: one entry per encoder
    assert set(res.baselines) == set(encoders)
    for enc_name in encoders:
        b = res.baselines[enc_name]
        assert "mean" in b and "std" in b and "per_seed" in b
        assert len(b["per_seed"]) == 5

    # comparisons: e1 vs e2
    assert "e1_vs_e2" in res.metrics["comparisons"]
    cmp_ = res.metrics["comparisons"]["e1_vs_e2"]
    assert "wilcoxon_p" in cmp_ and "cliffs_delta" in cmp_

    # encoders summary
    enc_summary = {e["name"] for e in res.metrics["encoders"]}
    assert enc_summary == {"stub:a", "stub:b"}


def test_morphology_single_encoder_no_comparison(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(_mini_dataset()), encoding="utf-8")

    res = MorphologyModule().run(
        encoders={"only": StubEncoder("c")},
        data_path=str(p),
        config=SimpleNamespace(seed=1, bootstrap_n=200),
    )
    assert res.metrics["comparisons"] == {}
    assert "only" in res.metrics["per_encoder_mean_consistency"]
