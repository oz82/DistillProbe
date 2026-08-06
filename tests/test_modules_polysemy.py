"""End-to-end smoke test for PolysemyModule using stub encoders."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from distillprobe.encoders.base import Encoder, l2_normalize
from distillprobe.modules import PolysemyModule


class StubEncoder(Encoder):
    """Deterministic 8-D encoder. Different `salt`s produce different mappings."""

    DIM = 8

    def __init__(self, salt: str = "x"):
        self._name = f"stub:{salt}"
        self._salt = salt

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self.DIM

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256((self._salt + ":" + text).encode("utf-8")).digest()
        v = np.frombuffer(h[: self.DIM], dtype=np.uint8).astype(np.float32) - 127.5
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        return list(text)


def _mini_dataset() -> dict:
    return {
        "language": "tr",
        "version": "test",
        "source": "synthetic",
        "words": [
            {
                "word": "yüz",
                "senses": [
                    {
                        "sense_id": "tdk:1",
                        "definition": "İnsan başının ön bölümü",
                        "examples": ["Çocuğun yüzü gülüyordu."],
                    },
                    {
                        "sense_id": "tdk:2",
                        "definition": "Suda hareket etmek",
                        "examples": ["Denizde yüzdüm."],
                    },
                    {
                        "sense_id": "tdk:3",
                        "definition": "Yüz sayısı",
                        "examples": ["Yüz kişi vardı."],
                    },
                ],
            },
            {
                "word": "çay",
                "senses": [
                    {
                        "sense_id": "tdk:4",
                        "definition": "Sıcak içecek",
                        "examples": ["Çay içtim."],
                    },
                    {
                        "sense_id": "tdk:5",
                        "definition": "Küçük dere",
                        "examples": ["Çay aktı."],
                    },
                ],
            },
            {
                "word": "ay",
                "senses": [
                    {
                        "sense_id": "tdk:6",
                        "definition": "Dünyanın uydusu",
                        "examples": ["Ay ışığı."],
                    },
                    {
                        "sense_id": "tdk:7",
                        "definition": "Yılın bölümlerinden biri",
                        "examples": ["Bir ay sonra."],
                    },
                ],
            },
        ],
    }


def test_polysemy_e2e_with_stubs(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(_mini_dataset()), encoding="utf-8")

    encoders = {"e1": StubEncoder("a"), "e2": StubEncoder("b")}
    cfg = SimpleNamespace(seed=1, bootstrap_n=200)

    res = PolysemyModule().run(encoders=encoders, data_path=str(p), config=cfg)

    assert res.module == "polysemy"
    # All 3 polysemic words have ≥2 senses → all kept
    assert len(res.per_category) == 3
    for r in res.per_category:
        # Each encoder produced top1/margin/entropy
        for en in encoders:
            assert f"top1_idx:{en}" in r
            assert f"top1_score:{en}" in r
            assert f"margin:{en}" in r
            assert f"entropy:{en}" in r
            assert isinstance(r[f"sense_sims:{en}"], list)
        # Cross-encoder fields
        assert "agree_winner:e1_e2" in r
        assert "spearman:e1_e2" in r

    # Summary structure
    metrics = res.metrics
    assert metrics["n_words"] == 3
    assert set(metrics["per_encoder"]) == set(encoders)
    for en in encoders:
        per = metrics["per_encoder"][en]
        assert "mean_top1_score" in per
        assert "top1_ci_low" in per
        assert "first_sense_bias_rate" in per

    cross = metrics["cross_encoder"]["e1_vs_e2"]
    assert "winner_agreement_rate" in cross
    assert "random_agreement_baseline" in cross
    assert "mean_spearman" in cross


def test_polysemy_skips_word_with_one_sense(tmp_path):
    ds = _mini_dataset()
    ds["words"].append(
        {"word": "tek", "senses": [
            {"sense_id": "tdk:99", "definition": "Yalnız", "examples": ["Tek başına."]}
        ]}
    )
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(ds), encoding="utf-8")

    res = PolysemyModule().run(
        encoders={"e": StubEncoder("c")},
        data_path=str(p),
        config=SimpleNamespace(seed=1, bootstrap_n=100),
    )
    words_kept = {r["word"] for r in res.per_category}
    assert "tek" not in words_kept
    assert words_kept == {"yüz", "çay", "ay"}
