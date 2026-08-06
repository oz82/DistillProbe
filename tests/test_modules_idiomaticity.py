"""End-to-end smoke test for IdiomaticityModule using stub encoders."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from distillprobe.encoders.base import Encoder, l2_normalize
from distillprobe.modules import IdiomaticityModule


class CompositionalStub(Encoder):
    """Bag-of-words stub: phrase vector = mean of token vectors. gap → 0."""

    DIM = 8

    def __init__(self, salt: str = "x"):
        self._name = f"compositional:{salt}"
        self._salt = salt

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self.DIM

    def _tok_vec(self, tok: str) -> np.ndarray:
        h = hashlib.sha256((self._salt + ":" + tok).encode("utf-8")).digest()
        v = np.frombuffer(h[: self.DIM], dtype=np.uint8).astype(np.float32) - 127.5
        return l2_normalize(v)

    def encode(self, text: str) -> np.ndarray:
        toks = text.split() or [text]
        v = np.mean(np.stack([self._tok_vec(t) for t in toks], axis=0), axis=0)
        return l2_normalize(v)

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class IdiomaticStub(Encoder):
    """Pseudo-contextual stub: phrase vector unrelated to its token vectors when
    the phrase is multi-token (simulating non-compositional behaviour). Single-
    token inputs are encoded compositionally."""

    DIM = 8

    @property
    def name(self) -> str:
        return "idiomatic-stub"

    @property
    def dim(self) -> int:
        return self.DIM

    def _h(self, key: str) -> np.ndarray:
        h = hashlib.sha256(key.encode("utf-8")).digest()
        v = np.frombuffer(h[: self.DIM], dtype=np.uint8).astype(np.float32) - 127.5
        return l2_normalize(v)

    def encode(self, text: str) -> np.ndarray:
        toks = text.split()
        if len(toks) >= 2:
            # Non-compositional: hash the whole phrase, ignore tokens.
            return self._h("PHRASE:" + text)
        return self._h("TOK:" + text)

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _mini_dataset() -> dict:
    return {
        "language": "tr",
        "version": "test",
        "source": "synthetic",
        "items": [
            # 4 each so bootstrap CI is meaningful
            {"sentence_text": "x", "phrase": "yol açtı", "lemma_phrase": "yol aç",
             "category": "VID", "token_count": 2},
            {"sentence_text": "x", "phrase": "sona erdi", "lemma_phrase": "son er",
             "category": "VID", "token_count": 2},
            {"sentence_text": "x", "phrase": "kafa ütüledi", "lemma_phrase": "kafa ütüle",
             "category": "VID", "token_count": 2},
            {"sentence_text": "x", "phrase": "el atmış", "lemma_phrase": "el at",
             "category": "VID", "token_count": 2},
            {"sentence_text": "x", "phrase": "ifade etti", "lemma_phrase": "ifade et",
             "category": "LVC.full", "token_count": 2},
            {"sentence_text": "x", "phrase": "rahat etti", "lemma_phrase": "rahat et",
             "category": "LVC.full", "token_count": 2},
            {"sentence_text": "x", "phrase": "neden oldu", "lemma_phrase": "neden ol",
             "category": "LVC.full", "token_count": 2},
            {"sentence_text": "x", "phrase": "ödeme yapıldı", "lemma_phrase": "öde yapıl",
             "category": "LVC.full", "token_count": 2},
            {"sentence_text": "x", "phrase": "kitap okudu", "lemma_phrase": "kitap oku",
             "category": "LITERAL", "token_count": 2},
            {"sentence_text": "x", "phrase": "su içti", "lemma_phrase": "su iç",
             "category": "LITERAL", "token_count": 2},
            {"sentence_text": "x", "phrase": "yemek yedi", "lemma_phrase": "yemek ye",
             "category": "LITERAL", "token_count": 2},
            {"sentence_text": "x", "phrase": "araba sürdü", "lemma_phrase": "araba sür",
             "category": "LITERAL", "token_count": 2},
        ],
    }


def test_idiomaticity_e2e_with_stubs(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(_mini_dataset()), encoding="utf-8")

    encoders = {
        "compos": CompositionalStub("a"),
        "idiom": IdiomaticStub(),
    }
    cfg = SimpleNamespace(seed=1, bootstrap_n=200)

    res = IdiomaticityModule().run(encoders=encoders, data_path=str(p), config=cfg)

    assert res.module == "idiomaticity"

    # All three categories present
    cats = [r["category"] for r in res.per_category]
    assert cats == ["VID", "LVC.full", "LITERAL"]

    # Compositional stub: phrase = mean(tokens) by construction → gap ≈ 0
    for row in res.per_category:
        assert abs(row["gap_mean:compos"]) < 1e-5

    # Idiomatic stub: phrase ≠ mean(tokens) for multi-token → mean gap > 0
    for row in res.per_category:
        assert row["gap_mean:idiom"] > 0.1

    # Comparison entries exist for adjacent pairs
    cmp_keys = res.metrics["comparisons"].keys()
    assert any("VID_gt_LVC.full" in k for k in cmp_keys)
    assert any("LVC.full_gt_LITERAL" in k for k in cmp_keys)


def test_idiomaticity_skip_unknown_category(tmp_path):
    """Items in non-spectrum categories (e.g. MVC) are silently dropped."""
    ds = _mini_dataset()
    ds["items"].append(
        {"sentence_text": "x", "phrase": "fluctus tide", "lemma_phrase": "f t",
         "category": "MVC", "token_count": 2}
    )
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(ds), encoding="utf-8")

    res = IdiomaticityModule().run(
        encoders={"e": CompositionalStub()},
        data_path=str(p),
        config=SimpleNamespace(seed=1, bootstrap_n=200),
    )
    cats = [r["category"] for r in res.per_category]
    assert "MVC" not in cats
