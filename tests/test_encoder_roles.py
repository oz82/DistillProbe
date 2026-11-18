"""Encoder role resolution and the static word-vector encoder.

Modules pick the student and the reference encoder either from an explicit
`role` (set from the run config) or, for backwards compatibility with the
canonical configs, from name heuristics. Both paths are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from distillprobe.config import EncoderConfig, RunConfig
from distillprobe.encoders.base import (
    Encoder,
    first_reference,
    first_student,
    is_reference_encoder,
    is_student_encoder,
    l2_normalize,
)


class _Named(Encoder):
    def __init__(self, name: str, role: str | None = None):
        self._name = name
        self.role = role

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return 2

    def encode(self, text: str) -> np.ndarray:
        return l2_normalize(np.ones(2, dtype=np.float32))

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def test_name_heuristics_match_canonical_configs():
    assert is_student_encoder(_Named("model2vec:minishlab/potion-multilingual-128M"))
    assert not is_student_encoder(_Named("fasttext:cc.tr.300.bin"))
    assert is_reference_encoder(_Named("transformer:sentence-transformers/LaBSE"))
    assert is_reference_encoder(_Named("transformer:intfloat/multilingual-e5-base"))
    assert not is_reference_encoder(_Named("model2vec:models/m2v_bgem3_pca256_sif"))
    assert not is_reference_encoder(_Named("fasttext:cc.en.300.bin"))


def test_explicit_role_overrides_name():
    # A sentence-transformers *static* model loads through the transformer encoder,
    # so its name would flag it as a reference; the config role must win.
    static = _Named("transformer:sentence-transformers/static-similarity-mrl-multilingual-v1",
                    role="student")
    assert is_student_encoder(static)
    assert not is_reference_encoder(static)
    baseline = _Named("model2vec:some/model", role="baseline")
    assert not is_student_encoder(baseline)
    assert not is_reference_encoder(baseline)


def test_first_student_and_reference_follow_config_order():
    encoders = {
        "ft": _Named("fasttext:cc.tr.300.bin"),
        "labse": _Named("transformer:sentence-transformers/LaBSE"),
        "wada": _Named("static_wordvec:swe_gte256_en.txt", role="student"),
        "potion": _Named("model2vec:minishlab/potion-multilingual-128M"),
    }
    assert first_reference(encoders) == "labse"
    assert first_student(encoders) == "wada"
    assert first_student({"potion": encoders["potion"]}) == "potion"
    assert first_reference({"ft": encoders["ft"]}) is None


def test_encoder_config_accepts_role_and_rejects_unknown():
    cfg = RunConfig(
        module="negation",
        data_path="x.json",
        encoders={
            "s": EncoderConfig(type="static_wordvec", path="vec.txt", role="student"),
            "r": EncoderConfig(type="transformer", model_id="m", role="reference"),
            "p": EncoderConfig(type="model2vec", model_id="m"),
        },
    )
    assert cfg.encoders["s"].role == "student"
    assert cfg.encoders["p"].role is None
    with pytest.raises(ValueError):
        EncoderConfig(type="model2vec", model_id="m", role="teacher")


# --- StaticWordVecEncoder: lookup, back-off and pooling without any download ------


def _static_with_table(table: dict[str, list[float]]):
    from distillprobe.encoders.static_wordvec_encoder import StaticWordVecEncoder

    enc = StaticWordVecEncoder.__new__(StaticWordVecEncoder)  # bypass __init__
    enc.path = "mock.txt"
    enc.word_tokenizer_id = "mock"
    enc.subword_tokenizer_id = "mock"
    enc._word2vec = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
    enc._dim = 4
    enc._pre_tokenize = lambda text: [(w, (0, 0)) for w in text.replace(",", " ,").split()]

    class _Sub:
        @staticmethod
        def tokenize(word):  # "unbelievably" -> un ##believ ##ably
            return [word[:2]] + ["##" + word[2:6], "##" + word[6:]]

    enc._subword = _Sub()
    return enc


def test_static_wordvec_mean_pools_and_skips_punctuation():
    enc = _static_with_table({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    v = enc.encode("a , b")
    np.testing.assert_allclose(v, [0.70710678, 0.70710678, 0, 0], atol=1e-6)
    assert enc.tokenize("a , b") == ["a", "b"]


def test_static_wordvec_case_fallback_and_subword_backoff():
    enc = _static_with_table({"hello": [0, 0, 1, 0], "unbeli": [0, 0, 0, 1]})
    np.testing.assert_allclose(enc.encode("Hello"), [0, 0, 1, 0], atol=1e-6)
    # "unbelievably" is OOV. The mock splits it un / ##beli / ##evably; the back-off
    # drops one subword at a time and looks up the joined prefix: "unbeli" hits.
    np.testing.assert_allclose(enc.encode("unbelievably"), [0, 0, 0, 1], atol=1e-6)
    # With no prefix in the table ("unbeli" then "un" both miss) the word contributes
    # nothing, and a sentence with no known word encodes to the zero vector.
    enc2 = _static_with_table({"hello": [0, 0, 1, 0]})
    assert float(np.linalg.norm(enc2.encode("unbelievably"))) == 0.0


def test_static_wordvec_load_parses_word2vec_text(tmp_path):
    from distillprobe.encoders.static_wordvec_encoder import StaticWordVecEncoder

    f = tmp_path / "vec.txt"
    f.write_text("2 3\nfoo 1 0 0\nbar 0 1 0 \n", encoding="utf-8")
    table, dim = StaticWordVecEncoder._load(f)
    assert dim == 3 and set(table) == {"foo", "bar"}
    np.testing.assert_array_equal(table["bar"], [0, 1, 0])
