from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    type: Literal["model2vec", "fasttext", "transformer", "static_wordvec"]
    model_id: str | None = None
    path: str | None = None
    pool: str | None = None  # transformer only: "mean" | "cls"
    # How probes should treat this encoder. Optional: when absent, modules fall back
    # to the name heuristics the canonical runs used (model2vec = student,
    # transformer/LaBSE/mE5 = reference). Required in practice for encoders whose
    # name would mislead the heuristic, e.g. a sentence-transformers *static* model.
    role: Literal["student", "reference", "baseline"] | None = None
    # static_wordvec only: tokenizers used to split words and to back off OOV words.
    word_tokenizer: str | None = None
    subword_tokenizer: str | None = None


class RunConfig(BaseModel):
    module: str
    encoders: dict[str, EncoderConfig]
    data_path: str
    seed: int = 42
    bootstrap_n: int = 10_000
    output_dir: str = "results"
    extra: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping; got {type(raw).__name__}")
    return RunConfig(**raw)


def build_encoder(cfg: EncoderConfig):
    """Instantiate the appropriate Encoder for a config block."""
    enc = _build(cfg)
    enc.role = cfg.role
    return enc


def _build(cfg: EncoderConfig):
    if cfg.type == "model2vec":
        from .encoders import Model2VecEncoder

        if not cfg.model_id:
            raise ValueError("model2vec encoder requires `model_id`")
        return Model2VecEncoder(model_id=cfg.model_id)

    if cfg.type == "fasttext":
        from .encoders import FastTextEncoder

        if not cfg.path:
            raise ValueError("fasttext encoder requires `path`")
        return FastTextEncoder(path=cfg.path)

    if cfg.type == "transformer":
        # Imported lazily: sentence-transformers is an optional extra, so the
        # package need not be installed unless a transformer encoder is used.
        from .encoders.transformer_encoder import TransformerEncoder  # type: ignore[attr-defined]

        if not cfg.model_id:
            raise ValueError("transformer encoder requires `model_id`")
        return TransformerEncoder(model_id=cfg.model_id, pool=cfg.pool or "mean")

    if cfg.type == "static_wordvec":
        from .encoders import StaticWordVecEncoder

        if not cfg.path:
            raise ValueError("static_wordvec encoder requires `path`")
        kwargs: dict[str, Any] = {}
        if cfg.word_tokenizer:
            kwargs["word_tokenizer"] = cfg.word_tokenizer
        if cfg.subword_tokenizer is not None:
            kwargs["subword_tokenizer"] = cfg.subword_tokenizer or None
        return StaticWordVecEncoder(path=cfg.path, **kwargs)

    raise ValueError(f"unknown encoder type: {cfg.type}")
