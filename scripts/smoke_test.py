"""End-to-end smoke test for the real Model2Vec and fastText encoders.

Run after `pip install -e .`:
    python scripts/smoke_test.py

This actually loads the models and verifies the Encoder contract on a few
Turkish words. fastText is heavy (~4 GB RAM); first invocation will be slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FASTTEXT_PATH = ROOT / "data" / "fasttext" / "cc.tr.300.bin"

WORDS = ["ev", "evler", "Mehmetoğlu", "kitap"]


def _check(enc, label: str) -> None:
    print(f"\n[{label}] dim={enc.dim} name={enc.name}")
    for w in WORDS:
        v = enc.encode(w)
        n = float(np.linalg.norm(v))
        toks = enc.tokenize(w)
        print(f"  {w:14s} ||v||={n:.5f} dim={v.shape}  tokens={toks[:6]}")
        assert v.shape == (enc.dim,), f"shape mismatch for {w}"
        assert abs(n - 1.0) < 1e-3 or n == 0.0, f"not L2-normalized for {w}: {n}"


def main() -> int:
    from distillprobe.encoders import FastTextEncoder, Model2VecEncoder

    print("loading Model2Vec…")
    m2v = Model2VecEncoder("minishlab/potion-multilingual-128M")
    _check(m2v, "Model2Vec")

    if FASTTEXT_PATH.exists():
        print(f"\nloading fastText from {FASTTEXT_PATH} (this can take a minute)…")
        ft = FastTextEncoder(FASTTEXT_PATH)
        _check(ft, "fastText")
    else:
        print(f"\nSKIP fastText: not found at {FASTTEXT_PATH}")

    print("\nsmoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
