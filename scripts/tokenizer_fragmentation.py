"""Subword fragmentation of the student tokenizer per language on FLORES-200.

Usage: python scripts/tokenizer_fragmentation.py [--data data/xlingual/tr_flores.v3.json]

For each language: total subword tokens of potion-multilingual-128M's tokenizer
(inherited from bge-m3; no special tokens) divided by total whitespace-delimited
words, pooled over the 997 FLORES sentences of the cross-lingual probe. Used for
the Turkmen outlier in the cross-lingual section (tk vs. the other Turkic languages).
"""

from __future__ import annotations

import argparse
import json

from model2vec import StaticModel

TURKIC = ["tk", "tr", "az", "kk", "ky", "uz"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/xlingual/tr_flores.v3.json")
    p.add_argument("--model", default="minishlab/potion-multilingual-128M")
    p.add_argument("--all", action="store_true", help="report every language, not only Turkic")
    args = p.parse_args()

    tokenizer = StaticModel.from_pretrained(args.model).tokenizer
    data = json.load(open(args.data, encoding="utf-8"))
    available = data["languages"]
    langs = available if args.all else [lang for lang in TURKIC if lang in available]
    for lang in langs:
        texts = [pair["text_by_lang"][lang] for pair in data["pairs"]]
        words = sum(len(t.split()) for t in texts)
        subwords = sum(len(tokenizer.encode(t, add_special_tokens=False).tokens) for t in texts)
        print(f"{lang}: {subwords / words:.2f} subwords/word ({len(texts)} sentences)")


if __name__ == "__main__":
    main()
