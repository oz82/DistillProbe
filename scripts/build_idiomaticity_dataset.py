"""Build an idiomaticity probe dataset from PARSEME 1.2 Turkish CUPT files.

Usage:
    python scripts/build_idiomaticity_dataset.py \
        data/idiomaticity/parseme_tr/train.cupt \
        data/idiomaticity/parseme_tr/dev.cupt   \
        data/idiomaticity/parseme_tr/test.cupt  \
        --out data/idiomaticity/tr_parseme.v1.json

The script extracts:
  - one record per MWE occurrence (VID, LVC.full, MVC, …): the gold idiomatic data
  - one literal-baseline record per sentence: the first non-MWE (verb, nominal-dep)
    pair found, ordered by token position. Category is `LITERAL`.

Filters applied:
  - MVC items dropped (typically n=5 in TR, too few for inference)
  - 1-token "MWE" rows dropped (annotation artefacts)
  - LITERAL pool optionally subsampled with `--max-literal` (default: 3500) to
    keep the sample sizes comparable across categories.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from distillprobe.data.parseme import Sentence, iter_many_cupt
from distillprobe.data.schemas import IdiomaticityDataset, IdiomItem

NOMINAL_UPOS = {"NOUN", "PROPN"}
LITERAL_DEPRELS = {"obj", "obl", "nmod", "nsubj"}
DROP_CATEGORIES = {"MVC"}  # too few annotations in TR
MIN_TOKEN_COUNT = 2


def _sentence_mwe_token_indices(sent: Sentence) -> set[int]:
    out: set[int] = set()
    for mwe in sent.mwes:
        out.update(mwe.token_indices)
    return out


def _first_literal_pair(sent: Sentence) -> tuple[list[int], str, str] | None:
    """Find the first non-MWE (verb, nominal-dep) pair in a sentence.

    Returns (sorted_token_indices, surface_phrase, lemma_phrase), or None.
    Indices are positions in `sent.tokens` (the full list including range rows).
    """
    mwe_set = _sentence_mwe_token_indices(sent)
    for i, verb in enumerate(sent.tokens):
        if verb.is_range or verb.upos != "VERB" or i in mwe_set:
            continue
        for j, dep in enumerate(sent.tokens):
            if dep.is_range or j in mwe_set or j == i:
                continue
            if dep.head != verb.id:
                continue
            if dep.deprel not in LITERAL_DEPRELS:
                continue
            if dep.upos not in NOMINAL_UPOS:
                continue
            indices = sorted([i, j])
            phrase = " ".join(sent.tokens[k].form for k in indices)
            lemma = " ".join(sent.tokens[k].lemma for k in indices)
            return indices, phrase, lemma
    return None


def _word_level_view(sent: Sentence) -> tuple[list[str], dict[int, int]]:
    """Return (`sentence_tokens` list of surface forms excluding range rows,
    map from full-token-list index → word-level index)."""
    full_to_word: dict[int, int] = {}
    forms: list[str] = []
    for i, t in enumerate(sent.tokens):
        if t.is_range:
            continue
        full_to_word[i] = len(forms)
        forms.append(t.form)
    return forms, full_to_word


def build(
    paths: list[Path],
    version: str,
    source: str,
    with_literal: bool,
    max_literal: int,
    seed: int,
) -> IdiomaticityDataset:
    items: list[IdiomItem] = []
    literal_pool: list[IdiomItem] = []
    n_sents = 0
    for sent in iter_many_cupt(paths):
        n_sents += 1
        if not sent.text:
            continue
        sentence_tokens, full_to_word = _word_level_view(sent)
        for mwe in sent.mwes:
            if not mwe.token_indices:
                continue
            if mwe.category in DROP_CATEGORIES:
                continue
            if len(mwe.token_indices) < MIN_TOKEN_COUNT:
                continue
            try:
                phrase_word_indices = sorted(full_to_word[i] for i in mwe.token_indices)
            except KeyError:
                continue  # shouldn't happen; would mean MWE points at a range row
            items.append(
                IdiomItem(
                    sent_id=sent.sent_id,
                    sentence_text=sent.text,
                    sentence_tokens=sentence_tokens,
                    phrase=mwe.phrase(sent),
                    lemma_phrase=mwe.lemma_phrase(sent),
                    category=mwe.category,
                    token_count=len(mwe.token_indices),
                    phrase_token_indices=phrase_word_indices,
                )
            )
        if with_literal:
            lit = _first_literal_pair(sent)
            if lit is not None:
                indices, phrase, lemma = lit
                try:
                    word_indices = sorted(full_to_word[i] for i in indices)
                except KeyError:
                    continue
                literal_pool.append(
                    IdiomItem(
                        sent_id=sent.sent_id,
                        sentence_text=sent.text,
                        sentence_tokens=sentence_tokens,
                        phrase=phrase,
                        lemma_phrase=lemma,
                        category="LITERAL",
                        token_count=len(indices),
                        phrase_token_indices=word_indices,
                    )
                )

    if with_literal and literal_pool:
        rng = random.Random(seed)
        if max_literal and len(literal_pool) > max_literal:
            literal_pool = rng.sample(literal_pool, max_literal)
        items.extend(literal_pool)

    print(f"sentences scanned:  {n_sents}", file=sys.stderr)
    print(f"items kept (total): {len(items)}", file=sys.stderr)
    cats = Counter(i.category for i in items)
    print("category distribution:", file=sys.stderr)
    for c, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:14s} {n:5d}", file=sys.stderr)

    return IdiomaticityDataset(
        language="tr",
        version=version,
        source=source,
        items=items,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="paths to .cupt files")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--version", default="v1-2026-05-05")
    p.add_argument("--source", default="PARSEME 1.2 Turkish (train+dev+test)")
    p.add_argument(
        "--with-literal",
        action="store_true",
        help="also extract one non-MWE (verb, nominal-dep) literal phrase per sentence",
    )
    p.add_argument(
        "--max-literal",
        type=int,
        default=3500,
        help="max number of literal samples to keep (subsampled with seed)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = [Path(s) for s in args.inputs]
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    ds = build(
        paths,
        version=args.version,
        source=args.source,
        with_literal=args.with_literal,
        max_literal=args.max_literal,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(ds.items)} items)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
