"""Build the English polysemy probe dataset from Princeton WordNet (via NLTK).

English counterpart of `extract_kenet_polysemic.py` + `build_polysemy_dataset.py`
(KeNet → TDK GTS). WordNet plays both roles at once: synsets give the sense
inventory and, like TDK entries, each synset carries a gloss (definition) and
example sentences, so the sense prototype used by the probe —
``mean(enc(definition), enc(example))`` — is built from the same kind of text.

Selection mirrors the Turkish recipe: single-word common nouns with >= `--min-senses`
noun synsets that each have >= `--min-examples` example sentences, optionally
restricted to everyday vocabulary by a corpus-frequency floor (`--freq-conllu`),
then a seeded random sample of `--n-words` (the Turkish 50 were likewise not chosen
for extreme polysemy). WordNet has at most a few examples per synset, so
`--min-examples` defaults to 1 (TDK allowed 2). Sense order follows WordNet's
frequency ordering (SemCor counts), which is what the probe's `first_sense_bias`
metric assumes.

Nothing is redistributed that is not already open: WordNet 3.0 is released under
the WordNet license, so the dataset ships glosses verbatim (unlike the TDK case).

Usage:
    python scripts/build_polysemy_dataset_en.py --out data/polysemy/en_polysemy.v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distillprobe.data.schemas import PolysemyDataset, PolysemySense, PolysemyWord


def build_word(lemma: str, min_examples: int) -> list[PolysemySense]:
    from nltk.corpus import wordnet as wn

    senses: list[PolysemySense] = []
    for syn in wn.synsets(lemma, pos="n"):
        # `synsets()` applies morphological stemming, so `hands` would inherit every
        # sense of `hand`; keep only synsets in which this exact lemma is listed.
        if lemma not in {lem.name().lower() for lem in syn.lemmas()}:
            continue
        examples = [e.strip() for e in syn.examples() if e and e.strip()]
        if len(examples) < min_examples:
            continue
        senses.append(
            PolysemySense(
                sense_id=f"wn:{syn.name()}",
                homograph_id=None,
                definition=syn.definition().strip(),
                pos="noun",
                examples=examples,
            )
        )
    return senses


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-words", type=int, default=50)
    p.add_argument("--min-senses", type=int, default=3)
    p.add_argument("--min-examples", type=int, default=1)
    p.add_argument("--min-len", type=int, default=3, help="minimum lemma length")
    p.add_argument(
        "--freq-conllu",
        nargs="*",
        default=None,
        help="UD CoNLL-U files; when given, candidates must occur as NOUN lemmas at least "
        "--min-freq times (keeps the sample to everyday vocabulary, as KeNet+TDK did for Turkish)",
    )
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument(
        "--select",
        choices=["random", "top"],
        default="random",
        help="random: seeded sample of qualifying nouns (default); top: most polysemous",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v1-en-wordnet-2026-09-14")
    args = p.parse_args()

    from nltk.corpus import wordnet as wn

    freq: dict[str, int] | None = None
    if args.freq_conllu:
        from collections import Counter

        from distillprobe.data.parseme import iter_many_cupt

        freq = Counter()
        for sent in iter_many_cupt([Path(s) for s in args.freq_conllu]):
            for tok in sent.tokens:
                if not tok.is_range and tok.upos == "NOUN" and tok.lemma:
                    freq[tok.lemma.lower()] += 1

    candidates: list[tuple[int, str, list[PolysemySense]]] = []
    for lemma in wn.all_lemma_names(pos="n"):
        if not lemma.isalpha() or not lemma.islower() or len(lemma) < args.min_len:
            continue
        if freq is not None and freq.get(lemma, 0) < args.min_freq:
            continue
        senses = build_word(lemma, args.min_examples)
        if len(senses) >= args.min_senses:
            candidates.append((len(senses), lemma, senses))
    print(
        f"{len(candidates)} nouns with >= {args.min_senses} example-bearing senses"
        + (f" and corpus frequency >= {args.min_freq}" if freq is not None else ""),
        file=sys.stderr,
    )

    candidates.sort(key=lambda t: (-t[0], t[1]))
    if args.select == "random":
        import random

        chosen = sorted(
            random.Random(args.seed).sample(candidates, min(args.n_words, len(candidates))),
            key=lambda t: t[1],
        )
    else:
        chosen = candidates[: args.n_words]
    kept = [PolysemyWord(word=lemma, senses=senses) for _, lemma, senses in chosen]
    for w in kept:
        print(
            f"  [{w.word}] {len(w.senses)} senses; examples per sense: {[len(s.examples) for s in w.senses]}",
            file=sys.stderr,
        )

    ds = PolysemyDataset(
        language="en",
        version=args.version,
        source=(
            f"Princeton WordNet {wn.get_version()} via NLTK; noun synsets listing the exact lemma, with "
            f">= {args.min_examples} example(s); words with >= {args.min_senses} such senses"
            + (
                f", NOUN lemma frequency >= {args.min_freq} in {len(args.freq_conllu)} UD file(s)"
                if freq is not None
                else ""
            )
            + (
                f"; seeded random sample of {args.n_words} (seed={args.seed})"
                if args.select == "random"
                else f"; top-{args.n_words} by sense count"
            )
        ),
        words=kept,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    n_senses = sum(len(w.senses) for w in kept)
    print(f"wrote {out}: {len(kept)} words / {n_senses} senses", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
