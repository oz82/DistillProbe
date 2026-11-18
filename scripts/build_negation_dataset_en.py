"""Build the English negation triplet dataset from WordNet adjective antonyms.

English counterpart of `build_negation_dataset_v2.py` (KeNet). WordNet encodes
antonymy at the lemma level (`Lemma.antonyms()`), so every (adjective, antonym)
pair is read directly; pairs are deduplicated symmetrically, subsampled to
`--max-pairs` with a fixed seed (mirroring the KeNet subsample), and combined
with a hand-curated seed list of gradable antonyms (the analogue of the Turkish v1
list). The negated form is the free-word construction ``not X`` — the English
analogue of Turkish ``X değil``.

The output is a *raw* v2 set; run `clean_negation_dataset_en.py` on it to remove
morphological-negation contaminants (un-/in-/dis- prefixes, -less privatives),
exactly as the Turkish v2 → v3_clean step did for -lI/-sIz.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from distillprobe.data.schemas import NegationDataset, NegationTriplet

# Hand-curated gradable antonyms (author-curated, cf. the 25 Turkish v1 triplets).
SEED_PAIRS: list[tuple[str, str]] = [
    ("good", "bad"),
    ("big", "small"),
    ("hot", "cold"),
    ("tall", "short"),
    ("fast", "slow"),
    ("happy", "sad"),
    ("rich", "poor"),
    ("strong", "weak"),
    ("new", "old"),
    ("light", "dark"),
    ("heavy", "light"),
    ("clean", "dirty"),
    ("easy", "difficult"),
    ("full", "empty"),
    ("wet", "dry"),
    ("cheap", "expensive"),
    ("early", "late"),
    ("loud", "quiet"),
    ("thick", "thin"),
    ("wide", "narrow"),
    ("deep", "shallow"),
    ("safe", "dangerous"),
    ("sweet", "bitter"),
    ("warm", "cool"),
    ("bright", "dim"),
]


def is_clean_word(s: str) -> bool:
    """Single alphabetic word, length >= 2 (WordNet lemmas may contain '_' or '-')."""
    return bool(s) and len(s) >= 2 and s.isalpha()


def wordnet_adjective_antonyms() -> list[tuple[str, str]]:
    from nltk.corpus import wordnet as wn

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    # Iterate in WordNet's deterministic synset order so the subsample is reproducible.
    for syn in wn.all_synsets("a"):
        for lemma in syn.lemmas():
            src = lemma.name().lower()
            if not is_clean_word(src):
                continue
            for ant in lemma.antonyms():
                tgt = ant.name().lower()
                if not is_clean_word(tgt) or src == tgt:
                    continue
                key = tuple(sorted((src, tgt)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((src, tgt))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="WordNet pairs kept after seeded subsampling (KeNet used 200)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v2-wordnet-2026-09-14")
    args = p.parse_args()

    pairs = wordnet_adjective_antonyms()
    print(f"WordNet adjective antonym pairs (deduped): {len(pairs)}", file=sys.stderr)
    if len(pairs) > args.max_pairs:
        pairs = random.Random(args.seed).sample(pairs, args.max_pairs)
        print(f"  subsampled to {len(pairs)}", file=sys.stderr)

    seed_keys = {tuple(sorted(k)) for k in SEED_PAIRS}
    triplets: list[NegationTriplet] = [
        NegationTriplet(positive=a, negation=f"not {a}", antonym=b, note="hand-curated seed")
        for a, b in SEED_PAIRS
    ]
    n_added = 0
    for src, tgt in pairs:
        if tuple(sorted((src, tgt))) in seed_keys:
            continue
        triplets.append(
            NegationTriplet(
                positive=src,
                negation=f"not {src}",
                antonym=tgt,
                note="from WordNet adjective lemma antonym",
            )
        )
        n_added += 1
    print(
        f"triplets total: {len(triplets)} (seed: {len(SEED_PAIRS)}, +WordNet: {n_added})",
        file=sys.stderr,
    )

    ds = NegationDataset(
        language="en",
        version=args.version,
        source=(
            f"hand-curated seed ({len(SEED_PAIRS)} triplets) + WordNet 3.0 adjective lemma "
            f"antonyms ({n_added} new triplets, seeded subsample of {args.max_pairs}, deduped vs seed); "
            f"negation = 'not X'"
        ),
        triplets=triplets,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
